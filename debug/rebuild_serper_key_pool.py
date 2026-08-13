"""Validate every usable key in the fixed Serper pool, one at a time.

This is intentionally a sequential debug utility.  It uses the normal
``SerperSearchClient`` request path (including the configured relay URL and
relay token), but positions the pool cursor so that each configured key is
tested once.  HTTP 401/403 responses are treated as invalid credentials for
this validation run and the corresponding key is disabled.

Run this only after stopping other Serper clients that share the state file.
The script never prints complete API keys.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from synthesis.search_client import SerperApiKeyPool, SerperSearchClient


_HTTP_STATUS_RE = re.compile(r"HTTP\s+(\d{3})\b", re.IGNORECASE)


def _mask_key(key: str) -> str:
    return SerperApiKeyPool._mask_key(key)


def _status_from_error(error: BaseException) -> int | None:
    match = _HTTP_STATUS_RE.search(str(error))
    return int(match.group(1)) if match else None


def _read_state(path: Path, *, retries: int = 10, delay_s: float = 0.5) -> dict[str, Any]:
    """Read state, tolerating a short HDFS visibility window but not corruption."""

    if not path.exists():
        return {}
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            raw = path.read_text(encoding="utf-8")
            if not raw.strip():
                raise ValueError("state file is empty")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("state must be a JSON object")
            return value
        except Exception as exc:  # noqa: BLE001 - retry transient HDFS visibility issues
            last_error = exc
            if attempt + 1 < max(1, retries):
                time.sleep(max(0.0, delay_s))
    raise RuntimeError(
        f"Serper state is not valid JSON after {max(1, retries)} attempts; "
        f"refusing to rebuild it: {path}: {last_error}"
    ) from last_error


def _prepare_target(pool: SerperApiKeyPool, key: str, predecessor_id: str) -> dict[str, Any]:
    """Re-enable a key with remaining credit and position it next in rotation."""

    key_id = pool.key_id(key)

    def update(state: dict[str, Any]) -> dict[str, Any]:
        state = pool._initialize_state(state)
        record = dict(state["keys"].get(key_id) or {})
        remaining = int(record.get("remaining_credits") or 0)
        if remaining > pool.min_remaining:
            # A previous validation may have disabled this key.  Re-enable it
            # for this explicit test; an invalid response will disable it again.
            record["disabled"] = False
            record.pop("disabled_reason", None)
            record.pop("disabled_at", None)
            state["keys"][key_id] = record
        state["last_selected_key_id"] = predecessor_id
        pool._store_pool_status(state)
        return state

    return pool._with_locked_state(update)


def _record_for(path: Path, key_id: str) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        return dict((state.get("keys") or {}).get(key_id) or {})
    except Exception:
        return {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="OpenAI", help="Search query used for every key.")
    parser.add_argument("--limit", type=int, default=1, help="Number of results requested per call.")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--sleep-s", type=float, default=0.0, help="Delay between key checks.")
    parser.add_argument("--state-read-retries", type=int, default=10)
    parser.add_argument("--state-read-delay-s", type=float, default=0.5)
    parser.add_argument(
        "--search-url",
        default=os.environ.get("SERPER_SEARCH_URL") or "https://google.serper.dev/search",
        help="Serper search URL; defaults to SERPER_SEARCH_URL.",
    )
    parser.add_argument("--hl", default="en")
    parser.add_argument("--ipv6-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    pool = SerperApiKeyPool.from_fixed_pool()
    _read_state(
        pool.state_path,
        retries=args.state_read_retries,
        delay_s=args.state_read_delay_s,
    )  # fail closed instead of silently resetting a corrupt state

    keys = list(pool.keys)
    key_ids = [pool.key_id(key) for key in keys]
    if not keys:
        print("No Serper keys configured.", file=sys.stderr)
        return 2

    # Use the normal client with the complete pool.  We only move its cursor
    # before each request; this preserves the normal state format and rotation.
    client = SerperSearchClient(
        api_keys=keys,
        search_url=args.search_url,
        timeout_s=args.timeout_s,
        pool_state_path=pool.state_path,
        pool_min_remaining=pool.min_remaining,
        ipv6_only=args.ipv6_only,
    )

    tested = valid = invalid = transient = skipped = 0
    print(
        f"Serper key validation: keys={len(keys)} url={args.search_url} "
        f"min_remaining={pool.min_remaining} query={args.query!r}"
    )

    for index, (key, key_id) in enumerate(zip(keys, key_ids), start=1):
        predecessor_id = key_ids[(index - 2) % len(key_ids)]
        prepared = _prepare_target(pool, key, predecessor_id)
        record = dict((prepared.get("keys") or {}).get(key_id) or {})
        remaining = int(record.get("remaining_credits") or 0)
        masked = _mask_key(key)
        if remaining <= pool.min_remaining:
            skipped += 1
            print(
                f"[{index:03d}/{len(keys):03d}] SKIP key={masked} "
                f"remaining={remaining} threshold={pool.min_remaining}",
                flush=True,
            )
            continue

        tested += 1
        started = time.perf_counter()
        try:
            response = client.search_text(args.query, limit=max(1, args.limit), hl=args.hl)
            metadata = response.metadata.get("serper_key_pool") if response.metadata else {}
            used_key_id = str((metadata or {}).get("key_id") or key_id)
            used_record = _record_for(pool.state_path, used_key_id)
            valid += 1
            print(
                f"[{index:03d}/{len(keys):03d}] OK   key={masked} status={response.status_code} "
                f"remaining={used_record.get('remaining_credits')} "
                f"elapsed_s={time.perf_counter() - started:.2f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - each key must be tested independently
            status = _status_from_error(exc)
            if status in {401, 403}:
                pool.mark_credits_exhausted(key_id, reason=f"validation_invalid_key_http_{status}")
                invalid += 1
                outcome = "INVALID"
            else:
                transient += 1
                outcome = "RETRYABLE_FAIL"
            record = _record_for(pool.state_path, key_id)
            print(
                f"[{index:03d}/{len(keys):03d}] {outcome:<14} key={masked} "
                f"status={status or 'n/a'} remaining={record.get('remaining_credits')} "
                f"elapsed_s={time.perf_counter() - started:.2f} error={str(exc)[:240]}",
                flush=True,
            )

        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    status = pool.status()
    print(
        "Summary: "
        f"tested={tested} valid={valid} invalid_disabled={invalid} "
        f"retryable_failures={transient} skipped={skipped} "
        f"available_keys={status['available_key_count']} "
        f"total_keys={status['total_key_count']} "
        f"remaining_credits={status['remaining_credits_total']} "
        f"min_remaining={status['min_remaining']}"
    )
    return 0 if invalid == 0 and transient == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
