"""Validate configured Firecrawl keys and optionally quarantine bad ones.

The script probes one configured key at a time through the normal
``FirecrawlClient`` path.  It uses a temporary state file for the probe, so a
dry run does not alter the shared pool state.  With ``--apply`` it records
successful probe usage and disables only API-level authentication failures.

Run this after stopping other Firecrawl clients that share the state file.
The probe consumes Firecrawl credits (normally at least one per tested key).
The script never prints complete API keys.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from synthesis.firecrawl_client import FirecrawlApiKeyPool, FirecrawlClient


_HTTP_STATUS_RE = re.compile(r"\bHTTP\s+(\d{3})\b", re.IGNORECASE)


def _status_from_exception(error: BaseException) -> int | None:
    for candidate in (error, getattr(error, "response", None)):
        if candidate is None:
            continue
        for attribute in ("status_code", "status", "code"):
            try:
                value = int(getattr(candidate, attribute))
            except (AttributeError, TypeError, ValueError):
                continue
            if 100 <= value <= 599:
                return value
    match = _HTTP_STATUS_RE.search(str(error))
    return int(match.group(1)) if match else None


def _mask_key(key: str) -> str:
    return FirecrawlApiKeyPool._mask_key(key)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Firecrawl state is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Firecrawl state must be a JSON object: {path}")
    return value


def _record(path: Path, key_id: str) -> dict[str, Any]:
    state = _read_json(path)
    value = (state.get("keys") or {}).get(key_id) or {}
    return dict(value) if isinstance(value, dict) else {}


def _prepare_probe_state(pool: FirecrawlApiKeyPool, key: str, path: Path) -> dict[str, Any]:
    """Copy the shared state into a temporary state and enable this key there."""

    state = _read_json(path)
    key_id = pool.key_id(key)
    state = pool._initialize_state(state)
    record = dict(state["keys"].get(key_id) or {})
    record["disabled"] = False
    record["state"] = "active"
    record.pop("disabled_at", None)
    record.pop("disabled_reason", None)
    state["keys"][key_id] = record
    pool._store_pool_status(state)
    return state


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _quarantine_key(
    pool: FirecrawlApiKeyPool,
    key_id: str,
    *,
    reason: str,
    status_code: int | None,
) -> None:
    """Disable a key while preserving its estimated remaining credits."""

    def update(state: dict[str, Any]) -> dict[str, Any]:
        state = pool._initialize_state(state)
        record = dict(state["keys"].get(key_id) or {})
        if not record:
            return state
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record["state"] = "disabled"
        record["disabled"] = True
        record["disabled_at"] = now
        record["disabled_reason"] = reason
        record["last_validation_at"] = now
        record["last_validation_status_code"] = status_code
        state["keys"][key_id] = record
        state["updated_at"] = now
        pool._store_pool_status(state)
        return state

    pool._with_locked_state(update)


def _record_probe_result(
    pool: FirecrawlApiKeyPool,
    key_id: str,
    *,
    success: bool,
    error: str | None,
    credits_used: int,
    status_code: int | None,
) -> None:
    pool.record_result(
        key_id,
        success=success,
        error=error,
        credits_used=credits_used,
        status_code=status_code,
    )


def _response_info(response: dict[str, Any]) -> tuple[int | None, int | None, int, bool]:
    root_status = FirecrawlClient._root_status_code(response)
    metadata = FirecrawlClient._response_metadata(response)
    target_status = FirecrawlClient._non_negative_int(
        metadata.get("statusCode", metadata.get("status_code"))
    )
    credits_used = FirecrawlClient._non_negative_int(
        metadata.get("creditsUsed", metadata.get("credits_used"))
    ) or 0
    success = response.get("success") is True or (
        "success" not in response and "data" in response and target_status in (None, 200)
    )
    return root_status, target_status, credits_used, success


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("FIRECRAWL_VALIDATION_URL") or "https://example.com/",
        help="Small public URL used for the probe; each tested key consumes credits.",
    )
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Probe disabled keys that still have estimated credits remaining.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist probe usage and quarantine API-authentication failures.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    pool = FirecrawlApiKeyPool.from_fixed_pool()
    shared_state = _read_json(pool.state_path)
    initialized = pool._initialize_state(shared_state)
    if args.apply:
        snapshot_path = pool.state_path.with_name(
            f"{pool.state_path.name}.before_validation.{int(time.time())}"
        )
        _write_json(snapshot_path, initialized)
        print(f"State snapshot: {snapshot_path}")

    keys = list(pool.keys)
    if not keys:
        print("No Firecrawl keys configured.", file=sys.stderr)
        return 2

    if os.environ.get("FIRECRAWL_RELAY_URL"):
        print(
            "WARNING: FIRECRAWL_RELAY_URL is set; an HTTP 401 from the Relay can also "
            "mean Relay/upstream configuration failure, not a bad key.",
            file=sys.stderr,
        )

    tested = valid = invalid = ambiguous = transient = skipped = 0
    print(
        f"Firecrawl key validation: keys={len(keys)} url={args.url!r} "
        f"apply={args.apply} state={pool.state_path}"
    )

    for index, key in enumerate(keys, start=1):
        key_id = pool.key_id(key)
        record = dict((initialized.get("keys") or {}).get(key_id) or {})
        remaining = int(record.get("remaining_credits") or 0)
        disabled = bool(record.get("disabled"))
        label = _mask_key(key)
        if remaining <= 0:
            skipped += 1
            print(f"[{index:03d}/{len(keys):03d}] SKIP       key={label} remaining=0", flush=True)
            continue
        if disabled and not args.include_disabled:
            skipped += 1
            print(f"[{index:03d}/{len(keys):03d}] SKIP       key={label} disabled=true", flush=True)
            continue

        tested += 1
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="firecrawl_key_probe_") as directory:
            probe_state = Path(directory) / "state.json"
            _write_json(probe_state, _prepare_probe_state(pool, key, pool.state_path))
            client = FirecrawlClient(api_key=key, pool_state_path=probe_state)
            response: dict[str, Any] | None = None
            error: BaseException | None = None
            try:
                response = client.scrape(
                    args.url,
                    only_main_content=True,
                    max_age=0,
                    formats=["markdown"],
                )
            except Exception as exc:  # noqa: BLE001 - classify each key independently
                error = exc

            probe_record = _record(probe_state, key_id)
            status_from_record = int(probe_record.get("last_status_code") or 0) or None
            status_from_error = _status_from_exception(error) if error is not None else None
            if response is not None:
                root_status, target_status, credits_used, success = _response_info(response)
            else:
                root_status = target_status = None
                credits_used = int(probe_record.get("last_credits_used") or 0)
                success = False

            observed_status = status_from_error or status_from_record or root_status or target_status
            api_auth_failed = bool(probe_record.get("disabled_reason") == "firecrawl_api_auth_failed")
            api_root_failure = root_status in {401, 403} and target_status is None

            if api_auth_failed or root_status == 401:
                outcome = "INVALID"
                invalid += 1
                if args.apply:
                    _quarantine_key(
                        pool,
                        key_id,
                        reason="validation_api_auth_failed",
                        status_code=observed_status,
                    )
            elif api_root_failure:
                # A root-level 403 can be caused by Relay or account policy;
                # do not quarantine it automatically.  Target-page 403s are
                # handled below as target failures as well.
                outcome = "AMBIGUOUS"
                ambiguous += 1
                if args.apply:
                    _record_probe_result(
                        pool,
                        key_id,
                        success=False,
                        error=str((response or {}).get("error") or error or "root-level HTTP 403"),
                        credits_used=credits_used,
                        status_code=observed_status,
                    )
            elif error is not None:
                outcome = "RETRYABLE" if observed_status not in {401, 403} else "AMBIGUOUS"
                if outcome == "RETRYABLE":
                    transient += 1
                else:
                    ambiguous += 1
                if args.apply:
                    _record_probe_result(
                        pool,
                        key_id,
                        success=False,
                        error=str(error),
                        credits_used=credits_used,
                        status_code=observed_status,
                    )
            elif success:
                outcome = "VALID"
                valid += 1
                if args.apply:
                    _record_probe_result(
                        pool,
                        key_id,
                        success=True,
                        error=None,
                        credits_used=credits_used,
                        status_code=target_status or root_status,
                    )
            else:
                outcome = "TARGET_FAIL" if target_status is not None else "UNKNOWN"
                ambiguous += 1
                if args.apply:
                    _record_probe_result(
                        pool,
                        key_id,
                        success=False,
                        error=str((response or {}).get("error") or "unknown validation response"),
                        credits_used=credits_used,
                        status_code=target_status or root_status,
                    )

            print(
                f"[{index:03d}/{len(keys):03d}] {outcome:<10} key={label} "
                f"status={observed_status or 'n/a'} credits={credits_used} "
                f"elapsed_s={time.perf_counter() - started:.2f}",
                flush=True,
            )

        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    status = pool.status()
    print(
        "Summary: "
        f"tested={tested} valid={valid} invalid_quarantined={invalid} "
        f"ambiguous={ambiguous} retryable={transient} skipped={skipped} "
        f"available_keys={status['available_key_count']} "
        f"total_keys={status['total_key_count']} "
        f"remaining_credits={status['remaining_credits_total']} apply={args.apply}"
    )
    return 0 if transient == 0 and ambiguous == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
