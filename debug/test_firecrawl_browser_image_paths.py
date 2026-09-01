#!/usr/bin/env python3
"""Compare repeated Firecrawl Browser APIRequestContext and page navigation.

Run after sourcing the same environment used by RL, for example:

    source RL/.env_remote
    python debug/test_firecrawl_browser_image_paths.py --mode request

The script creates one short-lived session per mode, reuses it for all
repetitions, and removes both sessions before exiting.  It never prints API
keys or image contents; only response metadata
and a small byte prefix are reported.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from synthesis.firecrawl_client import (
    FirecrawlApiKeyPool,
    FirecrawlBrowserSessionManager,
)


DEFAULT_URLS = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/"
    "1_Puerta_de_Bisagra_toledo_2014.jpg/"
    "1280px-1_Puerta_de_Bisagra_toledo_2014.jpg"
    "?utm_source=en.wikipedia.org&utm_campaign=index&utm_content=thumbnail",
    "https://upload.wikimedia.org/wikipedia/commons/9/9e/Pillar_of_Vasco_da_Gama.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/4/40/"
    "Th%C3%A9odore_Chass%C3%A9riau_-_Mesdemoiselles_Chass%C3%A9riau_%28Louvre_RF_2214%29_0000787160_OG.JPG",
    "https://upload.wikimedia.org/wikipedia/commons/3/3f/William_Powell_in_The_Great_Ziegfeld_trailer.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/9/9d/Bette_Davis_in_The_Letter_3.jpg",
)


def _keys_for_diagnostic_pool() -> list[str]:
    env_pool = FirecrawlApiKeyPool.from_env()
    if env_pool is not None:
        return env_pool.keys
    single_key = str(os.environ.get("FIRECRAWL_API_KEY") or "").strip()
    if single_key:
        return [single_key]
    return FirecrawlApiKeyPool.from_fixed_pool().keys


def _make_manager(state_dir: Path) -> FirecrawlBrowserSessionManager:
    return FirecrawlBrowserSessionManager.from_environment(
        api_keys=_keys_for_diagnostic_pool(),
        pool_state_path=state_dir / "api_pool.json",
        session_state_path=state_dir / "browser_sessions.json",
    )


def _diagnostic_code(*, url: str, mode: str, timeout_ms: int) -> str:
    payload = json.dumps(
        {
            "url": url,
            "mode": mode,
            "timeoutMs": max(1000, int(timeout_ms)),
        },
        ensure_ascii=True,
    )
    return f"""
await (async () => {{
const input = {payload};
const headers = {{
  "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
  "Accept-Language": "en-US,en;q=0.9",
  "User-Agent": "Mozilla/5.0 (compatible; OpenSearch-VL-firecrawl-diagnostic/1.0)",
}};
let response = null;
let phase = "request";
const output = {{
  mode: input.mode,
  status: null,
  resolved_url: input.url,
  content_type: "",
  byte_count: 0,
  byte_prefix_hex: "",
  error: null,
  target_error_type: null,
  target_error_message: null,
  target_phase: phase,
}};
try {{
  if (typeof page.setExtraHTTPHeaders === "function") {{
    await page.setExtraHTTPHeaders(headers);
  }}
  if (input.mode === "request") {{
    response = await page.request.get(input.url, {{
      failOnStatusCode: false,
      headers,
      timeout: input.timeoutMs,
    }});
  }} else {{
    response = await page.goto(input.url, {{
      waitUntil: "commit",
      timeout: input.timeoutMs,
    }});
  }}
  phase = "response";
  output.target_phase = phase;
  if (!response) {{
    output.error = "no_response";
  }} else {{
    const responseHeaders = response.headers();
    output.status = response.status();
    output.resolved_url = response.url();
    output.content_type = String(responseHeaders["content-type"] || "")
      .split(";", 1)[0].trim().toLowerCase();
    phase = "body";
    output.target_phase = phase;
    const body = await response.body();
    output.byte_count = body.length;
    output.byte_prefix_hex = body.subarray(0, 16).toString("hex");
    if (output.status < 200 || output.status >= 300) {{
      output.error = `http_status_${{output.status}}`;
    }}
  }}
}} catch (error) {{
  const message = String(error && error.message ? error.message : error || "request failed");
  const normalized = message.toLowerCase();
  output.target_phase = phase;
  output.target_error_type = normalized.includes("timeout") || normalized.includes("timed out")
    ? (input.mode === "request" ? "target_request_timeout" : "browser_navigation_timeout")
    : (input.mode === "request" ? "target_request_error" : "browser_navigation_error");
  output.target_error_message = message.slice(0, 500);
  output.error = output.target_error_type;
}} finally {{
  if (response && typeof response.dispose === "function") {{
    try {{ await response.dispose(); }} catch (_) {{}}
  }}
}}
return JSON.stringify(output);
}})();
""".strip()


def _decode_result(raw: dict[str, Any]) -> dict[str, Any]:
    result = raw.get("result") or raw.get("stdout") or raw.get("output")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return {"raw_result": result[:1000]}
    return result if isinstance(result, dict) else {"raw_result": str(result or "")[:1000]}


def _run_mode(
    manager: FirecrawlBrowserSessionManager,
    *,
    urls: list[str],
    mode: str,
    timeout_s: float,
    acquire_timeout_s: float,
    repeats_per_url: int,
) -> dict[str, Any]:
    timeout_s = max(1.0, min(float(timeout_s), 300.0))
    repeats_per_url = max(1, int(repeats_per_url))
    request_timeout_s = max(1.0, timeout_s - min(5.0, timeout_s * 0.1))
    lease = None
    result: dict[str, Any] = {
        "mode": mode,
        "urls": urls,
        "requested_urls": len(urls),
        "repeats_per_url": repeats_per_url,
        "attempts": [],
    }
    try:
        lease = manager.acquire(
            acquire_timeout_s=acquire_timeout_s,
            lease_timeout_s=max(
                timeout_s * len(urls) * repeats_per_url + 60.0,
                manager.relay_timeout_s + 30.0,
            ),
            request_url=urls[0],
        )
        result["session_id"] = lease.session_id
        result["key_id"] = lease.key_id
        stop = False
        for url_index, url in enumerate(urls, start=1):
            for repeat_index in range(1, repeats_per_url + 1):
                attempt_started_at = time.monotonic()
                attempt_result: dict[str, Any] = {
                    "url_index": url_index,
                    "url": url,
                    "repeat": repeat_index,
                }
                try:
                    raw = manager.execute(
                        lease,
                        code=_diagnostic_code(
                            url=url,
                            mode=mode,
                            timeout_ms=int(request_timeout_s * 1000),
                        ),
                        timeout_s=timeout_s,
                        request_type="browser_image",
                    )
                    attempt_result.update(
                        {
                            "outer_success": raw.get("success"),
                            "outer_killed": raw.get("killed"),
                            "outer_exit_code": raw.get("exit_code", raw.get("exitCode")),
                            "target": _decode_result(raw),
                        }
                    )
                except Exception as exc:
                    attempt_result.update(
                        {
                            "client_exception_type": type(exc).__name__,
                            "client_exception": str(exc)[:1000],
                            "status_code": getattr(exc, "status_code", None),
                        }
                    )
                    stop = True
                attempt_result["elapsed_s"] = round(time.monotonic() - attempt_started_at, 3)
                result["attempts"].append(attempt_result)
                if stop:
                    break
            if stop:
                break

        target_results = [
            item.get("target")
            for item in result["attempts"]
            if isinstance(item.get("target"), dict)
        ]
        result["summary"] = {
            "attempts_completed": len(result["attempts"]),
            "urls_completed": len({
                item["url_index"]
                for item in result["attempts"]
                if isinstance(item.get("target"), dict)
            }),
            "target_successes": sum(
                item.get("status") == 200 and int(item.get("byte_count") or 0) > 0
                for item in target_results
            ),
            "target_errors": sum(bool(item.get("error")) for item in target_results),
            "client_errors": sum("client_exception_type" in item for item in result["attempts"]),
            "target_statuses": {
                str(status): sum(item.get("status") == status for item in target_results)
                for status in sorted({item.get("status") for item in target_results if item.get("status") is not None})
            },
        }
        return result
    except Exception as exc:
        result["client_exception_type"] = type(exc).__name__
        result["client_exception"] = str(exc)[:1000]
        result["status_code"] = getattr(exc, "status_code", None)
        return result
    finally:
        if lease is not None:
            lease.invalidate(reason="diagnostic cleanup")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        dest="single_url",
        default=None,
        help="Test one URL instead of the five built-in Wikimedia URLs.",
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        default=None,
        help="Custom URL list; all URLs are tested on one session per mode.",
    )
    parser.add_argument(
        "--mode",
        choices=("request", "goto", "both"),
        default="request",
        help="Browser path to test; 'both' creates one session per path.",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--acquire-timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--repeats-per-url",
        "--repeats",
        type=int,
        default=1,
        help="Repeat each URL this many times on the same Browser session (alias: --repeats).",
    )
    parser.add_argument(
        "--session-ttl-s",
        type=int,
        default=None,
        help="Diagnostic session TTL; by default it is sized for all repetitions.",
    )
    parser.add_argument("--state-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.urls and args.single_url:
        parser.error("use either --url or --urls, not both")
    urls = list(args.urls or ([args.single_url] if args.single_url else DEFAULT_URLS))
    if not urls:
        parser.error("at least one URL is required")

    temporary = None
    if args.state_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="firecrawl_browser_diag_")
        state_dir = Path(temporary.name)
    else:
        state_dir = args.state_dir
        state_dir.mkdir(parents=True, exist_ok=True)

    try:
        repeats_per_url = max(1, int(args.repeats_per_url))
        total_requests = len(urls) * repeats_per_url
        if args.session_ttl_s is None:
            diagnostic_ttl_s = max(300, min(3600, int(float(args.timeout_s) * total_requests + 120)))
        else:
            diagnostic_ttl_s = max(30, min(3600, int(args.session_ttl_s)))
        os.environ["FIRECRAWL_BROWSER_SESSION_TTL_S"] = str(diagnostic_ttl_s)
        os.environ["FIRECRAWL_BROWSER_SESSION_ACTIVITY_TTL_S"] = str(
            max(300, min(3600, diagnostic_ttl_s))
        )
        manager = _make_manager(state_dir)
        print(
            json.dumps(
                {
                    "urls": urls,
                    "mode": args.mode,
                    "timeout_s": min(float(args.timeout_s), 300.0),
                    "repeats_per_url": repeats_per_url,
                    "session_ttl_s": diagnostic_ttl_s,
                    "relay_configured": bool(
                        os.environ.get("FIRECRAWL_BROWSER_RELAY_URL")
                        or os.environ.get("FIRECRAWL_RELAY_URL")
                    ),
                    "max_sessions": manager.max_sessions,
                    "state_dir": str(state_dir),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        modes = ("request", "goto") if args.mode == "both" else (args.mode,)
        for mode in modes:
            print(
                json.dumps(
                    _run_mode(
                        manager,
                        urls=urls,
                        mode=mode,
                        timeout_s=args.timeout_s,
                        acquire_timeout_s=args.acquire_timeout_s,
                        repeats_per_url=repeats_per_url,
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
    finally:
        if temporary is not None:
            temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
