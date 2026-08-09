#!/usr/bin/env python3
"""Compare high-concurrency HTTPS connectivity across several websites.

This test only performs unauthenticated GET requests.  It does not send a
Serper API key and therefore does not consume Serper credits.  HTTP errors
such as 401/404/405 still count as a successful network connection; transport
failures are reported separately.

Examples:
  python debug/stress_test_websites.py
  python debug/stress_test_websites.py --requests 128 --workers 32
  python debug/stress_test_websites.py --direct --timeout-s 20
  python debug/stress_test_websites.py --direct --ipv6-only --hosts google.serper.dev/search
  python debug/stress_test_websites.py --hosts example.com,www.cloudflare.com
"""

from __future__ import annotations

import argparse
import http.client
import json
import socket
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPHandler, HTTPSHandler, ProxyHandler, Request, build_opener


DEFAULT_HOSTS = [
    "example.com",
    "www.cloudflare.com",
    "www.wikipedia.org",
    "httpbin.org/get",
    "google.serper.dev/search",
]


def _create_ipv6_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
    """Create a TCP connection using only AF_INET6 DNS results."""

    host, port = address
    errors: list[OSError] = []
    infos = socket.getaddrinfo(host, port, socket.AF_INET6, socket.SOCK_STREAM)
    for family, socktype, proto, _canonname, sockaddr in infos:
        sock = socket.socket(family, socktype, proto)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            errors.append(exc)
            sock.close()
    if errors:
        raise errors[-1]
    raise OSError(f"No IPv6 address found for {host!r}")


class _IPv6HTTPConnection(http.client.HTTPConnection):
    _create_connection = staticmethod(_create_ipv6_connection)


class _IPv6HTTPSConnection(http.client.HTTPSConnection):
    _create_connection = staticmethod(_create_ipv6_connection)


class _IPv6HTTPHandler(HTTPHandler):
    def http_open(self, req):
        return self.do_open(_IPv6HTTPConnection, req)


class _IPv6HTTPSHandler(HTTPSHandler):
    def https_open(self, req):
        return self.do_open(
            _IPv6HTTPSConnection,
            req,
            context=getattr(self, "_context", None),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests",
        type=int,
        default=128,
        help="Number of requests per host (default: 128).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=128,
        help="Maximum concurrent requests per host (default: 128).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of sequential rounds over all hosts (default: 1).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=20.0,
        help="Timeout for each request (default: 20 seconds).",
    )
    parser.add_argument(
        "--hosts",
        default=",".join(DEFAULT_HOSTS),
        help="Comma-separated hostnames or URLs to test.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Ignore HTTP(S)_PROXY environment variables and connect directly.",
    )
    parser.add_argument(
        "--ipv6-only",
        action="store_true",
        help="Force AF_INET6 for DNS resolution and TCP connections.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser


def _normalize_url(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if "://" not in value:
        value = f"https://{value}"
    return value


def _build_opener(*, direct: bool, ipv6_only: bool):
    handlers = []
    if direct:
        handlers.append(ProxyHandler({}))
    if ipv6_only:
        handlers.extend([_IPv6HTTPHandler(), _IPv6HTTPSHandler()])
    return build_opener(*handlers)


def _run_one(*, index: int, url: str, timeout_s: float, opener) -> dict[str, Any]:
    started_at = time.perf_counter()
    request = Request(
        url,
        headers={
            "Accept": "*/*",
            "User-Agent": "OpenSearch-VL-network-stress-test/1.0",
        },
        method="GET",
    )
    try:
        with opener.open(request, timeout=timeout_s) as response:
            # Read a small amount so the request includes response delivery,
            # while avoiding unnecessary bandwidth during the diagnostic.
            response.read(1024)
            status_code = int(response.getcode() or 0)
        return {
            "index": index,
            "ok": True,
            "transport_ok": True,
            "status_code": status_code,
            "elapsed_s": round(time.perf_counter() - started_at, 3),
        }
    except HTTPError as exc:
        # An HTTP error proves that TCP/TLS/HTTP completed successfully.
        return {
            "index": index,
            "ok": True,
            "transport_ok": True,
            "status_code": int(exc.code),
            "elapsed_s": round(time.perf_counter() - started_at, 3),
        }
    except (URLError, TimeoutError, socket.timeout, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return {
            "index": index,
            "ok": False,
            "transport_ok": False,
            "status_code": 0,
            "elapsed_s": round(time.perf_counter() - started_at, 3),
            "error_type": reason.__class__.__name__,
            "error": str(reason),
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic should keep all workers alive
        return {
            "index": index,
            "ok": False,
            "transport_ok": False,
            "status_code": 0,
            "elapsed_s": round(time.perf_counter() - started_at, 3),
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }


def run_host_test(
    *,
    url: str,
    requests: int,
    workers: int,
    timeout_s: float,
    direct: bool,
    ipv6_only: bool,
) -> dict[str, Any]:
    opener = _build_opener(direct=direct, ipv6_only=ipv6_only)
    started_at = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(workers, requests), thread_name_prefix="web-stress") as executor:
        futures = {
            executor.submit(_run_one, index=index, url=url, timeout_s=timeout_s, opener=opener): index
            for index in range(1, requests + 1)
        }
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: int(item["index"]))
    elapsed_values = [float(item["elapsed_s"]) for item in results]
    status_codes = Counter(str(item["status_code"]) for item in results)
    failures = [item for item in results if not item["transport_ok"]]
    return {
        "url": url,
        "requests": requests,
        "workers": min(workers, requests),
        "timeout_s": timeout_s,
        "direct": direct,
        "ipv6_only": ipv6_only,
        "success_transport_count": requests - len(failures),
        "transport_failure_count": len(failures),
        "status_codes": dict(status_codes),
        "error_types": dict(Counter(str(item["error_type"]) for item in failures)),
        "wall_elapsed_s": round(time.perf_counter() - started_at, 3),
        "request_elapsed_s": {
            "min": round(min(elapsed_values), 3) if elapsed_values else 0.0,
            "mean": round(sum(elapsed_values) / len(elapsed_values), 3) if elapsed_values else 0.0,
            "max": round(max(elapsed_values), 3) if elapsed_values else 0.0,
        },
        "failures": failures,
    }


def _print_report(round_index: int, report: dict[str, Any]) -> None:
    print(
        f"[{round_index}] {report['url']} "
        f"requests={report['requests']} workers={report['workers']} "
        f"transport_ok={report['success_transport_count']} "
        f"transport_failed={report['transport_failure_count']} "
        f"wall_s={report['wall_elapsed_s']}"
    )
    print(f"    status_codes={json.dumps(report['status_codes'], sort_keys=True)}")
    print(f"    elapsed_s={json.dumps(report['request_elapsed_s'], sort_keys=True)}")
    if report["error_types"]:
        print(f"    error_types={json.dumps(report['error_types'], sort_keys=True)}")
        for item in report["failures"][:3]:
            print(f"    failure[{item['index']}] {item['error_type']}: {item['error']}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.requests <= 0 or args.workers <= 0 or args.rounds <= 0 or args.timeout_s <= 0:
        raise SystemExit("--requests, --workers, --rounds and --timeout-s must be positive")

    hosts = [_normalize_url(item) for item in str(args.hosts).split(",")]
    hosts = [item for item in hosts if item]
    if not hosts:
        raise SystemExit("--hosts must contain at least one host")

    reports: list[dict[str, Any]] = []
    for round_index in range(1, args.rounds + 1):
        print(f"Starting website connectivity round {round_index}/{args.rounds}")
        for url in hosts:
            report = run_host_test(
                url=url,
                requests=args.requests,
                workers=args.workers,
                timeout_s=args.timeout_s,
                direct=args.direct,
                ipv6_only=args.ipv6_only,
            )
            report["round"] = round_index
            reports.append(report)
            _print_report(round_index, report)

    total_failures = sum(int(report["transport_failure_count"]) for report in reports)
    print(f"Total transport failures: {total_failures}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"reports": reports}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Detailed JSON report: {args.output}")
    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
