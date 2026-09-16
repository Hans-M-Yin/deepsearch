#!/usr/bin/env python3
"""Unauthenticated HTTP fetch relay for image downloads.

The relay is intentionally a small, byte-preserving HTTP proxy for the
worker-side ``read_url`` image path.  A caller sends either::

    GET /fetch?url=https%3A%2F%2Fexample.com%2Fimage.jpg

or::

    POST /fetch
    {"url": "https://example.com/image.jpg"}

The relay downloads the target and returns its raw response body, status code,
content type, and a small amount of provenance in response headers.  It does
not expose a general CONNECT proxy and does not forward arbitrary request
headers.

Run this on a node with the desired public egress::

    python debug/http_relay.py --bind 0.0.0.0 --port 18083

The worker-side Relay endpoint is configured in
``synthesis/sft/tools.py`` via the ``_HTTP_RELAY_URL`` constant.

The normal worker request form is
``http://relay-host:18083/<encoded-target-url>``.
The relay also accepts the explicit ``/fetch?url=...`` form.  The relay blocks private,
loopback, link-local, multicast, and metadata destinations, and validates
every redirect target to avoid turning the service into an SSRF proxy.  DNS64/NAT64
addresses are handled specially: the embedded IPv4 address is checked instead of
rejecting the IPv6 address merely because Python classifies the translation prefix
as reserved.

When ``HTTP_RELAY_UPSTREAM_PROXY_URL`` (or ``--upstream-proxy``) is configured,
that upstream IP proxy is used only for Wikimedia/Wikipedia hostnames.  Other
targets are fetched directly and do not consume proxy traffic.
For Wikimedia file URLs, the relay first queries MediaWiki ``imageinfo`` using
the same egress path as the image request and downloads the largest available
variant whose long edge is within the configured limit (1000 pixels by
default). This resolution also works when the optional upstream proxy is
disabled; in that case it uses the relay's direct egress.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import mimetypes
import os
import re
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer as _StdThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


MAX_REQUEST_BODY_BYTES = 64 * 1024
MAX_RESPONSE_BODY_BYTES = 32 * 1024 * 1024
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_QUEUE_TIMEOUT_S = 60.0
DEFAULT_MAX_INFLIGHT = 8
DEFAULT_UPSTREAM_ATTEMPTS = 2
DEFAULT_RETRY_DELAY_S = 2.0
DEFAULT_MAX_REDIRECTS = 5
DEFAULT_REQUEST_QUEUE_SIZE = 512
DEFAULT_NAT64_PREFIXES = (ipaddress.ip_network("64:ff9b::/96"),)
UPSTREAM_PROXY_ENV = "HTTP_RELAY_UPSTREAM_PROXY_URL"
WIKIMEDIA_MAX_LONG_EDGE_ENV = "HTTP_RELAY_WIKIMEDIA_MAX_LONG_EDGE"
FIRECRAWL_WIKIMEDIA_MAX_LONG_EDGE_ENV = "FIRECRAWL_BROWSER_WIKIMEDIA_MAX_LONG_EDGE"
DEFAULT_WIKIMEDIA_MAX_LONG_EDGE = 1000
_WIKIMEDIA_VARIANT_CACHE_LIMIT = 4096
_WIKIMEDIA_VARIANT_CACHE: dict[tuple[str, int], dict[str, object]] = {}
_WIKIMEDIA_VARIANT_CACHE_LOCK = threading.Lock()
_TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
# ``urllib.request`` ultimately writes the request target through an ASCII
# code path. Keep already-valid percent escapes while quoting Unicode in the
# path/query as UTF-8 before constructing the upstream Request.
_URL_PATH_SAFE = "/%:@-._~!$&'()*+,;="
_URL_QUERY_SAFE = "/%?:@-._~!$&'()*+,;=[]"
_FORWARDED_REQUEST_HEADERS = {
    "User-Agent": "User-Agent",
    "Accept": "Accept",
    "Accept-Language": "Accept-Language",
    "Referer": "Referer",
}
_BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata",
}
_WIKIMEDIA_FILE_NAMESPACE_ALIASES = {
    "archivo",
    "arquivio",
    "arquivo",
    "datoteka",
    "datei",
    "dosiero",
    "fajl",
    "fasciculus",
    "fichier",
    "file",
    "fili",
    "imagine",
    "immagine",
    "plik",
    "soubor",
    "tiedosto",
}


def _is_wikimedia_hostname(hostname: str | None) -> bool:
    normalized = str(hostname or "").rstrip(".").lower()
    return normalized in {"wikimedia.org", "wikipedia.org"} or any(
        normalized.endswith(suffix) for suffix in (".wikimedia.org", ".wikipedia.org")
    )


def _configured_wikimedia_max_long_edge() -> int:
    raw = str(
        os.environ.get(WIKIMEDIA_MAX_LONG_EDGE_ENV)
        or os.environ.get(FIRECRAWL_WIKIMEDIA_MAX_LONG_EDGE_ENV)
        or DEFAULT_WIKIMEDIA_MAX_LONG_EDGE
    ).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_WIKIMEDIA_MAX_LONG_EDGE
    return value if value > 0 else DEFAULT_WIKIMEDIA_MAX_LONG_EDGE


def _wikimedia_file_name_from_wiki_url(url: str) -> str | None:
    """Extract a file name from an English or localized Wikimedia URL."""

    try:
        parsed = urlsplit(str(url or ""))
    except (TypeError, ValueError):
        return None
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if hostname != "commons.wikimedia.org" and not hostname.endswith(".wikipedia.org"):
        return None
    segments = [segment for segment in parsed.path.strip("/").split("/") if segment]
    if not segments:
        return None
    first_segment = segments[0].lower()
    if first_segment == "wiki":
        title_segments = segments[1:]
    else:
        # Some language Wikipedias expose file pages under a language path,
        # for example ``/zh-cn/File:...`` instead of ``/wiki/File:...``.
        # Accept the project language itself and its language-variant forms.
        project_language = hostname.split(".", 1)[0].lower()
        is_language_path = (
            first_segment == project_language
            or first_segment.startswith(f"{project_language}-")
            or bool(re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]+)+", first_segment))
        )
        if not is_language_path:
            return None
        title_segments = segments[1:]
    if len(title_segments) >= 2 and title_segments[0].lower() == "special:filepath":
        filename = unquote("/".join(title_segments[1:]))
    elif title_segments:
        title = unquote("/".join(title_segments)).replace("_", " ").strip()
        namespace, separator, filename = title.partition(":")
        if not separator:
            return None
        namespace = namespace.strip().lower()
        if namespace not in _WIKIMEDIA_FILE_NAMESPACE_ALIASES:
            guessed_type, _ = mimetypes.guess_type(filename)
            if not guessed_type or not guessed_type.startswith("image/"):
                return None
    else:
        return None
    filename = filename.strip()
    if filename.lower().startswith("file:"):
        filename = filename[5:].lstrip()
    return filename or None


def _wikimedia_file_spec(url: str) -> dict[str, str] | None:
    """Extract a MediaWiki API endpoint and file title from an image URL."""

    try:
        parsed = urlsplit(str(url or ""))
    except (TypeError, ValueError):
        return None
    hostname = (parsed.hostname or "").lower().rstrip(".")
    segments = [segment for segment in parsed.path.strip("/").split("/") if segment]
    project = ""
    filename_segment = ""
    if hostname == "upload.wikimedia.org":
        if len(segments) < 5 or segments[0].lower() != "wikipedia":
            return None
        project = segments[1].strip().lower()
        if segments[2].lower() == "thumb":
            if len(segments) < 7:
                return None
            filename_segment = segments[5]
        else:
            filename_segment = segments[4]
    elif hostname == "commons.wikimedia.org" or hostname.endswith(".wikipedia.org"):
        if hostname == "commons.wikimedia.org":
            project = "commons"
        else:
            project = hostname.split(".", 1)[0]
        filename = _wikimedia_file_name_from_wiki_url(url)
        if filename is None:
            return None
        filename_segment = filename
    else:
        return None
    filename = unquote(filename_segment).replace("_", " ").strip()
    if filename.lower().startswith("file:"):
        filename = filename[5:].lstrip()
    if not project or not filename:
        return None
    api_host = "commons.wikimedia.org" if project == "commons" else f"{project}.wikipedia.org"
    return {
        "api_url": f"https://{api_host}/w/api.php",
        "file_title": f"File:{filename}",
    }


def _wikimedia_special_filepath_url(url: str) -> str | None:
    """Convert a Wikimedia ``/wiki/File:...`` page URL to FilePath.

    A File page is HTML, not an image. It is only a last-resort transport
    fallback when imageinfo resolution is unavailable; ``Special:FilePath``
    redirects to the actual upload or thumbnail instead of returning the page
    HTML to the worker.
    """

    try:
        parsed = urlsplit(str(url or ""))
    except (TypeError, ValueError):
        return None
    filename = _wikimedia_file_name_from_wiki_url(url)
    if not filename:
        return None
    filepath = "/wiki/Special:FilePath/" + quote(filename, safe="")
    return urlunsplit((parsed.scheme, parsed.netloc, filepath, parsed.query, ""))


def _wikimedia_variant_widths(max_long_edge: int) -> list[int]:
    widths = [max_long_edge]
    for width in (
        1536,
        1440,
        1280,
        1200,
        1152,
        1080,
        1024,
        960,
        800,
        768,
        720,
        640,
        512,
        480,
        400,
        330,
        320,
        256,
    ):
        if width <= max_long_edge and width not in widths:
            widths.append(width)
    return widths[:6]


def _validate_upstream_proxy_url(proxy_url: str | None) -> str | None:
    """Validate an optional HTTP(S) upstream proxy URL.

    The feature is deliberately opt-in.  An empty value means direct access,
    preserving the historical relay behavior.  SOCKS URLs are rejected because
    urllib's standard ProxyHandler does not support them.
    """

    normalized = str(proxy_url or "").strip()
    if not normalized:
        return None
    parsed = urlsplit(normalized)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("upstream proxy must be an http:// or https:// URL with a hostname")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("upstream proxy URL must not contain a path, query, or fragment")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("upstream proxy URL has an invalid port") from exc
    return normalized


def _build_proxy_urlopen(proxy_url: str):
    """Build a redirect-aware opener that sends requests through one proxy."""

    return build_opener(
        ProxyHandler({"http": proxy_url, "https": proxy_url}),
        _NoRedirectHandler(),
    ).open


class _NoRedirectHandler(HTTPRedirectHandler):
    """Expose redirect responses so each Location can be validated."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        del req, fp, code, msg, headers, newurl
        return None


# Keep this as a module-level symbol so tests and operators can replace it
# without changing the relay implementation.
urlopen = build_opener(_NoRedirectHandler).open


class TargetValidationError(ValueError):
    """A deterministic target/redirect rejection that must not be retried."""


def _json_body(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _read_limited(stream, *, limit: int) -> bytes:
    body = stream.read(limit + 1)
    if len(body) > limit:
        raise ValueError(f"upstream response exceeds the {limit} byte limit")
    return body


def _target_for_log(target_url: str) -> str:
    """Hide query strings because signed URLs may contain credentials."""

    parsed = urlsplit(target_url)
    path = parsed.path or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _is_private_address(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    return bool(
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_unspecified
        or parsed.is_reserved
    )


def _nat64_embedded_ipv4(
    address: ipaddress.IPv6Address,
    nat64_prefixes: tuple[ipaddress.IPv6Network, ...],
) -> ipaddress.IPv4Address | None:
    """Extract an RFC 6052 IPv4 address from a configured NAT64 address.

    RFC 6052 permits /32, /40, /48, /56, /64, and /96 prefixes.  The
    standard 64:ff9b::/96 prefix used by DNS64 is the common case, but keeping
    the extraction generic avoids baking a particular network deployment into
    the SSRF check.
    """

    for prefix in nat64_prefixes:
        if address not in prefix:
            continue
        prefix_length = prefix.prefixlen
        if prefix_length == 96:
            return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
        if prefix_length not in {32, 40, 48, 56, 64}:
            continue

        high_bits = 64 - prefix_length
        low_bits = prefix_length - 32
        high = (int(address) >> (128 - 64)) & ((1 << high_bits) - 1)
        low = (int(address) >> (128 - 72 - low_bits)) & ((1 << low_bits) - 1)
        embedded = (high << low_bits) | low
        return ipaddress.IPv4Address(embedded)
    return None


def _is_blocked_address(
    address: str,
    *,
    nat64_prefixes: tuple[ipaddress.IPv6Network, ...] = DEFAULT_NAT64_PREFIXES,
) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False

    if isinstance(parsed, ipaddress.IPv6Address):
        embedded_ipv4 = _nat64_embedded_ipv4(parsed, nat64_prefixes)
        if embedded_ipv4 is not None:
            # A NAT64 address is safe only when the IPv4 destination encoded
            # inside it is safe under the same SSRF policy.
            return _is_private_address(str(embedded_ipv4))
    return _is_private_address(str(parsed))


def _resolve_public_addresses(
    hostname: str,
    port: int | None,
    *,
    nat64_prefixes: tuple[ipaddress.IPv6Network, ...] = DEFAULT_NAT64_PREFIXES,
) -> None:
    normalized_hostname = hostname.rstrip(".").lower()
    if normalized_hostname in _BLOCKED_HOSTNAMES:
        raise TargetValidationError(f"target hostname is blocked: {hostname!r}")

    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_blocked_address(str(literal), nat64_prefixes=nat64_prefixes):
            raise TargetValidationError(f"target address is blocked: {hostname!r}")
        return

    try:
        addresses = socket.getaddrinfo(
            hostname,
            port or 443,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise TargetValidationError(f"target hostname cannot be resolved: {hostname!r}") from exc
    resolved = {str(item[4][0]) for item in addresses if item[4]}
    if not resolved:
        raise TargetValidationError(f"target hostname has no addresses: {hostname!r}")
    blocked = sorted(
        address
        for address in resolved
        if _is_blocked_address(address, nat64_prefixes=nat64_prefixes)
    )
    if blocked:
        raise TargetValidationError(
            f"target hostname resolves to a blocked address: {hostname!r} -> {blocked[0]}"
        )


def _validate_target_url(
    target_url: str,
    *,
    nat64_prefixes: tuple[ipaddress.IPv6Network, ...] = DEFAULT_NAT64_PREFIXES,
) -> str:
    normalized = str(target_url or "").strip()
    parsed = urlsplit(normalized)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("target URL must use http or https")
    if not parsed.hostname:
        raise ValueError("target URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("target URL credentials are not allowed")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("target URL has an invalid port") from exc
    _resolve_public_addresses(parsed.hostname, port, nat64_prefixes=nat64_prefixes)
    return normalized


def _requote_target_url(target_url: str) -> str:
    """Quote Unicode URL components before passing them to urllib.

    The worker sends the target URL as a percent-encoded Relay path.  The
    handler decodes that wrapper once, which intentionally restores the
    target URL.  A target can therefore contain literal non-ASCII characters
    (for example ``č`` in a Wikimedia filename).  ``urllib.request`` does not
    reliably quote those characters before its HTTP/1.1 request line is
    encoded, so normalize them here while preserving existing percent escapes.
    """

    parsed = urlsplit(target_url)
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            quote(parsed.path or "", safe=_URL_PATH_SAFE),
            quote(parsed.query or "", safe=_URL_QUERY_SAFE),
            "",  # fragments are never sent in an HTTP request target
        )
    )


def _forwarded_headers(incoming_headers) -> dict[str, str]:  # type: ignore[no-untyped-def]
    forwarded: dict[str, str] = {}
    for source_name, target_name in _FORWARDED_REQUEST_HEADERS.items():
        value = str(incoming_headers.get(source_name) or "").strip()
        if value and "\r" not in value and "\n" not in value:
            forwarded[target_name] = value
    # Avoid compressed upstream responses: the relay returns the bytes as-is
    # and does not perform content decoding.
    forwarded["Accept-Encoding"] = "identity"
    return forwarded


def _log_proxy_variant(event: str, **fields: object) -> None:
    rendered: list[str] = []
    for key, value in fields.items():
        if isinstance(value, str) and (key == "url" or key.endswith("_url")):
            value = _target_for_log(value)
        rendered.append(f"{key}={value!r}" if isinstance(value, str) else f"{key}={value}")
    suffix = " " + " ".join(rendered) if rendered else ""
    print(
        f"[http-relay-proxy] event=wikimedia_variant_{event}{suffix}",
        file=sys.stderr,
        flush=True,
    )


def _positive_int(value: object) -> int | None:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _parse_wikimedia_imageinfo(body: object) -> dict[str, object] | None:
    if not isinstance(body, dict):
        return None
    query = body.get("query")
    pages = query.get("pages") if isinstance(query, dict) else None
    if isinstance(pages, list):
        page = pages[0] if pages else None
    elif isinstance(pages, dict):
        page = next(iter(pages.values()), None)
    else:
        page = None
    if not isinstance(page, dict):
        return None
    imageinfo = page.get("imageinfo")
    info = imageinfo[0] if isinstance(imageinfo, list) and imageinfo else None
    return info if isinstance(info, dict) else None


def _fetch_wikimedia_imageinfo(
    *,
    api_url: str,
    file_title: str,
    requested_width: int,
    request_open,
    timeout_s: float,
    nat64_prefixes: tuple[ipaddress.IPv6Network, ...],
) -> dict[str, object] | None:  # type: ignore[no-untyped-def]
    query = urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "titles": file_title,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|thumbmime",
            "iiurlwidth": str(requested_width),
            "iiurlheight": str(requested_width),
            "maxlag": "5",
        }
    )
    metadata_url = _requote_target_url(f"{api_url}?{query}")
    _validate_target_url(metadata_url, nat64_prefixes=nat64_prefixes)
    request = Request(
        metadata_url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": "OpenSearch-VL-HttpRelay/1.0",
        },
        method="GET",
    )
    with request_open(request, timeout=max(1.0, timeout_s)) as response:
        status_code = int(response.getcode() or 200)
        body = _read_limited(response, limit=2 * 1024 * 1024)
    if status_code < 200 or status_code >= 300:
        raise ValueError(f"Wikimedia imageinfo API returned HTTP {status_code}")
    decoded = json.loads(body.decode("utf-8"))
    if isinstance(decoded, dict) and decoded.get("error"):
        error = decoded.get("error")
        code = error.get("code") if isinstance(error, dict) else "api_error"
        raise ValueError(f"Wikimedia imageinfo API error: {code}")
    return _parse_wikimedia_imageinfo(decoded)


def _resolve_wikimedia_variant(
    target_url: str,
    *,
    request_open,
    timeout_s: float,
    nat64_prefixes: tuple[ipaddress.IPv6Network, ...],
    max_long_edge: int,
    attempts: int = DEFAULT_UPSTREAM_ATTEMPTS,
    retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
) -> dict[str, object] | None:  # type: ignore[no-untyped-def]
    file_spec = _wikimedia_file_spec(target_url)
    if file_spec is None:
        return None
    cache_key = (target_url, max_long_edge)
    with _WIKIMEDIA_VARIANT_CACHE_LOCK:
        cached = _WIKIMEDIA_VARIANT_CACHE.get(cache_key)
    if cached is not None:
        _log_proxy_variant(
            "cache_hit",
            requested_url=target_url,
            selected_url=str(cached.get("download_url") or target_url),
            variant_source=str(cached.get("variant_source") or "cache"),
            download_width=cached.get("download_width"),
            download_height=cached.get("download_height"),
        )
        return dict(cached)

    widths = _wikimedia_variant_widths(max_long_edge)
    resolution_attempts = max(1, int(attempts))
    for resolution_attempt in range(1, resolution_attempts + 1):
        _log_proxy_variant(
            "resolution_start",
            requested_url=target_url,
            api_url=file_spec["api_url"],
            max_long_edge=max_long_edge,
            request_widths=widths,
            attempt=resolution_attempt,
            max_attempts=resolution_attempts,
        )
        try:
            for requested_width in widths:
                info = _fetch_wikimedia_imageinfo(
                    api_url=file_spec["api_url"],
                    file_title=file_spec["file_title"],
                    requested_width=requested_width,
                    request_open=request_open,
                    timeout_s=min(timeout_s, 8.0),
                    nat64_prefixes=nat64_prefixes,
                )
                if info is None:
                    continue
                source_width = _positive_int(info.get("width"))
                source_height = _positive_int(info.get("height"))
                source_long_edge = (
                    max(source_width, source_height)
                    if source_width and source_height
                    else None
                )
                selected: dict[str, object] | None = None
                if source_long_edge and source_long_edge <= max_long_edge:
                    selected = {
                        "download_url": str(info.get("url") or target_url),
                        "variant_source": "imageinfo_original",
                        "original_width": source_width,
                        "original_height": source_height,
                        "download_width": source_width,
                        "download_height": source_height,
                    }
                else:
                    thumbnail_url = str(info.get("thumburl") or "").strip()
                    if not thumbnail_url.startswith(("http://", "https://")):
                        continue
                    thumbnail_width = _positive_int(info.get("thumbwidth"))
                    thumbnail_height = _positive_int(info.get("thumbheight"))
                    if not thumbnail_width:
                        match = re.search(r"/(\d+)px-[^/]+(?:\?.*)?$", urlsplit(thumbnail_url).path)
                        if match:
                            thumbnail_width = _positive_int(match.group(1))
                    if not thumbnail_height and thumbnail_width and source_width and source_height:
                        thumbnail_height = max(1, round(thumbnail_width * source_height / source_width))
                    if not thumbnail_width and thumbnail_height and source_width and source_height:
                        thumbnail_width = max(1, round(thumbnail_height * source_width / source_height))
                    thumbnail_long_edge = (
                        max(thumbnail_width, thumbnail_height)
                        if thumbnail_width and thumbnail_height
                        else None
                    )
                    if thumbnail_long_edge and thumbnail_long_edge <= max_long_edge:
                        selected = {
                            "download_url": thumbnail_url,
                            "variant_source": "imageinfo_thumbnail",
                            "original_width": source_width,
                            "original_height": source_height,
                            "download_width": thumbnail_width,
                            "download_height": thumbnail_height,
                        }
                if selected is None:
                    continue
                with _WIKIMEDIA_VARIANT_CACHE_LOCK:
                    if len(_WIKIMEDIA_VARIANT_CACHE) >= _WIKIMEDIA_VARIANT_CACHE_LIMIT:
                        _WIKIMEDIA_VARIANT_CACHE.pop(next(iter(_WIKIMEDIA_VARIANT_CACHE)))
                    _WIKIMEDIA_VARIANT_CACHE[cache_key] = dict(selected)
                _log_proxy_variant(
                    "selected",
                    requested_url=target_url,
                    selected_url=str(selected["download_url"]),
                    variant_source=selected["variant_source"],
                    original_width=selected.get("original_width"),
                    original_height=selected.get("original_height"),
                    download_width=selected.get("download_width"),
                    download_height=selected.get("download_height"),
                    attempt=resolution_attempt,
                )
                return selected
            _log_proxy_variant(
                "not_found",
                requested_url=target_url,
                max_long_edge=max_long_edge,
                attempt=resolution_attempt,
            )
            return None
        except Exception as exc:
            if resolution_attempt >= resolution_attempts:
                _log_proxy_variant(
                    "resolution_failed",
                    requested_url=target_url,
                    attempt=resolution_attempt,
                    max_attempts=resolution_attempts,
                    error=str(exc)[:300],
                )
                return None
            sleep_s = _retry_delay(
                attempt=resolution_attempt,
                base_delay_s=retry_delay_s,
            )
            _log_proxy_variant(
                "resolution_retry",
                requested_url=target_url,
                retry_attempt=resolution_attempt,
                max_attempts=resolution_attempts,
                sleep_s=sleep_s,
                error=str(exc)[:300],
            )
            time.sleep(sleep_s)
    return None


def _fetch_once(
    target_url: str,
    *,
    timeout_s: float,
    max_redirects: int,
    request_headers: dict[str, str],
    nat64_prefixes: tuple[ipaddress.IPv6Network, ...] = DEFAULT_NAT64_PREFIXES,
    upstream_proxy_url: str | None = None,
    upstream_proxy_open=None,  # type: ignore[no-untyped-def]
    wikimedia_max_long_edge: int | None = None,
    wikimedia_variant_attempts: int = DEFAULT_UPSTREAM_ATTEMPTS,
    wikimedia_variant_retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
) -> tuple[bytes, int, str, str]:
    current_url = _requote_target_url(target_url)
    file_spec = _wikimedia_file_spec(current_url)
    if file_spec is not None:
        # Variant resolution must also run without an upstream proxy. The
        # worker can still send a Wikimedia File page to the Relay when the
        # proxy is disabled, and fetching that page directly would return
        # text/html instead of the image bytes.
        variant_request_open = (
            upstream_proxy_open
            if upstream_proxy_url and upstream_proxy_open is not None
            else urlopen
        )
        selected_variant = _resolve_wikimedia_variant(
            current_url,
            request_open=variant_request_open,
            timeout_s=timeout_s,
            nat64_prefixes=nat64_prefixes,
            max_long_edge=wikimedia_max_long_edge or _configured_wikimedia_max_long_edge(),
            attempts=wikimedia_variant_attempts,
            retry_delay_s=wikimedia_variant_retry_delay_s,
        )
        if selected_variant is not None:
            current_url = _requote_target_url(str(selected_variant["download_url"]))
        else:
            filepath_url = _wikimedia_special_filepath_url(current_url)
            if filepath_url is not None:
                _log_proxy_variant(
                    "resolution_fallback",
                    requested_url=current_url,
                    fallback_url=filepath_url,
                    reason="imageinfo_unavailable",
                )
                current_url = _requote_target_url(filepath_url)
    for redirect_count in range(max_redirects + 1):
        current_url = _validate_target_url(current_url, nat64_prefixes=nat64_prefixes)
        current_url = _requote_target_url(current_url)
        request = Request(current_url, headers=request_headers, method="GET")
        current_hostname = urlsplit(current_url).hostname
        use_proxy = bool(
            upstream_proxy_url
            and upstream_proxy_open is not None
            and _is_wikimedia_hostname(current_hostname)
        )
        request_open = upstream_proxy_open if use_proxy else urlopen
        try:
            with request_open(request, timeout=timeout_s) as response:
                status_code = int(response.getcode() or 200)
                if 300 <= status_code < 400:
                    location = str(response.headers.get("Location") or "").strip()
                    response.read(0)
                    if not location:
                        raise ValueError(f"upstream redirect {status_code} has no Location")
                    if redirect_count >= max_redirects:
                        raise ValueError("upstream redirect limit exceeded")
                    current_url = _requote_target_url(urljoin(current_url, location))
                    continue
                body = _read_limited(response, limit=MAX_RESPONSE_BODY_BYTES)
                content_type = str(response.headers.get("Content-Type") or "application/octet-stream")
                if use_proxy:
                    _log_proxy_response(
                        target_url=current_url,
                        status_code=status_code,
                        byte_count=len(body),
                    )
                return body, status_code, content_type, current_url
        except HTTPError as exc:
            status_code = int(exc.code)
            if 300 <= status_code < 400:
                location = str(exc.headers.get("Location") or "").strip()
                if not location:
                    raise ValueError(f"upstream redirect {status_code} has no Location") from exc
                if redirect_count >= max_redirects:
                    raise ValueError("upstream redirect limit exceeded") from exc
                current_url = _requote_target_url(urljoin(current_url, location))
                continue
            body = _read_limited(exc, limit=MAX_RESPONSE_BODY_BYTES)
            content_type = str(exc.headers.get("Content-Type") or "application/octet-stream")
            if use_proxy:
                _log_proxy_response(
                    target_url=current_url,
                    status_code=status_code,
                    byte_count=len(body),
                )
            return body, status_code, content_type, current_url
    raise ValueError("upstream redirect limit exceeded")


def _retry_delay(*, attempt: int, base_delay_s: float, retry_after: str | None = None) -> float:
    delay = max(0.0, float(base_delay_s)) * (2 ** max(0, attempt - 1))
    if retry_after:
        try:
            delay = max(delay, float(retry_after))
        except (TypeError, ValueError):
            pass
    return delay


def _log_fetch(
    *,
    target_url: str,
    elapsed_s: float,
    attempt: int,
    status_code: int | None = None,
    byte_count: int | None = None,
    error: BaseException | None = None,
) -> None:
    fields = [
        f"target={_target_for_log(target_url)!r}",
        f"elapsed_s={elapsed_s:.3f}",
        f"attempt={attempt}",
    ]
    if status_code is not None:
        fields.append(f"status_code={status_code}")
    if byte_count is not None:
        fields.append(f"response_bytes={byte_count}")
    if error is not None:
        fields.append(f"error_type={error.__class__.__name__}")
        fields.append(f"error={str(error)!r}")
    print("[http-relay-fetch] " + " ".join(fields), file=sys.stderr, flush=True)


def _log_proxy_response(*, target_url: str, status_code: int, byte_count: int) -> None:
    """Log the target and byte count after a response came back via IP proxy."""

    print(
        "[http-relay-proxy] "
        f"url={_target_for_log(target_url)!r} "
        f"status_code={status_code} response_bytes={byte_count}",
        file=sys.stderr,
        flush=True,
    )


class ThreadingHTTPServer(_StdThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[BaseHTTPRequestHandler],
        *,
        request_queue_size: int = DEFAULT_REQUEST_QUEUE_SIZE,
    ) -> None:
        # socketserver.TCPServer defaults this to 5. Set it before the parent
        # constructor calls listen() so high fan-in connections reach handlers.
        if request_queue_size <= 0:
            raise ValueError("request_queue_size must be positive")
        self.request_queue_size = int(request_queue_size)
        super().__init__(server_address, request_handler_class)


class IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    daemon_threads = True
    allow_reuse_address = True


class HttpRelayHandler(BaseHTTPRequestHandler):
    server_version = "OpenSearch-VL-HttpRelay/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[http-relay] {self.address_string()} {fmt % args}\n")
        sys.stderr.flush()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path == "/healthz":
            self._write_json(
                200,
                {
                    "ok": True,
                    "service": "http-relay",
                    "max_inflight": self.server.max_inflight,
                    "nat64_prefixes": [
                        str(prefix) for prefix in getattr(self.server, "nat64_prefixes", DEFAULT_NAT64_PREFIXES)
                    ],
                    "upstream_proxy_enabled": bool(
                        getattr(self.server, "upstream_proxy_url", None)
                    ),
                    "upstream_proxy_included_hosts": ["*.wikimedia.org", "*.wikipedia.org"],
                    "wikimedia_max_long_edge": getattr(
                        self.server,
                        "wikimedia_max_long_edge",
                        _configured_wikimedia_max_long_edge(),
                    ),
                },
            )
            return
        if path == "/fetch":
            query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
            values = query.get("url") or []
            if len(values) != 1:
                self._write_json(400, {"error": "request_query_requires_one_url"})
                return
            self._handle_fetch(str(values[0]))
            return
        # The worker's compact mode is Relay-base + encoded target URL.  It
        # avoids requiring a second switch in the worker downloader while
        # still preserving the target's query string losslessly.
        encoded_target = path.lstrip("/")
        if not encoded_target:
            self._write_json(404, {"error": "not_found"})
            return
        self._handle_fetch(unquote(encoded_target))

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if urlsplit(self.path).path != "/fetch":
            self._write_json(404, {"error": "not_found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > MAX_REQUEST_BODY_BYTES:
            self._write_json(400, {"error": "invalid_content_length"})
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._write_json(400, {"error": "request_body_must_be_json"})
            return
        if not isinstance(payload, dict):
            self._write_json(400, {"error": "request_body_must_be_object"})
            return
        self._handle_fetch(str(payload.get("url") or ""))

    def _handle_fetch(self, target_url: str) -> None:
        try:
            target_url = _validate_target_url(
                target_url,
                nat64_prefixes=getattr(self.server, "nat64_prefixes", DEFAULT_NAT64_PREFIXES),
            )
        except ValueError as exc:
            self._write_json(400, {"error": "invalid_target_url", "detail": str(exc)})
            return

        slots = self.server.upstream_slots
        if not slots.acquire(timeout=self.server.queue_timeout_s):
            self._write_json(503, {"error": "http_relay_busy"})
            return
        try:
            self._fetch_and_write(target_url)
        finally:
            slots.release()

    def _fetch_and_write(self, target_url: str) -> None:
        attempts = max(1, int(self.server.upstream_attempts))
        headers = _forwarded_headers(self.headers)
        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            started_at = time.perf_counter()
            try:
                body, status_code, content_type, final_url = _fetch_once(
                    target_url,
                    timeout_s=self.server.timeout_s,
                    max_redirects=self.server.max_redirects,
                    request_headers=headers,
                    nat64_prefixes=getattr(self.server, "nat64_prefixes", DEFAULT_NAT64_PREFIXES),
                    upstream_proxy_url=getattr(self.server, "upstream_proxy_url", None),
                    upstream_proxy_open=getattr(self.server, "upstream_proxy_open", None),
                    wikimedia_max_long_edge=getattr(
                        self.server,
                        "wikimedia_max_long_edge",
                        _configured_wikimedia_max_long_edge(),
                    ),
                    wikimedia_variant_attempts=getattr(
                        self.server,
                        "upstream_attempts",
                        DEFAULT_UPSTREAM_ATTEMPTS,
                    ),
                    wikimedia_variant_retry_delay_s=getattr(
                        self.server,
                        "retry_delay_s",
                        DEFAULT_RETRY_DELAY_S,
                    ),
                )
                if status_code in _TRANSIENT_STATUS_CODES and attempt < attempts:
                    _log_fetch(
                        target_url=target_url,
                        elapsed_s=time.perf_counter() - started_at,
                        attempt=attempt,
                        status_code=status_code,
                        byte_count=len(body),
                        error=RuntimeError("transient upstream HTTP status"),
                    )
                    time.sleep(
                        _retry_delay(
                            attempt=attempt,
                            base_delay_s=self.server.retry_delay_s,
                        )
                    )
                    continue
                _log_fetch(
                    target_url=target_url,
                    elapsed_s=time.perf_counter() - started_at,
                    attempt=attempt,
                    status_code=status_code,
                    byte_count=len(body),
                )
                self._write_bytes(
                    status_code,
                    body,
                    content_type,
                    extra_headers={
                        "X-Relay-Upstream-Status": str(status_code),
                        "X-Relay-Final-URL": final_url,
                    },
                )
                return
            except (URLError, TimeoutError, OSError) as exc:
                last_error = exc
                _log_fetch(
                    target_url=target_url,
                    elapsed_s=time.perf_counter() - started_at,
                    attempt=attempt,
                    error=exc,
                )
                if attempt < attempts:
                    time.sleep(_retry_delay(attempt=attempt, base_delay_s=self.server.retry_delay_s))
                    continue
                status_code = 504 if "timed out" in str(exc).lower() or isinstance(exc, TimeoutError) else 502
                self._write_json(
                    status_code,
                    {
                        "ok": False,
                        "error": "upstream_http_fetch_failed",
                        "error_type": exc.__class__.__name__,
                        "detail": str(getattr(exc, "reason", exc)),
                    },
                )
                return
            except TargetValidationError as exc:
                last_error = exc
                _log_fetch(
                    target_url=target_url,
                    elapsed_s=time.perf_counter() - started_at,
                    attempt=attempt,
                    error=exc,
                )
                # A redirect to a blocked/private destination is deterministic
                # and must not become a retryable 5xx for the outer tool.
                self._write_json(
                    400,
                    {
                        "ok": False,
                        "error": "invalid_redirect_target",
                        "detail": str(exc),
                        "retryable": False,
                    },
                )
                return
            except ValueError as exc:
                last_error = exc
                _log_fetch(
                    target_url=target_url,
                    elapsed_s=time.perf_counter() - started_at,
                    attempt=attempt,
                    error=exc,
                )
                self._write_json(502, {"ok": False, "error": str(exc)})
                return

        self._write_json(
            502,
            {
                "ok": False,
                "error": "upstream_http_fetch_failed",
                "detail": str(last_error or "no response"),
            },
        )

    def _write_json(self, status_code: int, payload: object) -> None:
        self._write_bytes(status_code, _json_body(payload), "application/json")

    def _write_bytes(
        self,
        status_code: int,
        body: bytes,
        content_type: str,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for name, value in (extra_headers or {}).items():
            if "\r" not in value and "\n" not in value:
                self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="::", help="Bind address; default is all IPv6 interfaces.")
    parser.add_argument("--port", type=int, default=18083, help="Listen port; default is 18083.")
    parser.add_argument(
        "--request-queue-size",
        type=int,
        default=int(
            os.environ.get("HTTP_RELAY_REQUEST_QUEUE_SIZE")
            or DEFAULT_REQUEST_QUEUE_SIZE
        ),
        help=(
            "TCP listen backlog before requests enter Python handlers "
            f"(default: {DEFAULT_REQUEST_QUEUE_SIZE}; Python's default is 5)."
        ),
    )
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--queue-timeout-s", type=float, default=DEFAULT_QUEUE_TIMEOUT_S)
    parser.add_argument("--max-inflight", type=int, default=DEFAULT_MAX_INFLIGHT)
    parser.add_argument("--upstream-attempts", type=int, default=DEFAULT_UPSTREAM_ATTEMPTS)
    parser.add_argument("--retry-delay-s", type=float, default=DEFAULT_RETRY_DELAY_S)
    parser.add_argument("--max-redirects", type=int, default=DEFAULT_MAX_REDIRECTS)
    parser.add_argument(
        "--upstream-proxy",
        default=None,
        help=(
            f"Optional HTTP(S) proxy URL for Wikimedia targets only; defaults to "
            f"${UPSTREAM_PROXY_ENV}. Disabled when unset. Non-Wikimedia targets use direct access."
        ),
    )
    parser.add_argument(
        "--nat64-prefix",
        action="append",
        dest="nat64_prefixes",
        help="Additional/replacement RFC 6052 NAT64 prefix; may be repeated or comma-separated.",
    )
    return parser


def _parse_nat64_prefixes(values: list[str] | None) -> tuple[ipaddress.IPv6Network, ...]:
    """Parse CLI or HTTP_RELAY_NAT64_PREFIXES values.

    If neither is provided, use the RFC 6052 well-known prefix.  Supplying a
    value intentionally replaces the default, which lets an operator use a
    network-specific prefix without implicitly trusting another translation
    domain.
    """

    raw_values = values or []
    if not raw_values:
        raw_values = [os.environ.get("HTTP_RELAY_NAT64_PREFIXES", "")]
    parts = [part.strip() for value in raw_values for part in value.split(",") if part.strip()]
    if not parts:
        return DEFAULT_NAT64_PREFIXES

    prefixes: list[ipaddress.IPv6Network] = []
    for value in parts:
        try:
            network = ipaddress.ip_network(value, strict=False)
        except ValueError as exc:
            raise SystemExit(f"invalid NAT64 prefix {value!r}: {exc}") from exc
        if not isinstance(network, ipaddress.IPv6Network):
            raise SystemExit(f"NAT64 prefix must be IPv6, got {value!r}")
        if network.prefixlen not in {32, 40, 48, 56, 64, 96}:
            raise SystemExit(
                f"NAT64 prefix {value!r} must have length 32, 40, 48, 56, 64, or 96"
            )
        prefixes.append(network)
    return tuple(prefixes)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535.")
    if (
        args.timeout_s <= 0
        or args.queue_timeout_s < 0
        or args.retry_delay_s < 0
        or args.request_queue_size <= 0
    ):
        raise SystemExit(
            "timeout and request queue size must be positive; queue timeout and retry delay must be non-negative."
        )
    if args.max_inflight <= 0 or args.upstream_attempts <= 0 or args.max_redirects < 0:
        raise SystemExit("max-inflight and upstream-attempts must be positive; max-redirects must be non-negative.")

    server_class = IPv6ThreadingHTTPServer if ":" in args.bind else ThreadingHTTPServer
    server = server_class(
        (args.bind, args.port),
        HttpRelayHandler,
        request_queue_size=args.request_queue_size,
    )
    server.timeout_s = args.timeout_s
    server.queue_timeout_s = args.queue_timeout_s
    server.max_inflight = args.max_inflight
    server.upstream_attempts = args.upstream_attempts
    server.retry_delay_s = args.retry_delay_s
    server.max_redirects = args.max_redirects
    server.nat64_prefixes = _parse_nat64_prefixes(args.nat64_prefixes)
    server.wikimedia_max_long_edge = _configured_wikimedia_max_long_edge()
    configured_proxy = args.upstream_proxy
    if configured_proxy is None:
        configured_proxy = os.environ.get(UPSTREAM_PROXY_ENV, "")
    try:
        server.upstream_proxy_url = _validate_upstream_proxy_url(configured_proxy)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    server.upstream_proxy_open = (
        _build_proxy_urlopen(server.upstream_proxy_url)
        if server.upstream_proxy_url
        else None
    )
    import threading

    server.upstream_slots = threading.BoundedSemaphore(args.max_inflight)
    print(
        f"HTTP relay listening on {args.bind}:{args.port}; "
        f"max_inflight={args.max_inflight}; timeout_s={args.timeout_s:g}; "
        f"request_queue_size={args.request_queue_size}; "
        f"attempts={args.upstream_attempts}; "
        f"nat64_prefixes={','.join(map(str, server.nat64_prefixes))}; "
        f"wikimedia_max_long_edge={server.wikimedia_max_long_edge}; authentication=disabled",
        flush=True,
    )
    print(
        f"upstream_proxy_enabled={bool(server.upstream_proxy_url)}; "
        "proxy_included_hosts=*.wikimedia.org,*.wikipedia.org",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping HTTP relay.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
