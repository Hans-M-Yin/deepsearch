"""Shared URL normalization helpers for HTTP requests and headers."""

from __future__ import annotations

from urllib.parse import quote, unquote, urlparse, urlunparse


def normalize_http_url(url: object) -> str:
    """Return a request-safe URL while preserving URL delimiters.

    Search results can contain raw non-ASCII characters in paths, queries, or
    host names. Normalize those components before passing the URL to urllib,
    Playwright, or another HTTP client. Existing percent escapes are preserved.
    """

    raw_url = str(url or "").strip()
    if not raw_url:
        return raw_url
    if not raw_url.lower().startswith(("http://", "https://")):
        raw_url = f"https://{raw_url}"
    parsed = urlparse(raw_url)
    scheme = (parsed.scheme or "https").lower()
    hostname = parsed.hostname or ""
    try:
        hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        hostname = parsed.hostname or ""
    netloc = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username:
        userinfo = quote(unquote(parsed.username), safe="")
        if parsed.password:
            userinfo += ":" + quote(unquote(parsed.password), safe="")
        netloc = f"{userinfo}@{netloc}"
    path = quote(unquote(parsed.path or "/"), safe="/%:@!$&'()*+,;=-._~")
    query = quote(unquote(parsed.query or ""), safe="=&?/:@!$'()*+,;%-._~")
    fragment = quote(unquote(parsed.fragment or ""), safe="=&?/:@!$'()*+,;%-._~")
    return urlunparse((scheme, netloc, path, "", query, fragment))


def normalize_http_referer(value: object) -> str | None:
    """Return a valid HTTP Referer value, or ``None`` when it is unsafe.

    Referer is optional for image downloads. Omitting a malformed value is
    safer than failing the whole Browser request or invalidating a healthy
    Browser session.
    """

    raw_value = str(value or "").strip()
    if not raw_value:
        return None
    if any(ord(character) < 32 or ord(character) == 127 for character in raw_value):
        return None
    try:
        parsed = urlparse(raw_value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return None
        normalized = normalize_http_url(raw_value)
        if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
            return None
        normalized.encode("latin-1")
    except (TypeError, ValueError, UnicodeError):
        return None
    return normalized or None
