"""Unit tests for the standalone HTTP image-download relay."""

from __future__ import annotations

import io
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen as client_urlopen
from unittest import mock

from debug import http_relay
from synthesis.sft import tools


class _FakeResponse:
    def __init__(self, body: bytes, *, status_code: int, content_type: str = "image/jpeg") -> None:
        self.body = body
        self.status_code = status_code
        self.headers = {
            "Content-Type": content_type,
            "Location": "",
        }

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def getcode(self) -> int:
        return self.status_code

    def read(self, limit: int = -1) -> bytes:
        if limit == 0:
            return b""
        return self.body if limit < 0 else self.body[:limit]


class HttpRelayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = http_relay.ThreadingHTTPServer(
            ("127.0.0.1", 0), http_relay.HttpRelayHandler
        )
        self.server.timeout_s = 10.0
        self.server.queue_timeout_s = 1.0
        self.server.max_inflight = 2
        self.server.upstream_attempts = 2
        self.server.retry_delay_s = 0.0
        self.server.max_redirects = 3
        self.server.upstream_proxy_url = None
        self.server.upstream_proxy_open = None
        self.server.upstream_slots = threading.BoundedSemaphore(2)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _url(self, path: str) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}{path}"

    def _request(self, target_url: str) -> Request:
        from urllib.parse import quote

        return Request(
            self._url(f"/fetch?url={quote(target_url, safe='')}"),
            headers={
                "Referer": "https://example.com/page",
                "User-Agent": "relay-test",
            },
        )

    def test_wikimedia_file_spec_does_not_duplicate_file_namespace(self) -> None:
        file_page = http_relay._wikimedia_file_spec(
            "https://commons.wikimedia.org/wiki/File:Mosaic_of_Uril.jpg"
        )
        file_path = http_relay._wikimedia_file_spec(
            "https://commons.wikimedia.org/wiki/Special:FilePath/Mosaic_of_Uril.jpg"
        )

        self.assertIsNotNone(file_page)
        self.assertIsNotNone(file_path)
        assert file_page is not None
        assert file_path is not None
        self.assertEqual(file_page["file_title"], "File:Mosaic of Uril.jpg")
        self.assertEqual(file_path["file_title"], "File:Mosaic of Uril.jpg")
        self.assertEqual(file_page["api_url"], "https://commons.wikimedia.org/w/api.php")
        localized = http_relay._wikimedia_file_spec(
            "https://la.wikipedia.org/wiki/Fasciculus:St._Peter's_College_Cambridge_3.jpg"
        )
        self.assertIsNotNone(localized)
        assert localized is not None
        self.assertEqual(
            localized["file_title"],
            "File:St. Peter's College Cambridge 3.jpg",
        )
        self.assertEqual(localized["api_url"], "https://la.wikipedia.org/w/api.php")
        language_path = http_relay._wikimedia_file_spec(
            "https://zh.wikipedia.org/zh-cn/File:Kyivtrebuchet.JPG"
        )
        self.assertIsNotNone(language_path)
        assert language_path is not None
        self.assertEqual(language_path["file_title"], "File:Kyivtrebuchet.JPG")
        self.assertEqual(language_path["api_url"], "https://zh.wikipedia.org/w/api.php")

    def test_get_fetch_decodes_target_and_preserves_binary_response(self) -> None:
        calls: list[tuple[str, str, dict[str, str]]] = []
        target_url = "https://example.com/image.jpg?x=1&signed=a%2Bb"

        def fake_urlopen(request: Request, *, timeout: float) -> _FakeResponse:
            calls.append((request.method, request.full_url, dict(request.header_items())))
            del timeout
            return _FakeResponse(b"\xff\xd8relay-image", status_code=200)

        with mock.patch.object(http_relay, "urlopen", side_effect=fake_urlopen), mock.patch.object(
            http_relay, "_resolve_public_addresses"
        ):
            with client_urlopen(self._request(target_url), timeout=3) as response:
                body = response.read()

        self.assertEqual(body, b"\xff\xd8relay-image")
        self.assertEqual(response.headers["Content-Type"], "image/jpeg")
        self.assertEqual(response.headers["X-Relay-Upstream-Status"], "200")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0:2], ("GET", target_url))
        self.assertEqual(calls[0][2]["Referer"], "https://example.com/page")
        self.assertEqual(calls[0][2]["User-agent"], "relay-test")

    def test_base_prefix_path_form_decodes_target(self) -> None:
        from urllib.parse import quote

        target_url = "https://example.com/image.jpg?x=1&y=2"
        calls: list[str] = []

        def fake_urlopen(request: Request, *, timeout: float) -> _FakeResponse:
            del timeout
            calls.append(request.full_url)
            return _FakeResponse(b"path-image", status_code=200)

        request = Request(
            self._url(f"/{quote(target_url, safe='')}"),
            headers={},
        )
        with mock.patch.object(http_relay, "urlopen", side_effect=fake_urlopen), mock.patch.object(
            http_relay, "_resolve_public_addresses"
        ):
            with client_urlopen(request, timeout=3) as response:
                self.assertEqual(response.read(), b"path-image")
        self.assertEqual(calls, [target_url])

    def test_unicode_target_is_utf8_percent_encoded_upstream(self) -> None:
        from urllib.parse import quote

        target_url = "https://bs.wikipedia.org/wiki/Datoteka:Srpski_rječnik_1818.jpg"
        original_url = "https://upload.wikimedia.org/wikipedia/bs/s/s1/Srpski_rječnik_1818.jpg"
        calls: list[str] = []

        def fake_urlopen(request: Request, *, timeout: float) -> _FakeResponse:
            del timeout
            calls.append(request.full_url)
            if "/w/api.php?" in request.full_url:
                payload = {
                    "query": {
                        "pages": [
                            {
                                "imageinfo": [
                                    {
                                        "url": original_url,
                                        "width": 10,
                                        "height": 10,
                                        "mime": "image/jpeg",
                                    }
                                ]
                            }
                        ]
                    }
                }
                return _FakeResponse(
                    json.dumps(payload).encode("utf-8"),
                    status_code=200,
                    content_type="application/json",
                )
            return _FakeResponse(b"unicode-image", status_code=200, content_type="image/jpeg")

        request = Request(
            self._url(f"/{quote(target_url, safe='')}")
        )
        with mock.patch.object(http_relay, "urlopen", side_effect=fake_urlopen), mock.patch.object(
            http_relay, "_resolve_public_addresses"
        ):
            with client_urlopen(request, timeout=3) as response:
                self.assertEqual(response.read(), b"unicode-image")

        self.assertEqual(len(calls), 2)
        self.assertIn("/w/api.php?", calls[0])
        self.assertEqual(
            calls[1],
            quote(original_url, safe="/:@-._~!$&'()*+,;="),
        )

    def test_transient_upstream_status_is_retried(self) -> None:
        calls: list[str] = []

        def fake_urlopen(request: Request, *, timeout: float) -> _FakeResponse:
            del timeout
            calls.append(request.full_url)
            if len(calls) == 1:
                return _FakeResponse(b"temporary", status_code=502, content_type="text/plain")
            return _FakeResponse(b"ok", status_code=200)

        with mock.patch.object(http_relay, "urlopen", side_effect=fake_urlopen), mock.patch.object(
            http_relay, "_resolve_public_addresses"
        ), mock.patch.object(http_relay.time, "sleep") as sleep:
            with client_urlopen(self._request("https://example.com/retry.jpg"), timeout=3) as response:
                self.assertEqual(response.read(), b"ok")

        self.assertEqual(len(calls), 2)
        sleep.assert_called_once_with(0.0)

    def test_optional_proxy_is_used_for_wikimedia_targets(self) -> None:
        direct_calls: list[str] = []
        proxy_calls: list[str] = []

        def fake_direct(request: Request, *, timeout: float) -> _FakeResponse:
            del timeout
            direct_calls.append(request.full_url)
            return _FakeResponse(b"direct", status_code=200)

        def fake_proxy(request: Request, *, timeout: float) -> _FakeResponse:
            del timeout
            proxy_calls.append(request.full_url)
            return _FakeResponse(b"proxied", status_code=200)

        self.server.upstream_proxy_url = "http://proxy.example:12321"
        self.server.upstream_proxy_open = fake_proxy
        debug_output = io.StringIO()
        with mock.patch.object(http_relay.sys, "stderr", debug_output):
            with mock.patch.object(http_relay, "urlopen", side_effect=fake_direct), mock.patch.object(
                http_relay, "_resolve_public_addresses"
            ), mock.patch.object(http_relay, "_resolve_wikimedia_variant", return_value=None):
                with client_urlopen(
                    self._request("https://commons.wikimedia.org/wiki/Special:FilePath/Test.jpg"),
                    timeout=3,
                ) as response:
                    self.assertEqual(response.read(), b"proxied")

        self.assertEqual(direct_calls, [])
        self.assertEqual(
            proxy_calls,
            ["https://commons.wikimedia.org/wiki/Special:FilePath/Test.jpg"],
        )
        self.assertIn(
            "[http-relay-proxy] url='https://commons.wikimedia.org/wiki/Special:FilePath/Test.jpg' "
            "status_code=200 response_bytes=7",
            debug_output.getvalue(),
        )

    def test_wikimedia_file_svg_is_resolved_without_proxy(self) -> None:
        source_url = "https://commons.wikimedia.org/wiki/File:Flag_of_Pak_Commonwealth_of_Nations.svg"
        original_url = "https://upload.wikimedia.org/wikipedia/commons/e/e6/Flag_of_Pak_Commonwealth_of_Nations.svg"
        svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="600"></svg>'
        direct_calls: list[str] = []

        def fake_direct(request: Request, *, timeout: float) -> _FakeResponse:
            del timeout
            direct_calls.append(request.full_url)
            if "/w/api.php?" in request.full_url:
                payload = {
                    "query": {
                        "pages": [
                            {
                                "imageinfo": [
                                    {
                                        "url": original_url,
                                        "width": 900,
                                        "height": 600,
                                        "mime": "image/svg+xml",
                                    }
                                ]
                            }
                        ]
                    }
                }
                return _FakeResponse(
                    json.dumps(payload).encode("utf-8"),
                    status_code=200,
                    content_type="application/json",
                )
            if request.full_url == original_url:
                return _FakeResponse(svg, status_code=200, content_type="image/svg+xml")
            raise AssertionError(f"unexpected direct URL: {request.full_url}")

        debug_output = io.StringIO()
        with mock.patch.object(http_relay.sys, "stderr", debug_output):
            with mock.patch.object(http_relay, "urlopen", side_effect=fake_direct), mock.patch.object(
                http_relay, "_resolve_public_addresses"
            ):
                with client_urlopen(self._request(source_url), timeout=3) as response:
                    self.assertEqual(response.read(), svg)
                    self.assertEqual(response.headers["Content-Type"], "image/svg+xml")

        self.assertEqual(direct_calls[0].split("/w/api.php?", 1)[0], "https://commons.wikimedia.org")
        self.assertEqual(direct_calls[1], original_url)
        self.assertIn("event=wikimedia_variant_selected", debug_output.getvalue())

    def test_wikimedia_file_page_falls_back_to_special_filepath(self) -> None:
        source_url = "https://commons.wikimedia.org/wiki/File:Fallback.svg"
        filepath_url = "https://commons.wikimedia.org/wiki/Special:FilePath/Fallback.svg"
        svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"></svg>'
        direct_calls: list[str] = []

        def fake_direct(request: Request, *, timeout: float) -> _FakeResponse:
            del timeout
            direct_calls.append(request.full_url)
            if "/w/api.php?" in request.full_url:
                raise http_relay.URLError("temporary API failure")
            if request.full_url == filepath_url:
                return _FakeResponse(svg, status_code=200, content_type="image/svg+xml")
            raise AssertionError(f"unexpected direct URL: {request.full_url}")

        with mock.patch.object(http_relay, "urlopen", side_effect=fake_direct), mock.patch.object(
            http_relay, "_resolve_public_addresses"
        ), mock.patch.object(http_relay.time, "sleep"):
            with client_urlopen(self._request(source_url), timeout=3) as response:
                self.assertEqual(response.read(), svg)
                self.assertEqual(response.headers["Content-Type"], "image/svg+xml")

        self.assertEqual(direct_calls[-1], filepath_url)

    def test_proxy_selects_wikimedia_thumbnail_before_image_download(self) -> None:
        source_url = "https://commons.wikimedia.org/wiki/Special:FilePath/VariantTest.jpg"
        thumbnail_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/v/v1/VariantTest.jpg/1000px-VariantTest.jpg"
        proxy_calls: list[str] = []

        def fake_proxy(request: Request, *, timeout: float) -> _FakeResponse:
            del timeout
            proxy_calls.append(request.full_url)
            if "/w/api.php?" in request.full_url:
                payload = {
                    "query": {
                        "pages": [
                            {
                                "imageinfo": [
                                    {
                                        "url": "https://upload.wikimedia.org/wikipedia/commons/v/v1/VariantTest.jpg",
                                        "width": 4000,
                                        "height": 3000,
                                        "thumburl": thumbnail_url,
                                        "thumbwidth": 1000,
                                        "thumbheight": 750,
                                    }
                                ]
                            }
                        ]
                    }
                }
                return _FakeResponse(
                    json.dumps(payload).encode("utf-8"),
                    status_code=200,
                    content_type="application/json",
                )
            if request.full_url == thumbnail_url:
                return _FakeResponse(b"thumbnail-bytes", status_code=200)
            raise AssertionError(f"unexpected proxy URL: {request.full_url}")

        self.server.upstream_proxy_url = "http://proxy.example:12321"
        self.server.upstream_proxy_open = fake_proxy
        debug_output = io.StringIO()
        with mock.patch.object(http_relay.sys, "stderr", debug_output):
            with mock.patch.object(http_relay, "urlopen") as direct, mock.patch.object(
                http_relay, "_resolve_public_addresses"
            ):
                with client_urlopen(self._request(source_url), timeout=3) as response:
                    self.assertEqual(response.read(), b"thumbnail-bytes")

        direct.assert_not_called()
        self.assertEqual(len(proxy_calls), 2)
        self.assertIn("/w/api.php?", proxy_calls[0])
        self.assertIn("iiurlwidth=1000", proxy_calls[0])
        self.assertEqual(proxy_calls[1], thumbnail_url)
        debug = debug_output.getvalue()
        self.assertIn("event=wikimedia_variant_resolution_start", debug)
        self.assertIn("event=wikimedia_variant_selected", debug)
        self.assertIn("selected_url='https://upload.wikimedia.org/wikipedia/commons/thumb/v/v1/VariantTest.jpg/1000px-VariantTest.jpg'", debug)
        self.assertIn("[http-relay-proxy] url='https://upload.wikimedia.org/wikipedia/commons/thumb/v/v1/VariantTest.jpg/1000px-VariantTest.jpg' status_code=200 response_bytes=15", debug)

    def test_wikimedia_variant_resolution_retries_api_failure(self) -> None:
        source_url = "https://commons.wikimedia.org/wiki/Special:FilePath/RetryTest.jpg"
        thumbnail_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/r/r1/RetryTest.jpg/1000px-RetryTest.jpg"
        imageinfo = {
            "url": "https://upload.wikimedia.org/wikipedia/commons/r/r1/RetryTest.jpg",
            "width": 2000,
            "height": 1500,
            "thumburl": thumbnail_url,
            "thumbwidth": 1000,
            "thumbheight": 750,
        }
        debug_output = io.StringIO()
        with mock.patch.object(
            http_relay,
            "_fetch_wikimedia_imageinfo",
            side_effect=[http_relay.URLError("tunnel timeout"), imageinfo],
        ), mock.patch.object(http_relay.time, "sleep") as sleep, mock.patch.object(
            http_relay.sys, "stderr", debug_output
        ):
            selected = http_relay._resolve_wikimedia_variant(
                source_url,
                request_open=mock.Mock(),
                timeout_s=10.0,
                nat64_prefixes=http_relay.DEFAULT_NAT64_PREFIXES,
                max_long_edge=1000,
                attempts=2,
                retry_delay_s=0.0,
            )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["download_url"], thumbnail_url)
        self.assertEqual(sleep.call_count, 1)
        self.assertIn("event=wikimedia_variant_resolution_retry", debug_output.getvalue())
        self.assertIn("retry_attempt=1", debug_output.getvalue())

    def test_configured_proxy_always_bypasses_non_wikimedia_hosts(self) -> None:
        direct_calls: list[str] = []
        proxy_calls: list[str] = []

        def fake_direct(request: Request, *, timeout: float) -> _FakeResponse:
            del timeout
            direct_calls.append(request.full_url)
            return _FakeResponse(b"non-wikimedia-direct", status_code=200)

        def fake_proxy(request: Request, *, timeout: float) -> _FakeResponse:
            del timeout
            proxy_calls.append(request.full_url)
            return _FakeResponse(b"unexpected-proxy", status_code=200)

        self.server.upstream_proxy_url = "http://proxy.example:12321"
        self.server.upstream_proxy_open = fake_proxy
        debug_output = io.StringIO()
        with mock.patch.object(http_relay.sys, "stderr", debug_output):
            with mock.patch.object(http_relay, "urlopen", side_effect=fake_direct), mock.patch.object(
                http_relay, "_resolve_public_addresses"
            ):
                with client_urlopen(
                    self._request("https://example.com/image.jpg"),
                    timeout=3,
                ) as response:
                    self.assertEqual(response.read(), b"non-wikimedia-direct")

        self.assertEqual(proxy_calls, [])
        self.assertEqual(direct_calls, ["https://example.com/image.jpg"])
        self.assertNotIn("[http-relay-proxy]", debug_output.getvalue())

    def test_proxy_url_validation_is_opt_in_and_rejects_unsupported_schemes(self) -> None:
        self.assertIsNone(http_relay._validate_upstream_proxy_url(None))
        self.assertIsNone(http_relay._validate_upstream_proxy_url(""))
        self.assertEqual(
            http_relay._validate_upstream_proxy_url("http://user:pass@proxy.example:12321"),
            "http://user:pass@proxy.example:12321",
        )
        with self.assertRaises(ValueError):
            http_relay._validate_upstream_proxy_url("socks5://proxy.example:1080")

    def test_private_target_is_rejected_before_upstream_request(self) -> None:
        request = self._request("http://127.0.0.1/private.jpg")
        with mock.patch.object(http_relay, "urlopen") as upstream:
            with self.assertRaises(HTTPError) as context:
                client_urlopen(request, timeout=3)
        self.assertEqual(context.exception.code, 400)
        self.assertIn("blocked", json.loads(context.exception.read())["detail"])
        upstream.assert_not_called()

    def test_nat64_well_known_prefix_checks_embedded_ipv4(self) -> None:
        public_nat64 = "64:ff9b::8cf8:958c"  # embeds public 140.248.149.140
        private_nat64 = "64:ff9b::7f00:0001"  # embeds 127.0.0.1
        with mock.patch.object(
            http_relay.socket,
            "getaddrinfo",
            return_value=[(http_relay.socket.AF_INET6, 0, 0, "", (public_nat64, 443, 0, 0))],
        ):
            http_relay._resolve_public_addresses("www.reddit.com", 443)

        self.assertFalse(http_relay._is_blocked_address(public_nat64))
        self.assertTrue(http_relay._is_blocked_address(private_nat64))

    def test_blocked_redirect_is_non_retryable_client_error(self) -> None:
        calls: list[str] = []

        class _RedirectResponse(_FakeResponse):
            def __init__(self) -> None:
                super().__init__(b"", status_code=302, content_type="text/plain")
                self.headers["Location"] = "https://blocked.example/private.jpg"

        def fake_urlopen(request: Request, *, timeout: float):
            del timeout
            calls.append(request.full_url)
            return _RedirectResponse()

        def fake_resolve(hostname: str, port: int | None, **kwargs: object) -> None:
            del port, kwargs
            if hostname == "blocked.example":
                raise http_relay.TargetValidationError(
                    "target hostname resolves to a blocked address: 'blocked.example' -> 127.0.0.1"
                )

        with mock.patch.object(http_relay, "urlopen", side_effect=fake_urlopen), mock.patch.object(
            http_relay, "_resolve_public_addresses", side_effect=fake_resolve
        ), mock.patch.object(http_relay.time, "sleep") as sleep:
            with self.assertRaises(HTTPError) as context:
                client_urlopen(self._request("https://example.com/image.jpg"), timeout=3)

        self.assertEqual(context.exception.code, 400)
        payload = json.loads(context.exception.read())
        self.assertEqual(payload["error"], "invalid_redirect_target")
        self.assertFalse(payload["retryable"])
        self.assertEqual(len(calls), 1)
        sleep.assert_not_called()

    def test_worker_prefixes_all_binary_download_urls(self) -> None:
        target_url = "https://upload.wikimedia.org/wikipedia/commons/a/a.jpg?x=1&y=2"
        non_wikimedia_url = "https://example.com/a.jpg?x=1&y=2"
        with mock.patch.object(tools, "_HTTP_RELAY_URL", ""):
            self.assertEqual(tools._http_relay_request_url(target_url), target_url)
            self.assertEqual(tools._http_relay_request_url(non_wikimedia_url), non_wikimedia_url)
        with mock.patch.object(tools, "_HTTP_RELAY_URL", "http://relay:18083/"):
            self.assertEqual(
                tools._http_relay_request_url(target_url),
                "http://relay:18083/https%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fcommons%2Fa%2Fa.jpg%3Fx%3D1%26y%3D2",
            )
            self.assertEqual(
                tools._http_relay_request_url(non_wikimedia_url),
                "http://relay:18083/https%3A%2F%2Fexample.com%2Fa.jpg%3Fx%3D1%26y%3D2",
            )
        with mock.patch.object(tools, "_HTTP_RELAY_URL", "http://relay:18083/fetch?url="):
            self.assertEqual(
                tools._http_relay_request_url(target_url),
                "http://relay:18083/fetch?url=https%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fcommons%2Fa%2Fa.jpg%3Fx%3D1%26y%3D2",
            )
            self.assertEqual(
                tools._http_relay_request_url(non_wikimedia_url),
                "http://relay:18083/fetch?url=https%3A%2F%2Fexample.com%2Fa.jpg%3Fx%3D1%26y%3D2",
            )

    def test_worker_suppresses_success_path_wikimedia_debug(self) -> None:
        target_url = "https://upload.wikimedia.org/wikipedia/commons/a/a.jpg"
        response = mock.Mock()
        response.headers = {"Content-Type": "image/jpeg"}
        response.content = b"image-bytes"
        debug_output = io.StringIO()
        with mock.patch.object(
            tools, "_HTTP_RELAY_URL", "http://relay:18083/"
        ), mock.patch.object(
            tools, "_request_with_retry", return_value=response
        ) as request, mock.patch.object(
            tools.sys, "stderr", debug_output
        ):
            payload, content_type = tools._download_binary(
                target_url,
                use_http_relay=True,
            )

        self.assertEqual(payload, b"image-bytes")
        self.assertEqual(content_type, "image/jpeg")
        request_url = request.call_args.args[1]
        self.assertTrue(request_url.startswith("http://relay:18083/"))
        self.assertNotIn("[read_url-proxy] event=using_proxy", debug_output.getvalue())
        self.assertNotIn(f"original_url={target_url!r}", debug_output.getvalue())

    def test_worker_suppresses_success_path_relay_debug(self) -> None:
        target_url = "https://example.com/a.jpg"
        response = mock.Mock()
        response.headers = {"Content-Type": "image/jpeg"}
        response.content = b"image-bytes"
        debug_output = io.StringIO()
        with mock.patch.object(
            tools, "_HTTP_RELAY_URL", "http://relay:18083/"
        ), mock.patch.object(
            tools, "_request_with_retry", return_value=response
        ) as request, mock.patch.object(
            tools.sys, "stderr", debug_output
        ):
            payload, content_type = tools._download_binary(
                target_url,
                use_http_relay=True,
            )

        self.assertEqual(payload, b"image-bytes")
        self.assertEqual(content_type, "image/jpeg")
        self.assertTrue(request.call_args.args[1].startswith("http://relay:18083/"))
        self.assertNotIn("[read_url-relay] event=using_relay", debug_output.getvalue())
        self.assertNotIn("[read_url-proxy] event=using_proxy", debug_output.getvalue())

    def test_wikimedia_backend_override_covers_special_filepath_urls(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "SFT_IMAGE_DOWNLOAD_BACKEND": "firecrawl_browser",
                "SFT_WIKIMEDIA_IMAGE_DOWNLOAD_BACKEND": "http",
            },
            clear=False,
        ):
            self.assertEqual(
                tools._configured_image_download_backend(
                    "https://commons.wikimedia.org/wiki/Special:FilePath/Test.jpg"
                ),
                "http",
            )
            self.assertEqual(
                tools._configured_image_download_backend(
                    "https://upload.wikimedia.org/wikipedia/commons/a/a/Test.jpg"
                ),
                "http",
            )
            self.assertEqual(
                tools._configured_image_download_backend("https://example.com/Test.jpg"),
                "firecrawl_browser",
            )


if __name__ == "__main__":
    unittest.main()
