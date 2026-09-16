#!/usr/bin/env bash
# Check the eight self-hosted Jina Reader HTTP workers directly, then compare
# the same request through the public dispatcher on port 8002.
#
# Usage:
#   bash utils/check_reader_workers.sh
#   bash utils/check_reader_workers.sh 'https://catalog.hathitrust.org/Record/006167025'
#
# Optional environment variables:
#   READER_TEST_URL=https://example.com/
#   READER_TEST_CURL_TIMEOUT_S=45
#   READER_TEST_ENGINE=curl            # empty (default) preserves production auto mode
#   READER_TEST_UPSTREAM_TIMEOUT_S=30  # passed as Jina Reader X-Timeout
#   READER_HOST=127.0.0.1
#   READER_DISPATCHER_PORT=8002

set -u -o pipefail

reader_host="${READER_HOST:-127.0.0.1}"
dispatcher_port="${READER_DISPATCHER_PORT:-8002}"
test_url="${1:-${READER_TEST_URL:-https://example.com/}}"
curl_timeout_s="${READER_TEST_CURL_TIMEOUT_S:-45}"
reader_engine="${READER_TEST_ENGINE:-}"
reader_timeout_s="${READER_TEST_UPSTREAM_TIMEOUT_S:-}"
worker_ports=(8102 8112 8122 8132 8142 8152 8162 8172)

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required but was not found in PATH." >&2
  exit 1
fi

if ! [[ "${curl_timeout_s}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "READER_TEST_CURL_TIMEOUT_S must be numeric, got: ${curl_timeout_s}" >&2
  exit 1
fi

if [[ -n "${reader_timeout_s}" && ! "${reader_timeout_s}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "READER_TEST_UPSTREAM_TIMEOUT_S must be numeric, got: ${reader_timeout_s}" >&2
  exit 1
fi

curl_headers=(-H "Accept: text/plain")
if [[ -n "${reader_engine}" ]]; then
  curl_headers+=(-H "X-Engine: ${reader_engine}")
fi
if [[ -n "${reader_timeout_s}" ]]; then
  curl_headers+=(-H "X-Timeout: ${reader_timeout_s}")
fi

probe() {
  local label="$1"
  local endpoint="$2"
  local result

  if result="$(curl -sS --path-as-is --max-time "${curl_timeout_s}" \
    "${curl_headers[@]}" \
    -o /dev/null \
    -w 'code=%{http_code} total_s=%{time_total} connect_s=%{time_connect} ttfb_s=%{time_starttransfer} bytes=%{size_download}' \
    "${endpoint}" 2>&1)"; then
    printf '%-20s OK   %s\n' "${label}" "${result}"
  else
    printf '%-20s FAIL %s\n' "${label}" "${result}"
  fi
}

echo "Reader target: ${test_url}"
echo "Request mode: engine=${reader_engine:-auto/default}, reader_timeout=${reader_timeout_s:-server-default}s, curl_timeout=${curl_timeout_s}s"
echo

echo "Dispatcher stats before probes:"
curl -sS --max-time 3 "http://${reader_host}:${dispatcher_port}/__reader_dispatcher/stats" || true
echo

echo "Listener check:"
for port in "${worker_ports[@]}"; do
  if ss -lnt 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"; then
    printf 'worker=%s LISTEN\n' "${port}"
  else
    printf 'worker=%s NOT_LISTENING\n' "${port}"
  fi
done
echo

echo "Direct worker probes (run concurrently):"
for port in "${worker_ports[@]}"; do
  probe "worker=${port}" "http://${reader_host}:${port}/${test_url}" &
done
wait
echo

echo "Dispatcher probe:"
probe "dispatcher=${dispatcher_port}" "http://${reader_host}:${dispatcher_port}/${test_url}"
echo

echo "Dispatcher stats after probes:"
curl -sS --max-time 3 "http://${reader_host}:${dispatcher_port}/__reader_dispatcher/stats" || true
echo

echo "Interpretation:"
echo "  - One or a few slow/failed workers: inspect their Node/Chrome worker logs."
echo "  - All direct workers slow: the Jina Reader engine or source site is slow."
echo "  - Direct workers fast but dispatcher slow: dispatcher queue/load distribution is the bottleneck."
