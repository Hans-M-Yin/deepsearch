# Documentation of Data Synthesis Pipeline
Inspired by Zhilin Lu, even the coding agent is capable of leading the implementation task without supervision, we need to hold a deeper understanding of what we are doing. 

## 模块定义

### 节点定义
本方法的所有图节点目前可以分为三类，包括
* (1) TextNode 
* (2) ImageNode 
* (3)RegionNode

Node在设计上确保大数据不放进Node，而是放到OSS，或者根本不下载到本地，只保留饮用的URL。

### Firecrawl fixed-key client

`firecrawl_client.py` reads one fixed worker key from `FIRECRAWL_API_KEY` and
does not create or update a local API-key pool or lock.  The request URL alone
selects the route: a configured Relay URL sends the fixed key to the Relay,
where a pool-enabled Relay ignores it and selects the real upstream key; an
official Firecrawl URL uses the fixed key directly.

```python
from synthesis.firecrawl_client import FirecrawlClient

client = FirecrawlClient()
result = client.scrape(
    "https://sanmames.athletic-club.eus/en/blog/why-san-mames-called-that/",
    only_main_content=True,
    max_age=172800000,
    parsers=["pdf"],
    formats=["markdown"],
)
```

### Relay-owned API-key pools

To avoid every worker maintaining its own `/tmp` lock and pool state, point the
worker's provider URL at the Relay.  The worker always sends its fixed API key
header, but a pool-enabled Relay ignores that value and selects/rotates the
real key from its own pool.  Therefore the real key files and pool state must
exist only on the Relay host.

Run exactly one state-owning Relay per provider (or ensure all Relay
instances share a lock service; the local file lock is not a distributed lock
across hosts).  `--keys-file` enables pool selection; without it the Relay
forwards the caller's fixed key for backward compatibility.

### Serper Relay

On the Relay host:

```bash
python debug/serper_relay.py --bind '::' --port 18080 \
  --keys-file /path/to/serper_keys.txt \
  --state-file /path/to/serper_pool_state.json \
  --pool-default-credits 2500 --pool-min-remaining 100 \
  --request-queue-size 512 \
  --max-inflight 128
```

`--request-queue-size` controls the TCP listen backlog before requests enter
the Python handlers.  The relay defaults to 512 (the Python standard-library
default is only 5), which is important for high-concurrency workers.  It can
also be set with `SERPER_RELAY_REQUEST_QUEUE_SIZE`.

`--max-inflight` limits the number of requests being forwarded to Serper at
the same time.  Excess requests wait in the handler threads.  It defaults to
128 and can also be set with `SERPER_RELAY_MAX_INFLIGHT`; tune it separately
from the TCP listen backlog based on the relay's egress capacity.

On every worker, point all three Serper endpoints at the Relay and configure
one fixed `SERPER_API_KEY`:

```bash
export SERPER_API_KEY=relay-worker
export SERPER_SEARCH_URL='http://[relay-ipv6]:18080/search'
export SERPER_IMAGES_URL='http://[relay-ipv6]:18080/images'
export SERPER_LENS_URL='http://[relay-ipv6]:18080/lens'
```

The Relay alone updates `serper_pool_state.json` and its local lock after an
acquisition or an explicit credit-exhaustion response.  If an endpoint is
changed back to `google.serper.dev`, `SERPER_API_KEY` must contain a real
Serper key.  When Serper reports credit exhaustion, the Relay disables that
key and retries the same request with the next candidate, up to the number of
currently available keys.

This change only removes worker-side API-key-pool state/lock updates.  The
per-worker Firecrawl Browser session state and its lock are separate and are
still used to lease/reuse that worker's remote Browser sessions.

### Firecrawl Relay

When a worker cannot reach `api.firecrawl.dev` directly, run
`debug/firecrawl_relay.py` on a node with public egress.  The worker can then
use the existing Firecrawl key pool through the Relay without changing SFT or
RL tool code.  The same Relay proxies both text scrape and Browser session
requests, so Browser image downloads also use the public-egress node:

```bash
# On the public-egress/state-owning node
python debug/firecrawl_relay.py --bind '::' --port 18081 --max-inflight 8 \
  --request-queue-size 512 \
  --keys-file /path/to/firecrawl_keys.txt \
  --state-file /path/to/firecrawl_pool_state.json \
  --pool-default-credits 10000 \
  --browser-timeout-s 150 --browser-image-attempts 2

# On every worker
export FIRECRAWL_RELAY_URL='http://[fdbd:dccd:cde2:1701:3e21:640a:cbaf:b8a3]:18081'
export FIRECRAWL_RELAY_TIMEOUT_S=180
export FIRECRAWL_BROWSER_RELAY_TIMEOUT_S=300
export FIRECRAWL_API_KEY=relay-worker
```

The relay's TCP listen backlog defaults to 512; override it with
`--request-queue-size` or `FIRECRAWL_RELAY_REQUEST_QUEUE_SIZE` when running a
larger concurrency test. This is separate from `--max-inflight`, which limits
requests forwarded to Firecrawl.

The Relay exposes `GET /healthz`, `POST /v2/scrape`, and the Browser lifecycle
routes `POST /v2/browser`, `POST /v2/browser/{session_id}/execute`, and
`DELETE /v2/browser/{session_id}`.  The worker always sends its fixed key;
when the Relay is started with `--keys-file`, the Relay ignores that key and
selects the real key for text requests and keeps the real key/session
association for Browser requests.  Browser image
execution is marked as idempotent and receives its own retry budget; session
creation defaults to one attempt to avoid duplicate remote sessions after an
ambiguous timeout.  Explicit credit exhaustion during Browser session creation
also rotates to the next available key.  An already-created Browser session
cannot be safely rebound to another key during execute/delete, so those
operations disable the exhausted key and return the provider error.  Relay
authentication is intentionally disabled; use a
private network, firewall, or HTTPS if the endpoint is not meant to be openly
reachable.

The Browser session-to-key map is held in Relay memory.  If the Firecrawl
Relay is restarted, workers should also restart their Browser session manager
or allow it to create fresh sessions before issuing Browser execute/delete
requests.

Retry controls:

* text: `--text-upstream-attempts`, `--text-timeout-s`,
  `--text-retry-delay-s`;
* Browser image execution: `--browser-image-attempts`;
* other Browser execute calls: `--browser-execute-attempts`;
* Browser session creation/deletion: `--browser-create-attempts`,
  `--browser-delete-attempts`;
* Browser request timeout/backoff: `--browser-timeout-s`,
  `--browser-retry-delay-s`.


除了以上节点类以外，我们定义`Edge`类来描述节点之间的关系，


### 节点创建
节点主要有Builder实现，WikiTextBuilder的使用场景是当拿到一个Wiki的URL之后，怎么根据这个URL的页面（或者说Entity），构建出一个完整的节点，并且扩展出邻居节点，以及初始化`ImageNode`的扩展。

#### WikiTextBuilder (`wiki_text_builder.py`)
读取页面，并规范化页面内容，根据页面内容抓取信息，以填充`TextNode`，并保留这一过程中生成的`Evidence`和`Snapshot`。

核心函数：

```python

def build_from_url(     
   url: str,
    *,
    title: str | None = None,
    run_id: str | None = None,
    persist: bool = True
)
```
1. 输入url之后，首先调用`reader.read(url)`（默认是`EnhancedReaderClient`），得到一个`ReaderDocument`对象，包含`url`, `title`, `content`, `raw`等内容。
其中`CONTENT`是页面主体markdown格式文件。

2. 随后会进行一个规范化操作，用来规范`url`（去掉那种一大坨乱码的东西），并从`ReaderDocument`中提取`title`并规范化成真正给Node使用的标题。

3. 随后创建`SnapShot`，用来记录这个页面如何被创建，当前是使用：
```python
    snapshot = SearchSnapshot.create(
        SearchEngine.JINA_READER,
        query=page_url,
        request={"url": page_url, "reader": self.reader.__class__.__name__},
        response_preview=document.content[:2000],
        result_count=1 if document.content else 0,
        status_code=200,
        run_id=run_id,
        metadata={"raw": document.raw},
    )
```
即：Reader原始输出不保留到Node中（服从Node设计原则），而是作为trace信息挂在snapshot里。
4. 创建`TextNode`
5. 扩展节点，需要注意，`EnhancedReaderClient`返回了两种content，一种是根据8003端口抓取的、带缓存的未经过清洗的content（`raw_markdown`)，另一种是从8004抓取的清洗过后的content (`content`)。我们用前者获取页面内的超链接，用后者构建节点的语义信息。
需要注意的是，这里扩展节点时进行了超链接的过滤（`wiki_text_builder.extract_wiki_links`函数，只接受wiki的超链接)，并加入到等待扩展的队列
