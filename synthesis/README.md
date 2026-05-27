# Documentation of Data Synthesis Pipeline
Inspired by Zhilin Lu, even the coding agent is capable of leading the implementation task without supervision, we need to hold a deeper understanding of what we are doing. 

## 模块定义

### 节点定义
本方法的所有图节点目前可以分为三类，包括
* (1) TextNode 
* (2) ImageNode 
* (3)RegionNode

Node在设计上确保大数据不放进Node，而是放到OSS，或者根本不下载到本地，只保留饮用的URL。


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
5. 扩展节点，需要注意，`EnhancedReaderClient`返回了两种content，一种是根据8002端口抓取的，未经过清洗的content（`raw_markdown`)，另一种是从8004抓取的清洗过后的content (`content`)。我们用前者获取页面内的超链接，用后者构建节点的语义信息。
需要注意的是，这里扩展节点时进行了超链接的过滤（`wiki_text_builder.extract_wiki_links`函数，只接受wiki的超链接)，并加入到等待扩展的队列