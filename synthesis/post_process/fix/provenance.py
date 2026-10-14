"""Verify a recovered image's connection to a retained historical page."""

import re
from .trajectory_index import messages, content


def page_alignment(original, recovered, record):
    urls = {
        r.get("source_page_url")
        for r in recovered.resources
        if r.get("kind") == "image" and r.get("source_page_url")
    }
    old_url = original.get("source_page_url")
    if old_url and old_url in urls:
        return True
    page_id = original.get("source_page_id") or original.get("page_id")
    if not page_id:
        return False
    from synthesis.sft.tools import _resource_id

    if any(_resource_id("page", url) == page_id for url in urls):
        return True
    # Evidence must belong to this page observation, not another result/turn.
    observations = [
        content(m)
        for m in messages(record)
        if m.get("from", m.get("role")) in {"observation", "tool"}
        and re.search(r'"page_id"\s*:\s*"' + re.escape(page_id) + r'"', content(m))
    ]
    for observation in observations:
        if any(url in observation for url in urls):
            return True
        # Alamy embeds the stable photo ID in both image URL and page text.
        for resource in recovered.resources:
            url = resource.get("image_url") or ""
            match = re.search(r"https://c\d+\.alamy\.com/comp/([A-Z0-9]+)/", url)
            if match and re.search(
                r"Image\s+ID\s*:\s*" + re.escape(match[1]) + r"\b", observation, re.I
            ):
                return True
    return False
