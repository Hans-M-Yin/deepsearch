"""Conservative metadata alignment allowing formatting changes, not entity guesses."""

import html
import re
import unicodedata


def normalized(value):
    return re.sub(
        r"\W+",
        " ",
        unicodedata.normalize("NFKC", html.unescape(str(value or ""))).casefold(),
    ).strip()


def same_metadata(left, right):
    title, source = normalized(left.get("title")), normalized(left.get("source"))
    return bool(
        title
        and source
        and title == normalized(right.get("title"))
        and source == normalized(right.get("source"))
    )
