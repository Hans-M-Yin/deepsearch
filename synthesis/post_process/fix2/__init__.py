"""FVQA hard-case trajectory construction.

This package is intentionally independent from :mod:`synthesis.post_process.fix`.
The latter augments an existing image-search trajectory, while ``fix2`` mines
new FVQA cases, generates a trajectory without exposing its label to the
writer, and applies provenance-aware visual filtering before ShareGPT export.
"""

from .schema import CollectorConfig, FilterConfig, PublicCase, VERSION

__all__ = ["CollectorConfig", "FilterConfig", "PublicCase", "VERSION"]
