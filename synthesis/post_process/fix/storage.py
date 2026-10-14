"""Atomic checkpoints, deterministic identities and stage caches."""

import hashlib
import json
import os
import tempfile
from threading import Lock
from pathlib import Path
from .timing import measure


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class Cache:
    def __init__(self, root):
        self.root = Path(root)
        self._locks = {}
        self._guard = Lock()

    def get_or_compute(self, stage, inputs, compute):
        path = self.root / stage / (digest(inputs) + ".json")
        with self._guard:
            lock = self._locks.setdefault(path, Lock())
        # Same-image requests can race; compute each cache key only once.
        with lock:
            return self._get_or_compute(stage, path, compute)

    def _get_or_compute(self, stage, path, compute):
        hit = path.exists()
        with measure("cache." + stage, cache_hit=hit):
            if hit:
                return read_json(path)
            result = compute()
            write_json(path, result)
            return result
