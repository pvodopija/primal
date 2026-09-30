"""Exact-string edits that keep a file's line endings: edit(path, [(old, new), ...])."""
from pathlib import Path


def edit(path, pairs):
    raw = Path(path).read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    for old, new in pairs:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{path}: expected one match, found {count}: {old[:70]!r}")
        text = text.replace(old, new)
    if crlf:
        text = text.replace("\n", "\r\n")
    Path(path).write_bytes(text.encode("utf-8"))
