"""Plain overlapping-window chunker.

Keeps character offsets so every chunk can be cited back to a specific
span of the source filing, not just "somewhere in the 10-K".
"""
from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[dict]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "start": start, "end": end})
        if end == n:
            break
        start = end - overlap
    return chunks
