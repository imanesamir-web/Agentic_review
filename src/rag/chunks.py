"""Chunking + BM25 retrieval over one paper's full text (pure Python, no GPU)."""
import re


def tok(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def chunk_text(text: str, size: int = 900, overlap: int = 150):
    m = list(re.finditer(r"\b(References|REFERENCES|Bibliography)\b", text))
    if m and m[-1].start() > 0.5 * len(text):  # drop the reference list
        text = text[:m[-1].start()]
    chunks, step = [], max(size - overlap, 1)
    for i in range(0, len(text), step):
        c = text[i:i + size].strip()
        if len(c) > 80:
            chunks.append(c)
    return chunks


class ChunkRetriever:
    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.bm25 = BM25Okapi([tok(c) for c in chunks]) if chunks else None

    def top(self, query: str, k: int = 2):
        if not self.chunks:
            return []
        scores = self.bm25.get_scores(tok(query))
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])[:k]
        if scores[order[0]] <= 0:  # nothing matched: fall back to the first chunks
            return list(range(min(k, len(self.chunks))))
        return order
