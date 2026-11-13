"""Dense (FAISS) + sparse (BM25) hybrid index over title+abstract. CPU by default."""
import os
import numpy as np


class HybridIndex:
    def __init__(self, records, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        import faiss
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        self.records = records
        self.texts = [f"{r['title']}. {r.get('abstract', '')}" for r in records]
        self.model = SentenceTransformer(model_name, device=os.getenv("EMBED_DEVICE", "cpu"))
        vecs = self.model.encode(self.texts, normalize_embeddings=True, show_progress_bar=False)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(np.asarray(vecs, dtype="float32"))
        self.bm25 = BM25Okapi([t.lower().split() for t in self.texts])

    def search(self, query, k=5, alpha=0.5):
        """Hybrid score = alpha*dense + (1-alpha)*bm25 (both min-max normalised)."""
        q = self.model.encode([query], normalize_embeddings=True)
        dense = self.index.search(np.asarray(q, dtype="float32"), len(self.records))
        d = np.zeros(len(self.records))
        d[dense[1][0]] = dense[0][0]
        b = np.asarray(self.bm25.get_scores(query.lower().split()))

        def mm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-9)
        score = alpha * mm(d) + (1 - alpha) * mm(b)
        top = np.argsort(-score)[:k]
        return [(self.records[i], float(score[i])) for i in top]
