# src/retrieve.py

from __future__ import annotations
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

from .embed_store import EmbeddingStore
from .config import retrieval_cfg


class HybridRetriever:
    def __init__(self, chunks: List[str], ids: List[str], metadatas: List[Dict[str, Any]]):
        """
        chunks: 所有 chunk 的文字內容
        ids:    對應 chunk 的 id
        metadatas: 對應 metadata
        """
        self.store = EmbeddingStore()
        self.ids = ids
        self.metadatas = metadatas
        self.chunks = chunks

        tokenized = [c.split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def _semantic_retrieve(self, query: str, k: int):
        res = self.store.semantic_search(query, k=k)
        # res["ids"], res["documents"], res["metadatas"], res["distances"]
        return res

    def _keyword_retrieve(self, query: str, k: int):
        scores = self.bm25.get_scores(query.split())
        # 取得 top-k index
        ranked = sorted(
            list(enumerate(scores)),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        return ranked

    def hybrid_retrieve(self, query: str, k: int | None = None):
        if k is None:
            k = retrieval_cfg.TOP_K

        sem_res = self._semantic_retrieve(query, k=k)
        kw_res = self._keyword_retrieve(query, k=k)

        sem_ids = sem_res.get("ids", [[]])[0]
        sem_docs = sem_res.get("documents", [[]])[0]
        sem_metas = sem_res.get("metadatas", [[]])[0]
        sem_dists = sem_res.get("distances", [[]])[0]

        # distance -> similarity (cosine distance so 1 - d)
        sem_scores = {cid: 1.0 - float(dist) for cid, dist in zip(sem_ids, sem_dists)}

        max_kw = max([sc for _, sc in kw_res], default=0.0)
        kw_scores = {
            self.ids[idx]: (float(score) / max_kw if max_kw > 0 else 0.0)
            for idx, score in kw_res
        }

        # Collect union of ids
        id_to_content = {cid: (doc, meta) for cid, doc, meta in zip(self.ids, self.chunks, self.metadatas)}
        merged_ids = set(sem_ids) | set(kw_scores.keys())

        alpha = retrieval_cfg.HYBRID_ALPHA
        scored = []
        for cid in merged_ids:
            sem = sem_scores.get(cid, 0.0)
            kw = kw_scores.get(cid, 0.0)
            combined = alpha * sem + (1 - alpha) * kw
            doc, meta = id_to_content.get(cid, ("", {}))
            scored.append((cid, combined, doc, meta))

        topk = sorted(scored, key=lambda x: x[1], reverse=True)[:k]
        top_ids = [cid for cid, _, _, _ in topk]
        top_docs = [[doc for _, _, doc, _ in topk]]
        top_metas = [[meta for _, _, _, meta in topk]]
        top_scores = [[score for _, score, _, _ in topk]]

        return {
            "ids": [top_ids],
            "documents": top_docs,
            "metadatas": top_metas,
            "scores": top_scores,
        }
