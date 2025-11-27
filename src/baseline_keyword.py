# src/baseline_keyword.py

from __future__ import annotations
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi


class KeywordBaseline:
    def __init__(self, docs: List[str], metadatas: List[Dict[str, Any]]):
        self.docs = docs
        self.metadatas = metadatas
        tokenized = [d.split() for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 5):
        scores = self.bm25.get_scores(query.split())
        ranked = sorted(
            list(enumerate(scores)),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        results = []
        for idx, sc in ranked:
            results.append({
                "content": self.docs[idx],
                "metadata": self.metadatas[idx],
                "score": float(sc),
            })
        return results
