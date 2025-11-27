# src/embed_store.py

from __future__ import annotations
from typing import List, Optional

import torch

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .config import paths, embed_cfg
from .preprocess import Chunk


class EmbeddingStore:
    def __init__(
        self,
        collection_name: str = "research_notes",
        model_name: Optional[str] = embed_cfg.MODEL_NAME,
        device: Optional[str] = embed_cfg.DEVICE,
    ):
        self.client = chromadb.PersistentClient(
            path=str(paths.VECTOR_DB_DIR),
            settings=Settings()
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        resolved_device = device or embed_cfg.DEVICE
        if resolved_device == "cuda" and not torch.cuda.is_available():
            resolved_device = "cpu"

        self.model = SentenceTransformer(
            model_name or embed_cfg.MODEL_NAME,
            device=resolved_device,
        )

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        ALWAYS convert tensor → numpy → python list
        (This ensures compatibility with Chroma DB.)
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,     # returns np.ndarray
            normalize_embeddings=True, # optional but recommended
        )
        return embeddings.tolist()     # convert to pure python list

    def semantic_search(self, query: str, k: int = 5):
        """Query ChromaDB and return top-k results with distances."""
        if self.collection.count() == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        query_emb = self.encode([query])
        res = self.collection.query(
            query_embeddings=query_emb,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return res

    def fetch_all(self):
        """Load all stored docs/metadata for downstream keyword retrieval."""
        total = self.collection.count()
        if total == 0:
            return [], [], []

        # Chroma supports pagination via where with offset/limit; for small corpora a single call suffices.
        all_docs = self.collection.get(include=["documents", "metadatas", "ids"])
        return all_docs.get("documents", []), all_docs.get("ids", []), all_docs.get("metadatas", [])

    def add_chunks(self, chunks: List[Chunk]):
        ids = [c.id for c in chunks]
        docs = [c.content for c in chunks]
        metas = [c.metadata for c in chunks]

        print("📐 Encoding embeddings...")
        embs = self.encode(docs)  # <- now embeddings are list[list[float]]

        print("📥 Adding into ChromaDB...")
        self.collection.add(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embs,
        )

        print("🎉 Successfully added", len(chunks), "chunks to the vector DB.")
