# src/baseline_vanilla_rag.py

from __future__ import annotations
from typing import Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from .embed_store import EmbeddingStore
from .config import LLMConfig, EmbeddingConfig


class VanillaRAG:
    """
    - 不使用 personalized memory
    - 不使用 hybrid search
    - 使用較簡單的 embedding model / prompt
    """

    def __init__(self, llm_model_name: str, embed_model_name: str):
        self.llm_cfg = LLMConfig(MODEL_NAME=llm_model_name)
        self.embed_cfg = EmbeddingConfig(MODEL_NAME=embed_model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_cfg.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_cfg.MODEL_NAME,
            device_map="auto",
        )
        self.store = EmbeddingStore(
            model_name=self.embed_cfg.MODEL_NAME,
            device=self.embed_cfg.DEVICE,
        )

    def build_prompt(self, query: str, retrieved: Dict[str, Any]) -> str:
        docs = retrieved.get("documents", [[]])[0]
        context_text = "\n\n".join(docs)
        prompt = f"""
You are a Q&A assistant.

Question:
{query}

Context:
{context_text}

Answer the question using the context. If the answer is not present, say you don't know.
        """.strip()
        return prompt

    def answer(self, query: str) -> str:
        retrieved = self.store.semantic_search(query, k=5)
        prompt = self.build_prompt(query, retrieved)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.llm_cfg.MAX_NEW_TOKENS,
            temperature=self.llm_cfg.TEMPERATURE,
            top_p=self.llm_cfg.TOP_P,
        )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
