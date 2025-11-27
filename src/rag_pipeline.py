# src/rag_pipeline.py

from __future__ import annotations
from typing import Dict, Any, List
# ----hf llama 3 8B----
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from .config import llm_cfg, retrieval_cfg
# from .embed_store import EmbeddingStore

# ----gguf llama 3 8B----
from .config import retrieval_cfg
from .memory import load_memory, append_memory, summarize_dialogue_with_llm
from .embed_store import EmbeddingStore
from .llm_gguf import GGUFLLM


class RAGPipeline:
    def __init__(self):
        # ----hf llama 3 8B ----
        # self.tokenizer = AutoTokenizer.from_pretrained(llm_cfg.MODEL_NAME)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     llm_cfg.MODEL_NAME,
        #     device_map="auto",
        # )

        # ----gguf llama 3 8B ----
        # 初始化 GGUF LLM
        self.llm = GGUFLLM()

        self.store = EmbeddingStore()

    def build_prompt(
        self,
        query: str,
        retrieved_docs: Dict[str, Any],
        memory_str: str,
    ) -> str:
        context_blocks: List[str] = []
        docs = retrieved_docs.get("documents", [[]])[0]
        metas = retrieved_docs.get("metadatas", [[]])[0]

        for doc, meta in zip(docs, metas):
            title = meta.get("paper_title", "Unknown Paper")
            idx = meta.get("chunk_index", 0)
            context_blocks.append(
                f"[{title} / chunk {idx}]\n{doc}\n"
            )

        context_text = "\n\n".join(context_blocks)

        prompt = f"""
You are a personal research notes assistant.

User question:
{query}

Long-term memory (user's recent topics):
{memory_str}

Retrieved notes:
{context_text}

Instructions:
- Answer the question using ONLY the information in the retrieved notes.
- When you make a claim, try to mention the paper title or chunk index as a citation.
- If the answer is not present in the notes, explicitly say you cannot find it.
        """.strip()
        return prompt

    def generate_answer(self, prompt: str) -> str:
        # ----hf llama 3 8B ----
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        # outputs = self.model.generate(
        #     **inputs,
        #     max_new_tokens=llm_cfg.MAX_NEW_TOKENS,
        #     temperature=llm_cfg.TEMPERATURE,
        #     top_p=llm_cfg.TOP_P,
        # )
        # answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # ----gguf llama 3 8B ----
        answer = self.llm.generate(prompt)

        return answer

    def answer(self, query: str, history: str | None = None) -> str:
        memory_str = load_memory()
        retrieved = self.store.semantic_search(query, k=retrieval_cfg.TOP_K)
        prompt = self.build_prompt(query, retrieved, memory_str)
        answer = self.generate_answer(prompt)

        # 更新 memory（用目前對話 history）
        if history is not None:
            summary = summarize_dialogue_with_llm(history)
            append_memory(summary)

        return answer
