# src/rag_pipeline.py

from __future__ import annotations
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from .config import llm_cfg, retrieval_cfg
from .memory import load_memory, append_memory, summarize_dialogue_with_llm
from .embed_store import EmbeddingStore
from .llm_gguf import GGUFLLM
from .retrieve import HybridRetriever

from transformers import BitsAndBytesConfig
import torch


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
        # self.llm = GGUFLLM()

        # ----HF Mistral-7B-v0.1 (現用 backbone)----
        # self.tokenizer = AutoTokenizer.from_pretrained(llm_cfg.MODEL_NAME, trust_remote_code=False)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     llm_cfg.MODEL_NAME,
        #     device_map="auto",
        #     trust_remote_code=False,
        # )
        # # phi-3 / 某些模型需要指定 cache_implementation，否則 past_key_values 缺少 seen_tokens 會在 generate 時出錯
        # if getattr(self.model.generation_config, "cache_implementation", None) is None:
        #     self.model.generation_config.cache_implementation = "static"
        # # 啟用隨機取樣，避免 top_p / temperature 被忽略
        # self.model.generation_config.do_sample = True

        # ---- Gemma-2b ----
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b-it",
            quantization_config=bnb_config,
            device_map="auto"
        )

        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        # ---- Embedding Store ----

        self.store = EmbeddingStore()

        # ---- Hybrid Retriever ----
        # docs, ids, metas = self.store.fetch_all()
        # self.retriever = HybridRetriever(docs, ids, metas, store=self.store)

    def build_prompt(
        self,
        query: str,
        retrieved_docs: Optional[Dict[str, Any]] = None,
        memory_str: Optional[str] = None,
    ) -> str:
        context_blocks: List[str] = []

        if retrieved_docs is not None:
            docs = retrieved_docs.get("documents", [[]])[0]
            metas = retrieved_docs.get("metadatas", [[]])[0]

            for doc, meta in zip(docs, metas):
                title = meta.get("paper_title", "Unknown Paper")
                idx = meta.get("chunk_index", 0)
                context_blocks.append(
                    f"[{title} / chunk {idx}]\n{doc}\n"
                )

            context_text = "\n\n".join(context_blocks)
        
        mcq = False
        if "MCQ" in query: 
            mcq = True
            query = query.replace("MCQ", "")

        prompt = f"""
            You are a personal research notes assistant.

            User question:
            {query}

            
            {'Long-term memory (user\'s recent topics):\n'+ memory_str if memory_str else ''}

            
            {'Retrieved notes:\n'+context_text if retrieved_docs is not None else ''}
            
            Instructions:
            {'''
            - Provide the answer in the following format WITHOUT ANY OTEHR EXPLAINATION OR DESCRIPTION.             
             ''' if mcq else '''
            - Provide a detailed answer to the question.
            - Answer the question using ONLY the information in the retrieved notes.
            - When you make a claim, try to mention the paper title or chunk index as a citation.
            - If the answer is not present in the notes, explicitly say you cannot find it.'''}
            Final Answer:


        """.strip()

        return prompt

    def generate_answer(self, prompt: str) -> str:
        # ----hf llama 3 8B ----
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        # print(prompt)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=llm_cfg.MAX_NEW_TOKENS,
            do_sample=True,
            temperature=llm_cfg.TEMPERATURE,
            top_p=llm_cfg.TOP_P,
            return_dict_in_generate=True, # for gemma
        )
        generated = outputs.sequences[:, input_ids["input_ids"].shape[1]:] # for gemma
        answer = self.tokenizer.decode(generated[0], skip_special_tokens=True) # for gemma
        # answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # ----gguf llama 3 8B ----
        # print(prompt)
        # answer = self.llm.generate(prompt)

        return answer

    def answer(self, query: str, history: str | None = None) -> str:
        # memory_str = load_memory()
        retrieved = self.store.semantic_search(query, k=retrieval_cfg.TOP_K)
        # retrieved = self.retriever.hybrid_retrieve(query, k=retrieval_cfg.TOP_K)
        # prompt = self.build_prompt(query, retrieved, memory_str) # with memory and RAG
        prompt = self.build_prompt(query, retrieved) # with RAG only
        # prompt = self.build_prompt(query) # LLM only

        answer = self.generate_answer(prompt)

        # 更新 memory（用目前對話 history）
        if history is not None:
            summary = summarize_dialogue_with_llm(history)
            append_memory(summary)

        return answer
