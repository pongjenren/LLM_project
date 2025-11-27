# src/llm_gguf.py

from __future__ import annotations
from typing import Optional, List, Dict, Any

from llama_cpp import Llama

from .config import gguf_cfg


class GGUFLLM:
    def __init__(self):
        # 初始化 GGUF 模型（使用 llama.cpp backend）
        self.llm = Llama(
            model_path=str(gguf_cfg.MODEL_PATH),
            n_ctx=gguf_cfg.N_CTX,
            n_gpu_layers=gguf_cfg.N_GPU_LAYERS,  # -1: 盡可能放到 GPU
            n_threads=gguf_cfg.N_THREADS,
            logits_all=False,
            vocab_only=False,
            use_mmap=True,
            use_mlock=False,   # 如有記憶體問題可改成 True
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        if max_new_tokens is None:
            max_new_tokens = gguf_cfg.MAX_NEW_TOKENS
        if temperature is None:
            temperature = gguf_cfg.TEMPERATURE
        if top_p is None:
            top_p = gguf_cfg.TOP_P
        if stop is None:
            # 可以加上簡單的 stop token，避免模型繼續產生 "User:"
            stop = ["</s>", "User:", "USER:", "Assistant:", "ASSISTANT:"]

        output: Dict[str, Any] = self.llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        # llama-cpp-python 的輸出格式：{"choices": [{"text": "..."}], ...}
        text = output["choices"][0]["text"]
        return text.strip()
