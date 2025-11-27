# src/memory.py

from __future__ import annotations
from typing import Optional, Any

from transformers import pipeline

from .config import paths


def load_memory() -> str:
    """載入 long-term memory 摘要。"""
    if not paths.MEMORY_FILE.exists():
        return ""
    return paths.MEMORY_FILE.read_text(encoding="utf-8")


def append_memory(summary: str):
    """將新的摘要附加到 memory 檔案。"""
    paths.MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with paths.MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(summary.strip() + "\n")


def summarize_dialogue_with_llm(history: str) -> str:
    """
    使用 LLM 將近期對話 summarization 成 2~3 句。
    預設使用 transformers summarization pipeline，失敗時退回簡易 heuristics。
    """
    text = history.strip()
    if not text:
        return ""

    # avoid extremely long inputs to the summarizer
    text = text[-2000:]

    summarizer = _get_summarizer()
    if summarizer is not None:
        try:
            result = summarizer(
                text,
                max_length=120,
                min_length=30,
                do_sample=False,
            )
            if result and isinstance(result, list) and "summary_text" in result[0]:
                return result[0]["summary_text"]
        except Exception:
            # fall back to heuristic summarization below
            pass

    # Fallback: take the last few turns and truncate
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    tail = " ".join(lines[-6:])
    return tail[:500]


_summarizer: Optional[Any] = None


def _get_summarizer():
    global _summarizer
    if _summarizer is not None:
        return _summarizer
    try:
        _summarizer = pipeline("summarization", model="facebook/bart-base")
    except Exception:
        _summarizer = None
    return _summarizer
