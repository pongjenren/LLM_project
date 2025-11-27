# src/evaluation.py

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any

from .config import paths
from .rag_pipeline import RAGPipeline


@dataclass
class QAPair:
    question: str
    golden_answer: str


def load_golden_qa() -> List[QAPair]:
    data = json.loads(paths.GOLDEN_QA_FILE.read_text(encoding="utf-8"))
    return [QAPair(**item) for item in data]


def simple_string_overlap(pred: str, golden: str) -> float:
    """
    非正式的 string-based accuracy/overlap，只做骨架。
    之後可以改成 LLM-as-a-judge。
    """
    return float(golden.lower() in pred.lower())


def evaluate_rag_system():
    qa_pairs = load_golden_qa()
    rag = RAGPipeline()

    results: List[Dict[str, Any]] = []
    for pair in qa_pairs:
        start = time.time()
        answer = rag.answer(pair.question)
        latency = time.time() - start
        acc = simple_string_overlap(answer, pair.golden_answer)
        results.append(
            {
                "question": pair.question,
                "golden": pair.golden_answer,
                "answer": answer,
                "latency": latency,
                "string_accuracy": acc,
            }
        )

    summary = {
        "avg_latency": sum(r["latency"] for r in results) / len(results) if results else 0.0,
        "avg_string_accuracy": sum(r["string_accuracy"] for r in results) / len(results) if results else 0.0,
        "results": results,
    }

    output_path = paths.DATA_DIR / "evaluation_results.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary
