# src/preprocess.py

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import paths


@dataclass
class Chunk:
    id: str
    content: str
    metadata: Dict[str, Any]


def read_pdf(path: Path) -> str:
    """將 PDF 轉成純文字。"""
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def split_into_chunks(text: str, paper_title: str) -> List[Chunk]:
    """使用 RecursiveCharacterTextSplitter 切割為 semantic chunks。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " "],
    )
    raw_chunks = splitter.split_text(text)

    chunks: List[Chunk] = []
    for i, c in enumerate(raw_chunks):
        chunks.append(
            Chunk(
                id=f"{paper_title}_chunk_{i}",
                content=c,
                metadata={
                    "paper_title": paper_title,
                    "chunk_index": i,
                },
            )
        )
    return chunks


def preprocess_all_papers() -> List[Chunk]:
    """讀取 data/papers 下所有 PDF，並輸出 chunks list。"""
    paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Chunk] = []
    for pdf_path in paths.PAPERS_DIR.glob("*.pdf"):
        text = read_pdf(pdf_path)
        paper_title = pdf_path.stem
        chunks = split_into_chunks(text, paper_title)
        all_chunks.extend(chunks)

    # 將 chunks 存成 json 方便之後重用
    serialized = [
        {"id": c.id, "content": c.content, "metadata": c.metadata}
        for c in all_chunks
    ]
    output_path = paths.PROCESSED_DIR / "chunks.json"
    output_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")

    return all_chunks
