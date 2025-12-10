# src/config.py

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = ROOT / "data"
    PAPERS_DIR: Path = DATA_DIR / "papers"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    VECTOR_DB_DIR: Path = DATA_DIR / "vectordb"
    MEMORY_FILE: Path = DATA_DIR / "memory.txt"
    GOLDEN_QA_FILE: Path = DATA_DIR / "golden_qa.json"


@dataclass
class EmbeddingConfig:
    MODEL_NAME: str = "intfloat/e5-large-v2"
    DEVICE: str = "cuda"  # or "cpu"


@dataclass
class LLMConfig:
    # MODEL_NAME: str = "meta-llama/Meta-Llama-3-8B-Instruct"  # 原始 backbone
    # MODEL_NAME: str = "mistralai/Mistral-7B-v0.1"  # 先前 HF Mistral 7B
    # MODEL_NAME: str = "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF"  # 新 backbone
    # MODEL_NAME: str = "microsoft/Phi-3-mini-4k-instruct"
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.9


@dataclass
class RetrievalConfig:
    TOP_K: int = 5
    HYBRID_ALPHA: float = 0.5  # semantic / keyword 權重

@dataclass
class GGUFConfig:
    # MODEL_PATH: Path = Paths().ROOT / "models" / "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"  # 新 backbone GGUF
    # MODEL_PATH: Path = Paths().ROOT / "models" / "meta-llama-3-8b-instruct_Q4_K_M.gguf"
    MODEL_PATH: Path = Paths().ROOT / "models" / "Qwen3-8B-Q4_K_M.gguf"
    N_CTX: int = 8192         # context window
    N_GPU_LAYERS: int = 30    # -1 = as many as possible on GPU, 視 VRAM 而定
    N_THREADS: int = 8        # CPU threads，用來做部分計算
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.9

paths = Paths()
embed_cfg = EmbeddingConfig()
llm_cfg = LLMConfig()
retrieval_cfg = RetrievalConfig()
gguf_cfg = GGUFConfig()
