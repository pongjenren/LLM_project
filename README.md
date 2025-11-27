# How to run
## Environment
Just install all modules in ```requirements.txt```
> notice ```llama-cpp-python``` need special insturction!
## Add pdf
1. Add pdf into data/papers
2. run ```build_db.py```
## Run GUI
1. run ```main.py```

# About backbone
**LLM: Llama 3 8B Instruct Q4 K M GGUF**

I'm using GPU with only 8GB VRAM, if your VRAM is >= 16GB, consider using native Llama 3 8B by inversing ```Change to GGUF``` steps.

**Embedding: e5-large-v2**

# Chagne to GGUF:
1. install llama-cpp-python ```CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python```
1. download model: https://huggingface.co/NoelJacob/Meta-Llama-3-8B-Instruct-Q4_K_M-GGUF/tree/main
2. revise: src/config.py
3. add: src/llm_gguf.py
4. revise: src/rag_pipeline.py



# 📚 Personal Research Notes Assistant

*A local RAG system for reading & querying personal PDF research papers using Llama 3 (GGUF) + ChromaDB + E5 embeddings*

---

## ✨ Overview

This project implements a fully local **Retrieval-Augmented Generation (RAG)** system that turns your personal research papers (PDF) into a semantic knowledge base you can query in natural language.

功能包含：

* 🔍 **PDF ingestion**：讀取論文並切成語意片段
* 🧠 **Embedding + ChromaDB**：以 E5-large-v2 建立向量資料庫
* 🎯 **Hybrid Retrieval**（可加入 BM25）⚠️ Not implemented yet ⚠️
* 🤖 **LLM generation using Llama 3 8B GGUF**（llama-cpp-python）
* 📝 **Long-term memory module**（自動摘要最近對話）⚠️ Not implemented yet ⚠️
* 💬 **Gradio UI**（互動式查詢）

完全在本地執行，**不需要任何 API 金鑰、不上傳資料、不依賴外部雲端**。

---

## 🛠 System Architecture

```
PDFs → Preprocess → Chunks → E5 Embeddings → ChromaDB  
                                  ↓
                       Hybrid Retrieval (semantic/keyword)
                                  ↓
                     Memory Module (session summary)
                                  ↓
                 Llama 3 8B (GGUF, llama-cpp-python)
                                  ↓
                      Final Answer Generation
```

---

## 📁 Project Structure

```
LLM_project/
│
├── main.py
├── build_db.py
├── requirements.txt
│
├── models/
│   └── llama-3-8b-instruct-q4_k_m.gguf
│
├── data/
│   ├── papers/          # 放你的 PDF
│   ├── processed/       # 中間資料
│   ├── vectordb/        # ChromaDB 儲存位置
│   └── memory.txt       # 長期記憶
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── embed_store.py
│   ├── retrieve.py
│   ├── memory.py
│   ├── llm_gguf.py
│   ├── rag_pipeline.py
│   ├── baseline_keyword.py
│   ├── baseline_vanilla_rag.py
│   └── evaluation.py
│
└── app/
    └── ui_gradio.py
```

---

## 💾 Installation

### 1. 建立 conda / venv（推薦 conda）

```bash
conda create -n LLM_env python=3.10 -y
conda activate LLM_env
```

### 2. 安裝依賴

> **強烈建議使用預編譯 CUDA 版本 → 安裝成功率 100%**

```bash
pip install -r requirements.txt
pip install llama-cpp-python-cu122
```

> 如果你使用 CPU：
> `pip install llama-cpp-python`

---

## 🔽 Download GGUF Model

使用你喜歡的 Llama 3 GGUF 量化模型，例如：

* `Llama-3-8B-Instruct-Q4_K_M.gguf`
* `Llama-3-8B-Instruct-Q5_K_M.gguf`

從 HuggingFace 下載（例如 TheBloke 或 bartowski）。

放到：

```
models/llama-3-8b-instruct-q4_k_m.gguf
```

---

## 📄 Adding PDF Papers

把任何 `.pdf` 放進：

```
data/papers/
```

例：

```
data/papers/
    diffusion_models_paper.pdf
    llm_agents_survey.pdf
```

---

## 🏗 Build Vector Database

跑：

```bash
python build_db.py
```

會輸出：

```
📄 Loading and preprocessing PDFs...
✔ Loaded XXX chunks
🧠 Initializing vector store...
📥 Adding into ChromaDB...
🎉 All PDFs have been added into the RAG database!
```

---

## 🚀 Run the UI

```bash
python main.py
```

開啟你的瀏覽器，Gradio 介面會讓你：

* 輸入自然語言問題
* 檢索你的 PDF 中的段落
* 由 Llama 3 GGUF 生成答案
* 引用 chunk 與 paper title

---

## 🔧 Configuration

所有配置放在：

```
src/config.py
```

你可以調整：

* Embedding model（E5-large-v2）
* GGUF model 路徑
* context window（8k/16k）
* retrieval Top-K
* memory file path

如果你只有 8GB VRAM，建議：

```python
N_GPU_LAYERS = -1     # 自動放到 GPU
N_CTX = 8192
```



## 🙌 Acknowledgements

This project uses:

* **Meta Llama 3**
* **llama.cpp / llama-cpp-python**
* **ChromaDB**
* **SentenceTransformers (E5-large-v2)**
* **Gradio**
* **LangChain text splitters**