# build_db.py

from src.preprocess import preprocess_all_papers
from src.embed_store import EmbeddingStore
from src.config import paths

def main():
    print("📄 Loading and preprocessing PDFs...")
    chunks = preprocess_all_papers()
    print(f"✔ Loaded {len(chunks)} chunks from PDFs")

    print("🧠 Initializing vector store...")
    store = EmbeddingStore()

    print("📥 Adding chunks into vector DB...")
    store.add_chunks(chunks)
    
    print("🎉 All PDFs have been successfully added to the RAG database.")
    print(f"Database stored at: {paths.VECTOR_DB_DIR}")

if __name__ == "__main__":
    main()
