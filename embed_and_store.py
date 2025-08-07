import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

def get_embeddings():
    """Create MiniLM embeddings instance"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

def store_in_faiss(chunks: list[str], index_dir: str):
    """Store text chunks in FAISS index"""
    print("💾 Embedding with MiniLM and saving to FAISS...")
    
    # Delete existing FAISS directory if it exists
    index_path = Path(index_dir)
    if index_path.exists():
        print(f"🗑️ Deleting existing FAISS index at {index_dir}")
        shutil.rmtree(index_path)
    
    # Create parent directory if it doesn't exist
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create embeddings and vectorstore
    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    
    # Save the vectorstore
    vectorstore.save_local(index_dir)
    print(f"✅ FAISS index saved to {index_dir}")
    
    # Save metadata
    meta_path = Path(index_dir).parent / f"{Path(index_dir).name}_meta.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i} ---\n{chunk}\n\n")
    print(f"📝 Metadata saved to {meta_path}")