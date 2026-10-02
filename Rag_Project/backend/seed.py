from pathlib import Path
from loaders.document_loader_backup import ingest_document, clean_text
from retrieval.indexer import Indexer
from retrieval.chunker import DocumentChunker
from langchain_huggingface import HuggingFaceEmbeddings

BM25_DIR = Path("./bm25_index")
BM25_DIR.mkdir(exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
indexer = Indexer(embeddings, BM25_DIR)
chunker = DocumentChunker(embeddings)

def seed():
    target_path = Path("/data/uploads")

    # Check if the directory exists and contains any actual files
    if target_path.exists():
        for file_path in Path("/sample_docs").iterdir():
            if file_path.is_file():
                docs = ingest_document
                chunks = chunker.chunk_documents(docs)
                unique_chunks = chunker.remove_duplicates(chunks)
                indexer.add_to_chroma(unique_chunks)
                indexer.update_bm25(unique_chunks)

if __name__ == "__main__":
    seed()