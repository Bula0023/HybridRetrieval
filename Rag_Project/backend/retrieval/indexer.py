from langchain_chroma import Chroma
import pickle
from rank_bm25 import BM25Okapi

class Indexer:
    def __init__(self, embeddings, bm25_dir, chroma_dir="./chroma_db"):
        self.embeddings = embeddings
        self.bm25_dir = bm25_dir
        self.chroma_dir = chroma_dir
        self.vector_store = Chroma(
            collection_name="rag_documents",
            persist_directory=self.chroma_dir,
        )

    def load_bm25(self):
        bm25_path = self.bm25_dir / "bm25.pkl"
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        # print(f"data: {data}")
        bm25 = data["bm25"]
        chunks = data["chunks"]

        return bm25, chunks
    

    def update_bm25(self,new_chunks):
        # print(f"New chunks: {new_chunks}")
        # for chunk in new_chunks:
        #     print(f"Chunk page_content: {chunk.page_content}")

        bm25_path = self.bm25_dir / "bm25.pkl"

        # Load chunks already indexed
        if bm25_path.exists():

            with open(bm25_path, "rb") as f:
                saved = pickle.load(f)

            all_chunks = saved["chunks"]

        else:
            all_chunks = []

        # Add newly uploaded chunks
        all_chunks.extend(new_chunks)
        
        # Build BM25 over ALL chunks
        # print(f"chunks:{all_chunks}")
        chunk_texts = [
            chunk.page_content
            for chunk in all_chunks
        ]

        tokenized_chunks = [
            text.lower().split()
            for text in chunk_texts
        ]

        bm25 = BM25Okapi(tokenized_chunks)

        # Save both index and corresponding chunks
        with open(bm25_path, "wb") as f:
            pickle.dump(
                {
                    "bm25": bm25,
                    "chunks": all_chunks,
                    "chunk_ids": [
                        chunk.metadata["chunk_id"]
                        for chunk in all_chunks
                    ]
                },
                f
            )

        return bm25
    
    def add_to_chroma(self, chunks):
        ids = [
            chunk.metadata["chunk_id"]
            for chunk in chunks
        ]

        self.vector_store.add_documents(
            documents=chunks,
            ids=ids
        )