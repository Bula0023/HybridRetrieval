from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from scipy.spatial import distance

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
class DocumentChunker:
    def __init__(self,embeddings, chunk_size=1000, chunk_overlap=200):
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self,docs, strategy="fixed"):
        if strategy == "fixed":
            splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
        elif strategy == "semantic":
            splitter = SemanticChunker(embeddings=self.embeddings)
        elif strategy == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            # chunks = []

            # for doc in docs:
            #     section_chunks = splitter.split_documents([doc])
            #     chunks.extend(section_chunks)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        # print(f"{docs} documents")
        chunks = splitter.split_documents(docs)
        # print(f"Number of chunks created: {len(chunks)}")

        for index, chunk in enumerate(chunks):
            chunk_id = f"{chunk.metadata['source']}-{strategy}-{index}"
            chunk.metadata["chunk_strategy"] = strategy
            chunk.metadata["chunk_index"] = index
            chunk.metadata["character_count"] = len(chunk.page_content)
            chunk.metadata["section_heading"] = chunk.metadata.get("section", None)
            chunk.metadata["chunk_id"] = chunk_id

        # print(f"Sample chunk: {chunks[0]}")
        return chunks

    def remove_near_duplicates(self,chunks, threshold=0.9):
        unique_chunks = []
        embeddings_list = [self.embeddings.embed_query(chunk.page_content) for chunk in chunks]
        for i, chunk in enumerate(chunks):
            is_duplicate = False
            for j, unique_chunk in enumerate(unique_chunks):
                # print("start",unique_chunk,"unique chunk")
                sim = 1 - distance.cosine(embeddings_list[i], embeddings_list[j])
                if sim > threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_chunks.append(chunk)
        return unique_chunks