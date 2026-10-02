from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
)
import json
from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from loaders.document_loader_backup import ingest_document, clean_text
from generation.generator import RAGGenerator
from evaluation.evaluator import RAGEvaluator
from retrieval.chunker import DocumentChunker
from retrieval.indexer import Indexer
from retrieval.hybrid_retriever import Hybrid_Retrieval
from pathlib import Path
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import pickle
import torch
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from scipy.spatial import distance
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama
import shutil
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BM25_DIR = Path("./bm25_index")
BM25_DIR.mkdir(exist_ok=True)
SCORE_DIR = Path("./data/scores")
UPLOAD_DIR = Path("data/uploads")
SCORE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2", activation_fn=torch.nn.Sigmoid()
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


llm = ChatOllama(
    model="qwen3:4b",
    # base_url="http://127.0.0.1:11434",
    base_url="http://host.docker.internal:11434",
    temperature=0.2,
    # num_predict=250,
)
llm2 = ChatOllama(
    model="llama3.2:3b",
    # base_url="http://127.0.0.1:11434",
    base_url="http://host.docker.internal:11434",
    temperature=0.0,
    # num_predict=250,
)
rag_generator = RAGGenerator(llm)
rag_evaluator = RAGEvaluator(llm2)
document_chunker = DocumentChunker(embeddings)
indexer = Indexer(embeddings, BM25_DIR)

# docs = []
# for file_path in Path("uploads").iterdir():
#     if file_path.is_file():
#         document = ingest_document(file_path)
#         for item in document:
#             docs.append(Document(page_content=item["page_content"], metadata=item["metadata"]))



# chunks = document_chunker.chunk_documents(docs, strategy="semantic")


# updated_chunks = document_chunker.remove_near_duplicates(chunks, threshold=0.9)
# ids = [chunk.metadata["chunk_id"] for chunk in chunks]

# vector_store = Chroma.from_documents(
#     embedding=embeddings,
#     documents=updated_chunks,
#     ids=ids,
#     collection_name="rag_documents",
#     persist_directory="./chroma_db",
# )
# chunk_texts = [chunk.page_content for chunk in chunks]
# tokenize_chunks = [text.lower().split() for text in chunk_texts]
# bm25 = BM25Okapi(tokenize_chunks)
# with open(BM25_DIR / "bm25.pkl", "wb") as f:
#                 pickle.dump(
#                 {
#                     "bm25": bm25,
#                     "chunks": updated_chunks,
#                     "chunk_ids": [
#                         chunk.metadata["chunk_id"]
#                         for chunk in updated_chunks
#                     ]
#                 },
#                 f
#             )

bm25,chunks = indexer.load_bm25()
vector_store = Chroma(
    collection_name="rag_documents",
    persist_directory="./chroma_db",
)


hybrid_retrieval = Hybrid_Retrieval(bm25,vector_store,reranker)


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    docs = []
    document = ingest_document(file_path)

    for item in document:
        docs.append(Document(page_content=item["page_content"], metadata=item["metadata"]))

    chunks = document_chunker.chunk_documents(docs, strategy="semantic")

    updated_chunks = document_chunker.remove_near_duplicates(chunks, threshold=0.9)
    # ids = [chunk.metadata["chunk_id"] for chunk in updated_chunks]
    # vector_store.add_documents(
    #     documents=updated_chunks,
    #     ids=ids)
    indexer.add_to_chroma(chunks)
    indexer.update_bm25(updated_chunks)
    return {
        "status": "success",
        "filename": file.filename,
        "chunks_added": len(updated_chunks),
    }

@app.post("/query")
async def question(query):
    # query = "How does seek behavior work?"
    context_parts = hybrid_retrieval.hybrid_retrieval_engine(query, bm25, vector_store, chunks)


    retrieval_score = rag_evaluator.retrieval_confidence_score(context_parts)
    # print("retrieval_score:", retrieval_score)
    hallucinate_data = rag_evaluator.anti_hallucination(
        query, context_parts, retrieval_score, 0.1
    )
    if hallucinate_data:
        with open(SCORE_DIR / "halluciante_data.json", "w", encoding="utf-8") as f:
            json.dump(hallucinate_data, f, indent=4, ensure_ascii=False)
    else:
        # retrieval = rag_generator.rag_chain(query, context_parts, llm, 0.1)
        
        stream = rag_generator.rag_chain(
            query,
            context_parts,
            llm,
            0.1
        )
        def generate_response(
            retrieval,
            query,
            context_parts,
            rag_evaluator,
            retrieval_score
        ):  
            full_answer = ""

            # 1. Stream the answer to React
            for chunk in retrieval:
                # print(chunk, "this is the chunk")
                text = chunk.content

                if text:
                    full_answer += text

                    yield json.dumps({
                        "type": "answer",
                        "content": text
                    }) + "\n"

            claims = rag_evaluator.extract_claims(full_answer)

            verification_results = []
            total_claims = len(claims)
            verified_claims = 0

            for item in claims:
                # print(item, "claim item")
                claim = item["claim"]
                citation = item["citation"]
                # print(f"verifying claim: {claim} with citation: {citation}")
                document = context_parts[int(citation) - 1]["document"]
                status = rag_evaluator.verify_citation(claim, document)
                if status in ["Supported", "Partially_Supported"]:
                    verified_claims += 1
                verification_results.append(
                    {"claim": claim, "citation": citation, "status": status}
                )

            completeness_score = rag_evaluator.score_completeness(query, full_answer)
            coverage_score = rag_evaluator.citation_coverage(total_claims, verified_claims)
            confidence_score = rag_evaluator.calculate_confidence(
                retrieval_score, completeness_score, coverage_score
            )
            # print(f"confidence score: {confidence_score}")
            confidence_score_data = {
                "question": query,
                "answer": full_answer,
                "confidence": confidence_score,
                "confidence_breakdown": {
                    "retrieval_score": retrieval_score,
                    "completeness_score": completeness_score,
                    "coverage_score": coverage_score,
                },
                "citation_verification": verification_results,
            }
            with open(SCORE_DIR / "confidence_score.json", "w", encoding="utf-8") as f:
                json.dump(confidence_score_data, f, indent=4, ensure_ascii=False)
            # 3. Send evaluation data to React
            yield json.dumps({
                "type": "confidence",
                "data": confidence_score_data
            }) + "\n"

            # 4. Tell React streaming is finished
            yield json.dumps({
                "type": "done"
            }) + "\n"

        return StreamingResponse(
            generate_response(
                retrieval=stream,
                query=query,
                context_parts=context_parts,
                rag_evaluator=rag_evaluator,
                retrieval_score=retrieval_score
            ),
            media_type="application/x-ndjson"
        )
        # return {
        # "status": "success",
        # "confidence_score": confidence_score_data
        # }


# json_string = json.dumps(confidence_score_data, indent=4, sort_keys= True)
# vector_store.add_documents(chunks)
# vector_store.query(query_texts=["What is the main topic of the document?"], n_results=10)

# data = vector_store.get()
# print(data, "Chromadb data")
