from sentence_transformers import CrossEncoder
import torch
class Hybrid_Retrieval:
    def __init__(self, bm25, vector_store, reranker,chunks=None):
        self.bm25 = bm25
        self.vector_store = vector_store
        self.chunks = chunks
        self.reranker = reranker
    
    def reciprocal_rank_fusion(
        self,vector_results, bm25_top_results, chunks, dense_weight=0.7, sparse_weight=0.3, k=60
    ):
        # Create a dictionary to store the scores for each document
        scores = {}
        document = {}

        # Assign scores based on BM25 results
        for rank, doc in enumerate(bm25_top_results, start=1):
            # print(doc, rank, "bm25 doc")
            # print(chunks[rank], "chunk")
            chunk_id = chunks[rank].metadata["chunk_id"]
            scores[chunk_id] = sparse_weight / rank
            document[chunk_id] = chunks[rank]

        # Assign scores based on vector results
        for rank, doc in enumerate(vector_results, start=1):
            # print(doc, "vector doc")
            chunk_id = doc.metadata["chunk_id"]
            if chunk_id in scores:
                scores[chunk_id] += dense_weight / rank
            else:
                scores[chunk_id] = dense_weight / rank
            document[chunk_id] = doc
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for chunk_id, score in ranked_docs:
            results.append({"document": document[chunk_id], "rrf_score": score})
        return results
    


    def rerank_results(self,hybrid_results, query, candidate_k=20, final_k=5):
        candidates = hybrid_results[:candidate_k]
        pairs = [[query, result["document"].page_content] for result in candidates]

        scores = self.reranker.predict(pairs)

        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:final_k]
    

    def hybrid_retrieval_engine(self,query, bm25, vector_store, chunks, top_k=10):
        # BM25 retrieval
        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_top_k_indices = bm25_scores.argsort()[-top_k:][::-1]

        # Vector store retrieval by cosine similarity
        results_with_score = vector_store.similarity_search_with_score(query=query, k=top_k)
        results = [doc for doc, score in results_with_score]
        hybrid_results = self.reciprocal_rank_fusion(results, bm25_top_k_indices, chunks)
        final_results = self.rerank_results(hybrid_results, query)
        # print(bm25_top_k_indices, "BM25 top k indices")
        # print(results, "Vector store results")
        return final_results