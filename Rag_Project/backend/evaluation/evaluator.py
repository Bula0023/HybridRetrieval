from langchain_ollama import ChatOllama
import re
from langchain_core.prompts import ChatPromptTemplate

class RAGEvaluator:
    def __init__(self, llm):
        self.llm = llm
    
    def extract_claims(self,answer):
        claims = []
        pattern = r"Citation\s*:?\s*\[(\d+)\]\s*(.*?)(?=Citation\s*:?\s*\[\d+\]|$)"

        sentences = re.findall(pattern, answer, flags=re.DOTALL)
        # print(sentences, "extracted claims")
        for citation, claim in sentences:
            # print(claim, citation, "claim and citation", claim.strip(), citation.strip())
            claims.append({"claim": claim.strip(), "citation": citation.strip()})
        return claims
    
    def verify_citation(self,claim, document):
        prompt = ChatPromptTemplate.from_template("""
        You are a fact-checking assistant. You are verifying whether a citation supports a claim.
        Claim: {claim}
        Cited Passage:
        {document}
        Determine whether the passage directly supports the claim.
        Return exactly one of the following responses:
        Supported
        Partially_Supported
        Unsupported
        """)
        rag_chain = prompt | self.llm
        response = rag_chain.invoke({"claim": claim, "document": document.page_content})
        return response.content.strip()
    
    def retrieval_confidence_score(self,context_parts):
        scores = [item["rerank_score"] for item in context_parts]
        return sum(scores) / len(scores) if scores else 0.0
    
    def score_completeness(self,question, answer):
        prompt = ChatPromptTemplate.from_template(""" 
        You are a evaluation assistant. You are scoring the completeness of an answer to a question.
        Question: {question}
        Answer: {answer}
        Return only a number between 0 and 1 and nothing else.
        1.0 = completely answers every part
        0.5 = partially answers the question
        0.0 = does not answer the question at all
        """)
        rag_chain = prompt | self.llm
        response = rag_chain.invoke({"question": question, "answer": answer})
        text = response.content.strip()

        match = re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", text)

        return float(match.group())
    
    def citation_coverage(self,total_claims, verified_claims):
        if total_claims == 0:
            return 0.0
        return verified_claims / total_claims
    

    def calculate_confidence(self,retrieval, completeness, coverage, weights=(0.4, 0.3, 0.3)):
        retrieval_weight, completeness_weight, coverage_weight = weights
        confidence_score = (
            retrieval * retrieval_weight
            + completeness * completeness_weight
            + coverage * coverage_weight
        )
        return confidence_score
    
    def anti_hallucination(self,query, context_parts, retrieval_score, threshold=0.3, document=None):
        # print(retrieval_score, threshold, "retreival scorea nd threshold")
        if retrieval_score < threshold:
            return {
                "question": query,
                "answer": None,
                "status": "Insufficient_context",
                "confidence": retrieval_score,
                "message": "The retrieval score is below the threshold, indicating insufficient context to answer the question.",
                "found": [item["document"].page_content for item in context_parts],
                "documents_to_check": [
                    {
                        "source": item["document"].metadata.get("source", "Unknown source"),
                        "section": item["document"].metadata.get(
                            "section", "Unknown section"
                        ),
                        "page": item["document"].metadata.get("page", "Unknown page"),
                    }
                    for item in context_parts
                ],
            }
        else:
            return None