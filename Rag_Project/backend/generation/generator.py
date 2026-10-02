from langchain_core.prompts import ChatPromptTemplate


class RAGGenerator:
    def __init__(self, llm):
        self.llm = llm

    def rag_chain(self,query, context_parts, llm, some_threshold=0.1):
        parts = []
        for idx, item in enumerate(context_parts):
            docs = item["document"]
            source = docs.metadata.get("source", "Unknown source")
            section = docs.metadata.get("section", "Unknown section")
            page = docs.metadata.get("page", "Unknown page")
            parts.append(f"""
                Citation:[{idx+1}]
                Source:{source}
                Section:{section}
                Page:{page}
                Content:{docs.page_content}
                """)
        context = "\n\n---\n\n".join(parts)
        # print(context_parts, "rag chain context parts")
        # print(context, "rag chain context")
        system_prompt = """
        You are a question-answering assistant.

        Answer the user's question using only the provided context. 
        Write it in a maximum 6 sentences.

        If the context does not contain enough information to answer the question,
        say: "I could not find enough information in the provided documents."

        Do not make up facts or use outside knowledge.
        Each context chunk is labeled with only a citation number such as [1], [2], [3]. 
        Do not add extra sentences about where your claim came from just use the brackets and number as shown [1].

        For every factual claim you make, include the citation number of the chunk
        that supports that claim after such claim, 
        and explicitly state when the context doesn’t contain enough information to answer. 
        Include the retrieved chunks as numbered context blocks.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Question:{question} \n\nContext:\n{context}"),
            ]
        )
        # formatted = prompt.invoke({
        # "context": context,
        # "question": query
        # })

        # print(formatted)

        rag_chain = prompt | llm
        if not context_parts:
            return "I could not find enough information in the provided documents."
        if context_parts[0]["rerank_score"] < some_threshold:
            return "I could not find enough information in the provided documents."
        else:
            return rag_chain.stream({"question": query, "context": context})
