"""GAMER that connect to vectorized data assets"""

import asyncio
from typing import Annotated, Literal

from langchain import hub
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict

from metadata_chatbot.nodes.utils import HAIKU_3_5_LLM, SONNET_3_7_LLM
from metadata_chatbot.retrievers.docdb_retriever import DocDBRetriever


class FilterGenerator(TypedDict):
    """MongoDB filter to be applied before vector retrieval"""

    filter_query: Annotated[dict, ..., "MongoDB match filter"]
    top_k: int = Annotated[dict, ..., "Number of documents"]


filter_prompt = hub.pull("eden19/filtergeneration")
filter_generator_llm = SONNET_3_7_LLM.with_structured_output(FilterGenerator)
filter_generation_chain = filter_prompt | filter_generator_llm


async def filter_generator(state: dict) -> dict:
    """
    Filter database by constructing basic MongoDB match filter
    and determining number of documents to retrieve
    """
    query = state["query"]
    if state.get("chat_history") is None or state.get("chat_history") == "":
        chat_history = state["messages"]
    else:
        chat_history = state["chat_history"]

    try:
        result = await filter_generation_chain.ainvoke(
            {"query": query, "chat_history": chat_history}
        )
        filter = result["filter_query"]
        top_k = result["top_k"]
        message = (
            f"Using MongoDB filter: {filter} on the database "
            f"and retrieving {top_k} documents"
        )

    except Exception as ex:
        filter = None
        top_k = None
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)

    return {
        "filter": filter,
        "top_k": top_k,
        "messages": [AIMessage(message)],
    }


async def retrieve_VI(state: dict) -> dict:
    """
    Retrieve documents
    """
    query = state["query"]
    filter = state["filter"]
    top_k = state["top_k"]

    try:
        retriever = DocDBRetriever(k=top_k)
        documents = await retriever.aget_relevant_documents(
            query=query, query_filter=filter
        )
        message = "Retrieving relevant documents from vector index..."

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        documents = []

    return {
        "documents": documents,
        "messages": [AIMessage(message)],
    }


# Check if retrieved documents answer question
class RetrievalGrader(TypedDict):
    """Relevant material in the retrieved document +
    Binary score to check relevance to the question"""

    binary_score: Annotated[
        Literal["yes", "no"],
        ...,
        "Retrieved documents are relevant to the query, 'yes' or 'no'",
    ]


retrieval_grader = HAIKU_3_5_LLM.with_structured_output(RetrievalGrader)
retrieval_grade_prompt = hub.pull("eden19/retrievalgrader")
doc_grader = retrieval_grade_prompt | retrieval_grader


async def grade_doc(query: str, doc: Document):
    """
    Grades whether each document is relevant to query
    """
    score = await doc_grader.ainvoke(
        {"query": query, "document": doc.page_content}
    )
    grade = score["binary_score"]

    try:
        if grade == "yes":
            return doc.page_content
        else:
            return None
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        return message


async def grade_documents(state: dict) -> dict:
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    query = state["query"]
    documents = state["documents"]

    filtered_docs = await asyncio.gather(
        *[grade_doc(query, doc) for doc in documents],
        return_exceptions=True,
    )
    filtered_docs = [doc for doc in filtered_docs if doc is not None]

    return {
        "documents": filtered_docs,
        "messages": [
            AIMessage("Checking document relevancy to your query...")
        ],
    }


# Generating response to documents retrieved from the vector index
answer_generation_prompt = hub.pull("eden19/answergeneration")
rag_chain = answer_generation_prompt | SONNET_3_7_LLM | StrOutputParser()


async def generate_VI(state: dict) -> dict:
    """
    Generate answer
    """
    query = state["query"]
    documents = state["documents"]

    try:
        message = await rag_chain.ainvoke(
            {"documents": documents, "query": query}
        )
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)

    return {
        "messages": [AIMessage(str(message))],
        "generation": message,
    }
