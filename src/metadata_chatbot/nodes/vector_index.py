"""GAMER that connect to vectorized data assets"""

import asyncio

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from metadata_chatbot.models import (
    doc_grader,
    filter_generation_chain,
    rag_chain,
)
from metadata_chatbot.retrievers.docdb_retriever import DocDBRetriever


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
    route_to_mongodb = False
    documents = []

    try:
        message = AIMessage(
            "Retrieving relevant documents from vector index..."
        )
        retriever = DocDBRetriever(k=top_k)
        documents = await retriever.aget_relevant_documents(
            query=query, query_filter=filter
        )

        # If retrieval worked but returned no documents
        if not documents:
            message = AIMessage(
                "No documents found in vector index, routing to MongoDB."
            )
            route_to_mongodb = True

    except Exception as ex:
        # Catch all exceptions from the retrieval process
        error_type = type(ex).__name__
        error_message = str(ex)

        error = f"Vector retrieval failed: {error_type} - {error_message}"
        message = AIMessage(
            "Vector index retrieval error. Routing to MongoDB."
        )
        route_to_mongodb = True

    return {
        "documents": documents,
        "messages": [message],
        "route_to_mongodb": route_to_mongodb,
    }


# Check if vector index is able to retrieve relevant info, if not route to mongodb
def route_to_mongodb(state: dict):
    if state["route_to_mongodb"] is True:
        return "route_query"
    elif state.get("documents", None) is None:
        return "route_query"
    else:
        return "grade_documents"


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
