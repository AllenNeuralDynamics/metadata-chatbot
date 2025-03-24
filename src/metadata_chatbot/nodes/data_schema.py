"""GAMER that connect to vectorized data schema"""

from langchain_core.messages import AIMessage

from metadata_chatbot.models import schema_chain
from metadata_chatbot.retrievers.data_schema_retriever import (
    DataSchemaRetriever,
)


def retrieve_schema(state: dict) -> dict:
    """
    Retrieves info about data schema in prod DB
    """

    """
    Retrieve context from data schema collection
    """
    query = state["query"]

    try:
        retriever = DataSchemaRetriever(
            k=5, collection="aind_data_schema_vectors"
        )
        documents = retriever._get_relevant_documents(query=query)
        message = AIMessage("Retrieving context about data schema...")

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)

    return {
        "query": query,
        "documents": documents,
        "messages": [message],
    }


async def generate_schema(state: dict) -> dict:
    """
    Generate answer
    """
    query = state["query"]
    documents = state["documents"]

    try:
        query = "Using the AIND metadata " + query
        message = await schema_chain.ainvoke(
            {"query": query, "context": documents}
        )
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)

    return {
        "messages": [AIMessage(str(message))],
        "generation": message,
    }
