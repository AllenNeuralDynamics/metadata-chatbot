"""Langgraph workflow for GAMER"""

import asyncio
import warnings
from typing import Annotated, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from metadata_chatbot.agents.agentic_graph import (
    datasource_router,
    doc_grader,
    filter_generation_chain,
    rag_chain,
    summary_chain,
)
from metadata_chatbot.agents.data_schema_retriever import DataSchemaRetriever
from metadata_chatbot.agents.docdb_retriever import DocDBRetriever

warnings.filterwarnings("ignore")


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        query: question asked by user
        generation: LLM generation
        documents: list of documents
    """

    messages: Annotated[list[AnyMessage], add_messages]
    query: Optional[str]
    generation: str
    documents: Optional[List[str]]
    filter: Optional[dict]
    top_k: Optional[int]


async def route_question(state: dict) -> dict:
    """
    Route question to database or vectorstore
    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    query = state["messages"][-1].content
    chat_history = state["messages"]

    source = await datasource_router.ainvoke(
        {"query": query, "chat_history": chat_history}
    )

    if source["datasource"] == "direct_database":
        return "direct_database"
    elif source["datasource"] == "vectorstore":
        return "vectorstore"
    elif source["datasource"] == "claude":
        return "claude"
    elif source["datasource"] == "data_schema":
        return "data_schema"


def retrieve_DB(state: dict) -> dict:
    """
    Retrieves from data asset collection in prod DB
    after constructing a MongoDB query
    """

    message_iterator = []

    return {"messages": message_iterator, "generation": ""}


def retrieve_schema(state: dict) -> dict:
    """
    Retrieves info about data schema in prod DB
    """

    """
    Retrieve context from data schema collection
    """
    query = state["messages"][-1].content

    try:
        retriever = DataSchemaRetriever(k=7)
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


async def filter_generator(state: dict) -> dict:
    """
    Filter database by constructing basic MongoDB match filter
    and determining number of documents to retrieve
    """
    query = state["messages"][-1].content
    chat_history = state["messages"]

    try:
        result = await filter_generation_chain.ainvoke(
            {"query": query, "chat_history": chat_history}
        )
        filter = result["filter_query"]
        top_k = result["top_k"]
        message = AIMessage(
            f"Using MongoDB filter: {filter} on the database \
                    and retrieving {top_k} documents"
        )

    except Exception as ex:
        filter = None
        top_k = None
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)

    return {
        "query": query,
        "filter": filter,
        "top_k": top_k,
        "messages": [message],
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
        message = AIMessage(
            "Retrieving relevant documents from vector index..."
        )

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)

    return {
        "documents": documents,
        "messages": [message],
    }


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


async def generate_summary(state: dict) -> dict:
    """
    Generate answer
    """

    if "query" in state and state["query"] is not None:
        query = state["query"]
    else:
        query = state["messages"][-1].content
    chat_history = state["messages"]

    try:

        if "documents" in state:
            context = state["documents"]
        else:
            context = chat_history

        message = await summary_chain.ainvoke(
            {"query": query, "context": context}
        )
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)

    return {
        "messages": [AIMessage(str(message))],
        "generation": message,
    }


async_workflow = StateGraph(GraphState)
async_workflow.add_node("database_query", retrieve_DB)
async_workflow.add_node("data_schema_query", retrieve_schema)
async_workflow.add_node("filter_generation", filter_generator)
async_workflow.add_node("retrieve", retrieve_VI)
async_workflow.add_node("document_grading", grade_documents)
async_workflow.add_node("generate_vi", generate_VI)
async_workflow.add_node("generate_summary", generate_summary)

async_workflow.add_conditional_edges(
    START,
    route_question,
    {
        "direct_database": "database_query",
        "vectorstore": "filter_generation",
        "claude": "generate_summary",
        "data_schema": "data_schema_query",
    },
)
async_workflow.add_edge("data_schema_query", "generate_summary")
async_workflow.add_edge("generate_summary", END)
async_workflow.add_edge("database_query", END)
async_workflow.add_edge("filter_generation", "retrieve")
async_workflow.add_edge("retrieve", "document_grading")
async_workflow.add_edge("document_grading", "generate_vi")
async_workflow.add_edge("generate_vi", END)

memory = MemorySaver()
async_app = async_workflow.compile()

# query = "What are the unique modalities in the database??"

# from langchain_core.messages import HumanMessage

# query = "hi"


# async def new_astream(query):
#     async def main(query):

#         inputs = {
#             "messages": [HumanMessage(query)],
#         }
#         async for output in async_app.astream(inputs):
#             for key, value in output.items():
#                 if key != "database_query":
#                     yield value["messages"][0].content
#                 else:
#                     for message in value["messages"]:
#                         yield message
#                     yield value["generation"]

#     async for result in main(query):
#         print(result)  # Process the yielded results


# # Run the main coroutine with asyncio
# asyncio.run(new_astream(query))
