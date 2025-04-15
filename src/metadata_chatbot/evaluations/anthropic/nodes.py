import asyncio

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from metadata_chatbot.test_models import (
    doc_grader,
    filter_generation_chain,
    rag_chain,
    schema_chain,
    datasource_router, 
    summary_chain
)
from metadata_chatbot.retrievers.docdb_retriever import DocDBRetriever

from metadata_chatbot.retrievers.data_schema_retriever import (
    DataSchemaRetriever,
)
from metadata_chatbot.utils import HAIKU_3_5_LLM
from langchain_core.callbacks import get_usage_metadata_callback



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
        with get_usage_metadata_callback() as cb:
            result = await filter_generation_chain.ainvoke(
                {"query": query, "chat_history": chat_history}
            )
            token_metadata = cb.usage_metadata

        filter = result["filter_query"]
        top_k = result["top_k"]
        message = (
            f"Using MongoDB filter: {filter} on the database "
            f"and retrieving {top_k} documents"
        )
        token_dict = state.get("token_metadata", {})
        token_dict_copy = token_dict.copy()
        token_dict_copy["filter_generator"] = token_metadata

    except Exception as ex:
        filter = None
        top_k = None
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)

    return {
        "filter": filter,
        "top_k": top_k,
        "messages": [AIMessage(message)],
        "token_metadata" : token_dict_copy
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
    with get_usage_metadata_callback() as cb:
        score = await doc_grader.ainvoke(
            {"query": query, "document": doc.page_content}
        )
        token_metadata = cb.usage_metadata
    grade = score["binary_score"]

    try:
        if grade == "yes":
            return {"content": doc.page_content, "metadata": token_metadata}
        else:
            return None
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        return  {"content": message, "metadata": token_metadata}


async def grade_documents(state: dict) -> dict:
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    query = state["query"]
    documents = state["documents"]

    token_dict = state.get("token_metadata", {}).copy()

    graded_results = await asyncio.gather(
        *[grade_doc(query, doc) for doc in documents],
        return_exceptions=True,
    )

    filtered_docs = []
    all_token_metadata = []
    
    for result in graded_results:
        if result is not None and not isinstance(result, Exception):
            filtered_docs.append(result["content"])
            all_token_metadata.append(result["metadata"])


    filtered_docs = [doc for doc in filtered_docs if doc is not None]

    token_dict["grade_documents"] = all_token_metadata

    return {
        "documents": filtered_docs,
        "messages": [
            AIMessage("Checking document relevancy to your query...")
        ],
        "token_metadata": token_dict,
    }


async def generate_VI(state: dict) -> dict:
    """
    Generate answer
    """
    query = state["query"]
    documents = state["documents"]

    try:
        with get_usage_metadata_callback() as cb:
            message = await rag_chain.ainvoke(
                {"documents": documents, "query": query}
            )
            token_metadata = cb.usage_metadata

        token_dict = state.get("token_metadata", {})
        token_dict_copy = token_dict.copy()
        token_dict_copy["generate_VI"] = token_metadata

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)

    return {
        "messages": [AIMessage(str(message))],
        "generation": message,
        "token_metadata" : token_dict_copy,
    }

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
        with get_usage_metadata_callback() as cb:
            message = await schema_chain.ainvoke(
                {"query": query, "context": documents}
            )
            token_metadata = cb.usage_metadata
        
        token_dict = state.get("token_metadata", {})
        token_dict_copy = token_dict.copy()
        token_dict_copy["generate_schema"] = token_metadata

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)

    return {
        "messages": [AIMessage(str(message))],
        "generation": message,
        "token_metadata" : token_dict_copy,
    }


async def route_question(state: dict) -> dict:
    """
    Route question to database or vectorstore
    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    query = state["messages"][-1].content

    if state.get("chat_history") is None or state.get("chat_history") == "":
        chat_history = state["messages"]
    else:
        chat_history = state["chat_history"]

    with get_usage_metadata_callback() as cb:
        source = await datasource_router.ainvoke(
            {"query": query, "chat_history": chat_history}
        )
        token_metadata = cb.usage_metadata
    
    token_dict = state.get("token_metadata", {})
    token_dict_copy = token_dict.copy()
    token_dict_copy["datasource_router"] = token_metadata

    data_source = source["datasource"]

    if data_source == "direct_database":
        message = AIMessage("Connecting to MongoDB and generating a query...")
    elif data_source == "vectorstore":
        message = AIMessage(
            "Reviewing data assets and finding relevant information..."
        )
    elif data_source == "claude":
        message = AIMessage(
            "Reviewing chat history to find relevant information..."
        )
    elif data_source == "data_schema":
        message = AIMessage(
            "Reviewing the AIND data schema and "
            "finding relevant information..."
        )

    return {
        "query": query,
        "chat_history": chat_history,
        "data_source": source["datasource"],
        "messages": [message],
        "token_metadata" : token_dict_copy,
    }


def determine_route(state: dict) -> dict:
    """Determine which route model should take"""
    data_source = state["data_source"]

    if data_source == "direct_database":
        return "direct_database"
    elif data_source == "vectorstore":
        return "vectorstore"
    elif data_source == "claude":
        return "claude"
    elif data_source == "data_schema":
        return "data_schema"


async def generate_chat_history(state: dict) -> dict:
    """
    Generate answer
    """
    query = state["query"]

    if state.get("chat_history") is None or state.get("chat_history") == "":
        chat_history = state["messages"]
    else:
        chat_history = state["chat_history"]

    try:
        message = await summary_chain.ainvoke(
            {"query": query, "context": chat_history}
        )
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)

    return {
        "messages": [AIMessage(str(message))],
        "generation": message,
    }


def should_summarize(state: dict):
    """Return the next node to execute."""
    messages = state["messages"]
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize"
    # Otherwise we can just end
    return "end"


async def summarize_conversation(state: dict):
    """Summarize chat"""
    summary = state.get("chat_history", "")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary concisely by taking into account "
            "the new messages above:"
        )
    else:
        summary_message = (
            "Create a summary of the conversation above. Be concise:"
        )

    messages = str(state["messages"] + [HumanMessage(content=summary_message)])
    response = await HAIKU_3_5_LLM.ainvoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    return {"chat_history": response.content, "messages": delete_messages}
