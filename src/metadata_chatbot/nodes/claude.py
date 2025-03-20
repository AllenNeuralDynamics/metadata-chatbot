"""GAMER nodes that only connect to Claude"""

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from metadata_chatbot.models import datasource_router, summary_chain
from metadata_chatbot.utils import HAIKU_3_5_LLM


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

    source = await datasource_router.ainvoke(
        {"query": query, "chat_history": chat_history}
    )

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
