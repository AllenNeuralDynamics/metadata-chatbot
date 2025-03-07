"""GAMER nodes that only connect to Claude"""

from typing import Annotated, Literal

from langchain import hub
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from typing_extensions import TypedDict

from metadata_chatbot.nodes.utils import HAIKU_3_5_LLM, SONNET_3_7_LLM

# Summarizing chat_history
summary_prompt = ChatPromptTemplate.from_template(
    "Succinctly summarize the chat history of the conversation "
    "{chat_history}, including the user's queries"
    " and the relevant answers retaining important details"
)
chat_history_chain = summary_prompt | HAIKU_3_5_LLM | StrOutputParser()


# Determining if entire database needs to be surveyed
class RouteQuery(TypedDict):
    """Route a user query to the most relevant datasource."""

    datasource: Annotated[
        Literal["vectorstore", "direct_database", "claude", "data_schema"],
        ...,
        (
            "Given a user question choose to route it to the direct database"
            "or its vectorstore. If a question can be answered without"
            "retrieval, route to claude. If a question is about the"
            "schema/structure/definitions, route to data schema"
        ),
    ]


structured_llm_router = SONNET_3_7_LLM.with_structured_output(RouteQuery)
router_prompt = hub.pull("eden19/query_rerouter")
datasource_router = router_prompt | structured_llm_router


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

    data_source = source["datasource"]

    if data_source == "direct_database":
        message = AIMessage(
            "Connecting to MongoDB and generating a query..."
            )
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
            "Reviewing the AIND data schema and finding relevant information..."
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

# Generating response from previous context
prompt = ChatPromptTemplate.from_template(
    "Answer {query} based on the following texts: {context}"
)
summary_chain = prompt | HAIKU_3_5_LLM | StrOutputParser()

async def generate_chat_history(state: dict) -> dict:
    """
    Generate answer
    """

    if "query" in state and state["query"] is not None:
        query = state["query"]
    else:
        query = state["messages"][-1].content
    chat_history = state["messages"]

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
    # First, we summarize the conversation
    summary = state.get("chat_history", "")
    if summary:
        # If a summary already exists, we use a different system prompt
        # to summarize it than if one didn't
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = str(state["messages"] + [HumanMessage(content=summary_message)])
    response = await HAIKU_3_5_LLM.ainvoke(messages)
    # We now need to delete messages that we no longer want to show up
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"chat_history": response.content, "messages": delete_messages}