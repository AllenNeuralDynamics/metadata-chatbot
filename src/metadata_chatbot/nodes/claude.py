"""GAMER that connect to Claude"""

from typing import Annotated, Literal

from langchain import hub
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict

from metadata_chatbot.nodes.utils import HAIKU_3_5_LLM

# Generating response from previous context
prompt = ChatPromptTemplate.from_template(
    "Answer {query} based on the following texts: {context}"
)
summary_chain = prompt | HAIKU_3_5_LLM | StrOutputParser()

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


structured_llm_router = HAIKU_3_5_LLM.with_structured_output(RouteQuery)
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

    if len(state["messages"]) > 6:
        chat_history = await chat_history_chain.ainvoke(
            {"chat_history": state["messages"][-6:]}
        )
        # print(chat_history)
    else:
        chat_history = state["messages"]

    source = await datasource_router.ainvoke(
        {"query": query, "chat_history": chat_history}
    )

    message = AIMessage("Connecting to relevant data source..")

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
