"""Langsmith agent class to communicate with DocDB"""

import json
from typing import Annotated, Sequence, TypedDict

from aind_data_access_api.document_db import MetadataDbClient
from langchain import hub
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from metadata_chatbot.agents.agentic_graph import SONNET_3_5_LLM

API_GATEWAY_HOST = "api.allenneuraldynamics.org"
DATABASE = "metadata_index"
COLLECTION = "data_assets"

docdb_api_client = MetadataDbClient(
    host=API_GATEWAY_HOST,
    database=DATABASE,
    collection=COLLECTION,
)


@tool
def aggregation_retrieval(agg_pipeline: list) -> list:
    """
    Given a MongoDB query and list of projections, this function
    retrieves and returns the relevant information in the documents.
    Use a project stage as the first stage to minimize the size of
    the queries before proceeding with the remaining steps.
    The input to $map must be an array not a string, avoid using it
    in the $project stage.

    Parameters
    ----------
    agg_pipeline
        MongoDB aggregation pipeline

    Returns
    -------
    list
        List of retrieved documents
    """
    try:
        result = docdb_api_client.aggregate_docdb_records(
            pipeline=agg_pipeline
        )
        return result

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        return message


tools = [aggregation_retrieval]
model = SONNET_3_5_LLM.bind_tools(tools)

template = hub.pull("eden19/entire_db_retrieval")
# system_prompt =  SystemMessage(system_rompt)
retrieval_agent_chain = template | model


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


tools_by_name = {tool.name: tool for tool in tools}


async def tool_node(state: AgentState):
    """
    Determining if call to MongoDB is required
    """
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = await tools_by_name[tool_call["name"]].ainvoke(
            tool_call["args"]
        )
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


async def call_model(state: AgentState):
    """
    Invoking LLM to generate response
    """
    if ToolMessage in state["messages"]:
        response = await SONNET_3_5_LLM.ainvoke(state["messages"])
    else:
        response = await retrieval_agent_chain.ainvoke(state["messages"])
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the conditional edge that determines whether to continue or not
async def should_continue(state: AgentState):
    """
    Determining if model should continue querying DocDB to answer query
    """
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
workflow.add_edge("tools", "agent")

react_agent = workflow.compile()

query = "How many records are in the database"


async def astream_input(query):
    """
    Streaming result from the react agent node
    """
    inputs = {"messages": [("user", query)]}
    async for s in react_agent.astream(inputs, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, AIMessage):
            if message.tool_calls:
                yield {
                    "type": "intermediate_steps",
                    "content": message.content[0]["text"],
                }
                yield {
                    "type": "agg_pipeline",
                    "content": message.tool_calls[0]["args"]["agg_pipeline"],
                }
            else:
                yield {
                    "type": "final_answer",
                    "content": message.content[0]["text"],
                }
        if isinstance(message, ToolMessage):
            yield {"type": "tool_response", "content": message.content}


# async def agent_astream(query):

#     async for result in astream_input(query):
#         response = result["type"]
#         if response == "intermediate_steps":
#             print(result["content"])
#         if response == "agg_pipeline":
#             print(
#                 "The MongoDB pipeline used to on the database is:",
#                 result["content"],
#             )
#         if response == "tool_response":
#             print("Retrieved output from MongoDB:", result["content"])
#         if response == "final_answer":
#             generation = result["content"]
#     return generation


# print(asyncio.run(agent_astream(query)))
