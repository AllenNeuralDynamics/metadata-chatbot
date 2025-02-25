"""GAMER nodes that connect to MongoDB"""

import json

import botocore
from aind_data_access_api.document_db import MetadataDbClient
from langchain import hub
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from metadata_chatbot.nodes.utils import HAIKU_3_5_LLM, SONNET_3_5_LLM

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

template = hub.pull("eden19/shortened_entire_db_retrieval")
model = HAIKU_3_5_LLM.bind_tools(tools)
retrieval_agent = template | model

sonnet_model = SONNET_3_5_LLM.bind_tools(tools)
sonnet_agent = template | sonnet_model


chain = retrieval_agent  # | tool_def | str_transform | aggregation_retrieval

tools_by_name = {tool.name: tool for tool in tools}


async def tool_node(state: dict):
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


async def call_model(state: dict):
    """
    Invoking LLM to generate response
    """
    try:
        if ToolMessage in state["messages"]:
            response = await HAIKU_3_5_LLM.ainvoke(state["messages"])
        else:
            response = await chain.ainvoke(state["messages"])
    except botocore.exceptions.EventStreamError as e:
        response = (
            "An error has occured:"
            f"Requested information exceeds model's context length: {e}"
        )

    # if isinstance(response, list):
    #     response = str(response)

    return {"messages": [response]}


# Define the conditional edge that determines whether to continue or not
async def should_continue(state: dict):
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
