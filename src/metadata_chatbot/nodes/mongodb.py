"""GAMER nodes that connect to MongoDB"""

import json

import botocore
from aind_data_access_api.document_db import MetadataDbClient
from langchain import hub
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from typing import Optional

from metadata_chatbot.nodes.utils import HAIKU_3_5_LLM, SONNET_3_5_LLM, SONNET_3_7_LLM

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
    Executes a MongoDB aggregation pipeline and returns the aggregated results.

    This function processes complex queries using MongoDB's aggregation framework,
    allowing for data transformation, filtering, grouping, and analysis operations.
    It handles the execution of multi-stage aggregation pipelines and provides
    error handling for failed aggregations.

    Parameters
    ----------
    agg_pipeline : list
        A list of dictionary objects representing MongoDB aggregation stages.
        Each stage should be a valid MongoDB aggregation operator.
        Common stages include: \$match, \$project, \$group, \$sort, \$unwind.

    Returns
    -------
    list
        If successful, returns a list of documents resulting from the aggregation pipeline.
        If an error occurs, returns an error message string describing the exception.

    Notes
    -----
    - For optimal performance, include a \$project stage early in the pipeline to reduce data transfer
    - Avoid using \$map operator in \$project stages as it requires array inputs
    - Complex pipelines may impact performance; consider indexing frequently queried fields
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
    
@tool
def get_records(filter: dict = None, projection: Optional[dict] = None) -> dict:
    """
    Retrieves documents from MongoDB database based on specified filters and projections.

    This function interfaces with a MongoDB database through a document database API client
    to fetch records. It supports filtering and field projection to optimize query performance
    and minimize data transfer.

    Parameters
    ----------
    filter : dict, optional
        MongoDB query filter specification to narrow down the documents to retrieve.
        Example: {"subject.sex": "Male"} will return only records for male subjects.
        If None, returns all documents.

    projection : dict, optional
        Specification of fields to include or exclude in the returned documents.
        Use 1 to include a field, 0 to exclude.
        Example: {"subject.genotype": 1, "_id": 0} will return only the genotype field.
        If None, returns all fields.

    Returns
    -------
    list
        List of dictionary objects representing the matching documents.
        Each dictionary contains the requested fields based on the projection.

    """

    records = docdb_api_client.retrieve_docdb_records(
        filter_query=filter,
        projection=projection,
    )

    return records


tools = [aggregation_retrieval, get_records]

template = hub.pull("eden19/shortened_entire_db_retrieval")
model = HAIKU_3_5_LLM.bind_tools(tools)
retrieval_agent = template | model

sonnet_model = SONNET_3_5_LLM.bind_tools(tools)
sonnet_agent = template | sonnet_model

summary_prompt = hub.pull("eden19/mongodb_summary")
summary_agent = summary_prompt | SONNET_3_7_LLM

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
        if type(state["messages"][-1]) == ToolMessage:
            query = state['query']
            #print(state['messages'][-1])
            response = await summary_agent.ainvoke(            
                {"query": query, "documents": state["messages"][-1].content}
            )
            return {"generation": response}
        else:
            response = await sonnet_agent.ainvoke(state["messages"])
        #print(response.tool_calls)


    except botocore.exceptions.EventStreamError as e:
        response = (
            "An error has occured:"
            f"Requested information exceeds model's context length: {e}"
        )
        
        
    return {"messages": [response]}

    # if isinstance(response, list):
    #     response = str(response)



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
