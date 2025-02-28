"""GAMER nodes that connect to MongoDB"""

import json

import botocore
from aind_data_access_api.document_db import MetadataDbClient
from langchain import hub
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from metadata_chatbot.nodes.utils import HAIKU_3_5_LLM, SONNET_3_7_LLM
from metadata_chatbot.nodes.vector_index import doc_grader

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
    Executes a MongoDB aggregation pipeline for complex data
    transformations and analysis.

    WHEN TO USE THIS FUNCTION:
    - When you need to perform multi-stage data processing operations
    - For complex queries requiring grouping, filtering, sorting in sequence
    - When you need to calculate aggregated values (sums, averages, counts)
    - For data transformation operations that can't be done with simple queries

    NOT RECOMMENDED FOR:
    - Simple document retrieval (use get_records instead)
    - When you only need to filter data without transformations
    Executes a MongoDB aggregation pipeline and returns the aggregated results.

    This function processes complex queries using MongoDB's aggregation
    framework, allowing for data transformation, filtering, grouping, and
    analysis operations. It handles the execution of multi-stage aggregation
    pipelines and provides error handling for failed aggregations.

    Parameters
    ----------
    agg_pipeline : list
        A list of dictionary objects representing MongoDB aggregation stages.
        Each stage should be a valid MongoDB aggregation operator.
        Common stages include: $match, $project, $group, $sort, $unwind.

    Returns
    -------
    list
        Returns a list of documents resulting from the aggregation pipeline.
        If an error occurs, returns an error message string describing
        the exception.

    Notes
    -----
    - Include a $project stage early in the pipeline to reduce data transfer
    - Avoid using $map operator in $project stages as it requires array inputs
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
def get_records(filter: dict = {}, projection: dict = {}) -> dict:
    """
    Retrieves documents from MongoDB database using simple filters and
    projections.

    WHEN TO USE THIS FUNCTION:
    - For straightforward document retrieval based on specific criteria
    - When you need only a subset of fields from documents
    - When the query logic doesn't require multi-stage processing
    - For better performance with simpler queries

    NOT RECOMMENDED FOR:
    - Complex data transformations (use aggregation_retrieval instead)
    - Grouping operations or calculations across documents
    - Joining or relating data across collections

    Parameters
    ----------
    filter : dict
        MongoDB query filter to narrow down the documents to retrieve.
        Example: {"subject.sex": "Male"}
        If empty dict object, returns all documents.

    projection : dict
        Fields to include or exclude in the returned documents.
        Use 1 to include a field, 0 to exclude.
        Example: {"subject.genotype": 1, "_id": 0}
        will return only the genotype field.
        If empty dict object, returns all documents.

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


tools = [get_records, aggregation_retrieval]

template = hub.pull("eden19/entire_db_retrieval")


sonnet_model = SONNET_3_7_LLM.bind_tools(tools)
sonnet_agent = template | sonnet_model

summary_prompt = hub.pull("eden19/mongodb_summary")
summary_agent = summary_prompt | HAIKU_3_5_LLM


tools_by_name = {tool.name: tool for tool in tools}


async def tool_node(state: dict):
    """
    Determining if call to MongoDB is required
    """
    query = state["query"]
    outputs = []
    use_summary = False

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

        score = await doc_grader.ainvoke(
            {"query": query, "document": json.dumps(tool_result)}
        )

        if score == "yes":
            use_summary = True

        return {"messages": outputs, "use_tool_summary": use_summary}


async def call_model(state: dict):
    """
    Invoking LLM to generate response
    """
    try:
        if state.get("use_tool_summary", False) is True:
            response = await summary_agent.ainvoke(
                {
                    "query": state["query"],
                    "documents": state["messages"][-1].content,
                }
            )
            return {"generation": response, "use_tool_summary": False}

        else:
            response = await sonnet_agent.ainvoke(state["messages"])

    except botocore.exceptions.EventStreamError as e:
        response = (
            "An error has occured:"
            f"Requested information exceeds model's context length: {e}"
        )

    return {"messages": [response], "use_tool_summary": False}


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
