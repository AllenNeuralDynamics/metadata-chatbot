"""GAMER nodes that connect to MongoDB"""

import json
import warnings
from typing import Annotated, List, Optional
import asyncio

import botocore
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from metadata_chatbot.test_models import (  # mongodb_agent,
    mongodb_agent,
    mongodb_summary_agent,
    tool_summarizer_agent,
)
from metadata_chatbot.test_models import get_records, aggregation_retrieval
from langchain_core.callbacks import get_usage_metadata_callback



async def call_model(state: dict):
    """
    Invoking LLM to call tools
    """
    messages = state["messages"]
    try:
        with get_usage_metadata_callback() as cb:

            if state.get("error", None):
                messages.append(state["error"])
            response = await mongodb_agent.ainvoke(messages)
            token_metadata = cb.usage_metadata
            cache_metadata = token_metadata['claude-3-7-sonnet-20250219']['input_token_details']
        
        token_dict = state.get("token_metadata", {})
        token_dict_copy = token_dict.copy()
        token_dict_copy["database_query"] = token_metadata

        cache_dict = state.get("cache_metadata", {})
        cache_dict_copy = cache_dict.copy()
        cache_dict_copy["database_query"] = cache_metadata

    except botocore.exceptions.EventStreamError as e:
        response = (
            "An error has occured:"
            f"Requested information exceeds model's context length: {e}"
        )

    return {"messages": [response],
            "token_metadata" : token_dict_copy,
            "cache_metadata": cache_dict_copy}

tools_by_name = {"get_records": get_records,
                 "aggregation_retrieval": aggregation_retrieval}

async def tool_node(state: dict):
    """
    Retrieving information from MongoDB with tools
    """
    outputs = []
    agg_pipeline = {}

    for i, tool_call in enumerate(state["messages"][-1].tool_calls):

        agg_pipeline[f"tool_call_{i}"] = tool_call

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

        tool_output_size = sum(len(output.content.encode('utf-8')) for output in outputs)

    return {
        "messages": outputs,
        "tool_output": outputs,
        "mongodb_query": agg_pipeline,
        "tool_output_size": tool_output_size,
    }


async def tool_summarizer(state: dict):
    """Check if tool output answers user query"""
    query = state["query"]
    tool_output = state["tool_output"][0].content

    if len(tool_output) > 10000:  # Adjust threshold as needed
        # Simple chunking approach
        tool_output = (
            tool_output[:10000] + "... [Content truncated due to length]"
        )

    try:
        response = await tool_summarizer_agent.ainvoke(
            {"query": query, "tool_output": tool_output}
        )

        if response["query_tool_match"] == "yes":
            return "end"
        else:
            return "continue"

    except Exception as e:
        error_message = str(e)
        # Check if it's the specific context length error
        if "validationException" in error_message:
            # Log the error
            # print(f"Context too long for tool output: {error_message}")

            state["error"] = HumanMessage(
                content="The tool output was too long for processing."
            )

            return "continue"
        else:
            raise


async def generate_mongodb(state: dict):
    """Generate response to user query based on tool output"""
    query = state["query"]
    tool_output = state["tool_output"]
    mongodb_query = state["mongodb_query"]

    with get_usage_metadata_callback() as cb:

        response = await mongodb_summary_agent.ainvoke(
            {"query": query, "tool_call": mongodb_query, "documents": tool_output}
        )
        token_metadata = cb.usage_metadata
        cache_metadata = token_metadata['claude-3-5-haiku-20241022']['input_token_details']
    
    token_dict = state.get("token_metadata", {})
    token_dict_copy = token_dict.copy()
    token_dict_copy["generate_mongodb"] = token_metadata

    cache_dict = state.get("cache_metadata", {})
    cache_dict_copy = cache_dict.copy()
    cache_dict_copy["generate_mongodb"] = cache_metadata

    return {"generation": response,
            "token_metadata" : token_dict_copy,
            "cache_metadata": cache_dict_copy}


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes --
    messages: Conversation between user and GAMER stored in a list
    query: Question asked by user
    chat_history: Summary of conversation
    generation: LLM generation
    data_source: Chosen db to query
    documents: List of documents
    filter: Used before vector retrieval to minimize the size of db
    top_k: # of docs to retrieve from VI
    tool_output: Retrieved info from MongoDB
    """

    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    chat_history: Optional[str]
    generation: str
    data_source: str
    documents: Optional[List[str]]
    filter: Optional[dict]
    top_k: Optional[int]
    tool_output: Optional[List[ToolMessage]]
    route_to_mongodb: Optional[bool]
    mongodb_query: Optional[dict]
    error: Optional[str]
    token_metadata: Optional[dict]
    cache_metadata: Optional[dict]


workflow = StateGraph(GraphState)

workflow.add_node("database_query", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node(generate_mongodb)

workflow.add_edge(START, "database_query")
workflow.add_edge("database_query", "tools")
workflow.add_conditional_edges(
    "tools",
    tool_summarizer,
    {
        "continue": "database_query",
        "end": "generate_mongodb",
    },
)
workflow.add_edge("generate_mongodb", END)

anthropic_app = workflow.compile()

# async def main(query: str):

#     inputs = {
#         "messages": [HumanMessage(query)],
#         "query": query
#     }
#     answer = await anthropic_app.ainvoke(inputs)
#     return answer

#     # async for chunk in anthropic_app.astream(inputs):
#     #     print(chunk)

#     #return answer["generation"]


# if __name__ == "__main__":

#     print(asyncio.run(main("What is the genotype of subject 740955")))

