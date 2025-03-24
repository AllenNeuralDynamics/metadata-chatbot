"""GAMER nodes that connect to MongoDB"""

import json

import botocore
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from metadata_chatbot.models import (
    mongodb_summary_agent,
    sonnet_agent,
    tool_summarizer_agent,
)
from metadata_chatbot.tools import tools


async def call_model(state: dict):
    """
    Invoking LLM to call tools
    """
    messages = state["messages"]
    try:
        if state.get("error", None):
            messages.append(state["error"])
        response = await sonnet_agent.ainvoke(messages)

    except botocore.exceptions.EventStreamError as e:
        response = (
            "An error has occured:"
            f"Requested information exceeds model's context length: {e}"
        )

    return {"messages": [response]}


tools_by_name = {tool.name: tool for tool in tools}


async def tool_node(state: dict):
    """
    Retrieving information from MongoDB with tools
    """
    outputs = []
    agg_pipeline = {}

    for i, tool_call in enumerate(state["messages"][-1].tool_calls):

        agg_pipeline = state.get("mongodb_query", {})

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

    return {
        "messages": outputs,
        "tool_output": outputs,
        "mongodb_query": agg_pipeline,
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

    response = await mongodb_summary_agent.ainvoke(
        {"query": query, "tool_call": mongodb_query, "documents": tool_output}
    )
    return {"generation": response}
