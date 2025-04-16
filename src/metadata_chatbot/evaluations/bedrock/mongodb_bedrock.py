import warnings, json
from typing import Annotated, List, Optional

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from metadata_chatbot.nodes.mongodb import (
    call_model,
    generate_mongodb,
    tool_summarizer,
)
from metadata_chatbot.tools import tools


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



    if outputs:
        tool_output_size = sum(len(output.content.encode('utf-8')) for output in outputs)
    else:
        tool_output_size = 0

    return {
        "messages": outputs,
        "tool_output": outputs,
        "mongodb_query": agg_pipeline,
        "tool_output_size": tool_output_size,
    }


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
    tool_output_size: Optional[int]


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

bedrock_app = workflow.compile()

# async def main(query: str):

#     inputs = {
#         "messages": [HumanMessage(query)],
#         "query": query
#     }

#     answer = await bedrock_app.ainvoke(inputs)

#     return answer["generation"]


# if __name__ == "__main__":
#     import asyncio

#     asyncio.run(main("what is the genotype of mouse 740955"))
