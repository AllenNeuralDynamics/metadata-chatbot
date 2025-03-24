"""Langgraph workflow for GAMER"""

import warnings
from typing import Annotated, List, Optional

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from metadata_chatbot.nodes.claude import (
    determine_route,
    generate_chat_history,
    route_question,
    should_summarize,
    summarize_conversation,
)
from metadata_chatbot.nodes.data_schema import (
    generate_schema,
    retrieve_schema,
)
from metadata_chatbot.nodes.mongodb import (
    call_model,
    generate_mongodb,
    tool_node,
    tool_summarizer,
)
from metadata_chatbot.nodes.vector_index import (
    filter_generator,
    generate_VI,
    grade_documents,
    retrieve_VI,
    route_to_mongodb,
)

warnings.filterwarnings("ignore")


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


workflow = StateGraph(GraphState)

workflow.add_node(route_question)
workflow.add_node("database_query", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("data_schema_query", retrieve_schema)
workflow.add_node(filter_generator)
workflow.add_node("retrieve", retrieve_VI)
workflow.add_node(grade_documents)
workflow.add_node(generate_VI)
workflow.add_node(generate_chat_history)
workflow.add_node(generate_schema)
workflow.add_node(generate_mongodb)
workflow.add_node(summarize_conversation)


workflow.add_edge(START, "route_question")
workflow.add_conditional_edges(
    "route_question",
    determine_route,
    {
        "direct_database": "database_query",
        "vectorstore": "filter_generator",
        "claude": "generate_chat_history",
        "data_schema": "data_schema_query",
    },
)

# data schema route
workflow.add_edge("data_schema_query", "generate_schema")
# workflow.add_edge("generate_schema", END)
workflow.add_conditional_edges(
    "generate_schema",
    should_summarize,
    {"summarize": "summarize_conversation", "end": END},
)

# claude route
# workflow.add_edge("generate_claude", END)
workflow.add_conditional_edges(
    "generate_chat_history",
    should_summarize,
    {"summarize": "summarize_conversation", "end": END},
)

# mongodb route
workflow.add_edge("database_query", "tools")
workflow.add_conditional_edges(
    "tools",
    tool_summarizer,
    {
        "continue": "database_query",
        "end": "generate_mongodb",
    },
)
workflow.add_conditional_edges(
    "generate_mongodb",
    should_summarize,
    {"summarize": "summarize_conversation", "end": END},
)

# vector index route
workflow.add_edge("filter_generator", "retrieve")
workflow.add_conditional_edges(
    "retrieve",
    route_to_mongodb,
    {
        "route_query": "database_query",
        "grade_documents": "grade_documents",
    },
)
# workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "generate_VI")
# workflow.add_edge("generate_vi", END)
workflow.add_conditional_edges(
    "generate_VI",
    should_summarize,
    {"summarize": "summarize_conversation", "end": END},
)

workflow.add_edge("summarize_conversation", END)

app = workflow.compile()


async def stream_response(inputs, config, app, prev_generation=""):
    """Stream responses in each node in workflow"""

    async for output in app.astream(
        inputs, config, stream_mode=["values", "updates"]
    ):
        # message = output["messages"][-1]
        ai_message = output[1]

        if (
            "generation" in ai_message
            and ai_message["generation"] != prev_generation
        ):
            message = ai_message["generation"]
            # print(message)
            yield {"type": "final_response", "content": message}

        elif output[0] == "values":
            message = ai_message["messages"][-1]

            if isinstance(message, AIMessage):
                if message.tool_calls:
                    yield {
                        "type": "intermediate_steps",
                        "content": message.content[0]["text"],
                    }
                    yield {
                        "type": "agg_pipeline",
                        "content": message.tool_calls[0][
                            "args"
                        ],  # ["agg_pipeline"],
                    }
                elif isinstance(message.content, str):
                    yield {
                        "type": "backend_process",
                        "content": message.content,
                    }
                elif isinstance(message.content[0]["text"], str):
                    yield {
                        "type": "final_response",
                        "content": message.content[0]["text"],
                    }

            if isinstance(message, ToolMessage):
                yield {
                    "type": "tool_response",
                    "content": "Retrieved output from MongoDB: ",
                }
                yield {"type": "tool_output", "content": message.content}


# from langchain_core.messages import HumanMessage
# import asyncio

# query = ""

# async def new_astream(query):

#     inputs = {
#         "messages": [HumanMessage(query)],
#     }

#     config = {}

#     async for result in stream_response(inputs,config,app):
#         print(result)  # Process the yielded results


# # Run the main coroutine with asyncio
# asyncio.run(new_astream(query))
