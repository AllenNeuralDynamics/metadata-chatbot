"""Langgraph workflow for GAMER"""

import warnings
from typing import Annotated, List, Optional

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from metadata_chatbot.nodes.claude import (
    determine_route,
    generate_summary,
    route_question,
)
from metadata_chatbot.nodes.data_schema import (
    generate_schema,
    retrieve_schema,
)
from metadata_chatbot.nodes.mongodb import (
    call_model,
    should_continue,
    tool_node,
)
from metadata_chatbot.nodes.vector_index import (
    filter_generator,
    generate_VI,
    grade_documents,
    retrieve_VI,
)

warnings.filterwarnings("ignore")


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        query: question asked by user
        generation: LLM generation
        documents: list of documents
    """

    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    chat_history: Optional[str]
    generation: str
    data_source: str
    documents: Optional[List[str]]
    filter: Optional[dict]
    top_k: Optional[int]
    use_tool_summary: False


workflow = StateGraph(GraphState)

workflow.add_node("route_question", route_question)
workflow.add_node("database_query", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("data_schema_query", retrieve_schema)
workflow.add_node("filter_generation", filter_generator)
workflow.add_node("retrieve", retrieve_VI)
workflow.add_node("document_grading", grade_documents)
workflow.add_node("generate_vi", generate_VI)
workflow.add_node("generate_summary", generate_summary)
workflow.add_node("generate_schema", generate_schema)

workflow.add_edge(START, "route_question")
workflow.add_conditional_edges(
    "route_question",
    determine_route,
    {
        "direct_database": "database_query",
        "vectorstore": "filter_generation",
        "claude": "generate_summary",
        "data_schema": "data_schema_query",
    },
)

# data schema route
workflow.add_edge("data_schema_query", "generate_schema")
workflow.add_edge("generate_schema", END)

# claude route
workflow.add_edge("generate_summary", END)

# mongodb route
workflow.add_conditional_edges(
    "database_query",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
workflow.add_edge("tools", "database_query")

# vector index route
workflow.add_edge("filter_generation", "retrieve")
workflow.add_edge("retrieve", "document_grading")
workflow.add_edge("document_grading", "generate_vi")
workflow.add_edge("generate_vi", END)

app = workflow.compile()


async def stream_response(inputs, config, app, prev_generation):
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

# query = "hi"
# prev_generation = "hi"

# async def new_astream(query):

#     inputs = {
#         "messages": [HumanMessage(query)],
#     }

#     config = {}

#     async for result in stream_response(inputs,config,app,prev_generation):
#         print(result)  # Process the yielded results


# # Run the main coroutine with asyncio
# asyncio.run(new_astream(query))
