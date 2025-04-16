import json
from typing import Annotated, Dict, List, Literal, Optional, TypedDict

import re

import langgraph
from langchain import hub
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from metadata_chatbot.models import sonnet_model
from metadata_chatbot.nodes.mongodb import (  # should_continue,
    tools,
)
from metadata_chatbot.retrievers.data_schema_retriever import (
    DataSchemaRetriever,
)
from metadata_chatbot.utils import (
    HAIKU_3_5_LLM,
    SONNET_3_5_LLM,
    SONNET_3_7_LLM,
)
from metadata_chatbot.models import sonnet_agent
from metadata_chatbot.test_models import mongodb_agent
from metadata_chatbot.test_models import (  # mongodb_agent,
    tool_summarizer_agent,
)
class CodeExecutor(TypedDict):
    """Route a user query to the most relevant datasource."""

    python_code: Annotated[
        str,
        ...,
        (
            "String representation of pythonic code with the "
            "necessary imports and print statements"
        ),
    ]


structured_code_executor = SONNET_3_7_LLM.with_structured_output(CodeExecutor)

prompt = hub.pull("eden19/python_formatter")
python_executor_model = prompt | structured_code_executor

fix_prompt = hub.pull("eden19/fix_code")
fix_code_model = fix_prompt | structured_code_executor

python_repl = PythonREPL()

class ShouldContinue(TypedDict):
    """Route a user query to the most relevant datasource."""

    score: Annotated[
        Literal["yes", "no"],
        ...,
        (
            "No if the query only asks for code, yes if otherwise"
        ),
    ]


should_continue_executor = HAIKU_3_5_LLM.with_structured_output(ShouldContinue)
prompt = hub.pull("eden19/should_continue_execute")
should_continue_model = prompt | should_continue_executor

class RetrievalGrader(TypedDict):
    """Route a user query to the most relevant datasource."""

    score: Annotated[
        Literal["yes", "no"],
        ...,
        (
            "No if the query doesn't answer the question, yes if otherwise"
        ),
    ]


retrieval_grader = HAIKU_3_5_LLM.with_structured_output(RetrievalGrader)

prompt = hub.pull("eden19/code_retrieval_grader")
code_grader_model = prompt | retrieval_grader 


import asyncio

async def database_query(state:dict):
    query = state['query']
    result = await mongodb_agent.ainvoke({"query": query})
    return {"messages": [result]}

async def python_formatter(state:dict):
    query = state['query']
    mongodb_query = state['messages'][-1]
    answer = await python_executor_model.ainvoke({"query": query,
                                                "mongodb_query": mongodb_query})
    python_code = answer['python_code']
    return {"python_code": python_code, "generation": python_code, "repeat_count": 0}  

async def python_executor(state: dict):
    python_code = state["python_code"]
    try:
        answer = python_repl.run(python_code)
        return {"generation": answer, "tool_output_size":len(answer), "error": None}
    except Exception as e:
        return {"generation": str(e), "tool_output_size":0, "error": str(e)}
    
async def should_continue(state: dict):
    query = state['query']
    answer = await should_continue_model.ainvoke({"query": query})
    return answer['score']

def answer_grader(state: dict):  # Changed to regular function for conditional edge
    """Check if tool output answers user query or if there was an error"""
    if state.get("error"):
        # If there's an error, go to fix_code
        return "fix_code"
    
    query = state["query"]
    tool_output = state["generation"]
    repeat_count = state.get("repeat_count", 0)
    
    if repeat_count >= 2:
        return "end"

    response = code_grader_model.invoke(
        {"query": query, "tool_output": tool_output}
    )
    if response["score"] == "yes":
        return "end"
    else:
        return "fix_code"
    
async def fix_code(state: dict):
    """Fix the Python code based on the error message."""
    query = state['query']
    original_code = state["python_code"]
    repeat_count = state.get("repeat_count", 0) + 1  # Increment counter
    prev_answer = state['generation']

    
    fixed_code_response = await fix_code_model.ainvoke({"query":query,
                                                        "original_code": original_code,
                                                        "prev_answer": prev_answer})
    
    try:
        python_code = fixed_code_response['python_code']
    except Exception as e:
        python_code = "There was an error in the code generation: {e}"
    
    return {"python_code": python_code, "repeat_count": repeat_count}

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes --
    messages: Conversation between user and GAMER stored in a list
    query: Question asked by user
    generation: LLM generation
    python_code: Python code to execute
    error: Error message if execution failed
    """
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    generation: str
    python_code: Optional[str]
    repeat_count: Optional[int]

# Create the workflow
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("database_query", database_query)
workflow.add_node("python_formatter", python_formatter)
workflow.add_node("python_executor", python_executor)
workflow.add_node("fix_code", fix_code)

# Add edges
workflow.add_edge(START, "database_query")
workflow.add_edge("database_query", "python_formatter")
workflow.add_conditional_edges(
    "python_formatter",
    should_continue,
    {
        "yes": "python_executor",
        "no": END
    }
)
# Add conditional routing
workflow.add_conditional_edges(
    "python_executor",
    answer_grader,
    {
        "fix_code": "fix_code",
        "end": END
    }
)

# Add loop back to executor after fixing code
workflow.add_edge("fix_code", "python_executor")

app = workflow.compile()

async def main(query: str):

    inputs = {
        "messages": [HumanMessage(query)],
        "query": query
    }
    answer = await app.ainvoke(inputs)
    return answer

    # async for chunk in anthropic_app.astream(inputs):
    #     print(chunk)

    #return answer["generation"]


if __name__ == "__main__":

    print(asyncio.run(main("find the number of animals in the thalamus in the middle project who received injections in the following coordinate: AP: 2.8, ML: 0.2, DV: 0.6?")))