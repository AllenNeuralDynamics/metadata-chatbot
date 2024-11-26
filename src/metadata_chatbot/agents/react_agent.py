from langchain_aws.chat_models.bedrock import ChatBedrock
from typing import Annotated, Sequence, TypedDict
from langchain_core.tools import tool
from langchain import hub
from aind_data_access_api.document_db import MetadataDbClient
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import json
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, END

from langchain import hub
import asyncio


MODEL_ID_SONNET_3_5 = "anthropic.claude-3-5-sonnet-20240620-v1:0"

SONNET_3_5_LLM = ChatBedrock(
    model_id= MODEL_ID_SONNET_3_5,
    model_kwargs= {
        "temperature": 0
    }
)

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
    """Given a MongoDB query and list of projections, this function retrieves and returns the 
    relevant information in the documents. 
    Use a project stage as the first stage to minimize the size of the queries before proceeding with the remaining steps.
    The input to $map must be an array not a string, avoid using it in the $project stage.

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
    except:
        return("An error has occured, try structuring the query another way!")

        
tools = [aggregation_retrieval]
model = SONNET_3_5_LLM.bind_tools(tools)

template = hub.pull("eden19/entire_db_retrieval")
#system_prompt =  SystemMessage(system_rompt)
retrieval_agent_chain = template | model

class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

tools_by_name = {tool.name: tool for tool in tools}

async def tool_node(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = await tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

async def call_model(
    state: AgentState
):
    if ToolMessage in state['messages']:
    # this is similar to customizing the create_react_agent with state_modifier, but is a lot more flexible
        response = await SONNET_3_5_LLM.ainvoke(state["messages"])
    else:
        response = await retrieval_agent_chain.ainvoke(state["messages"])
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the conditional edge that determines whether to continue or not
async def should_continue(state: AgentState):
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

async def print_stream(stream):
    async for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


# async def main():
#     inputs = {"messages": [("user", "What is the total number of record in the database?")]}
#     async for s in react_agent.astream(inputs, stream_mode="values"):
#         message = s["messages"][-1]
#         if isinstance(message, tuple):
#             print(message)
#         else:
#             message.pretty_print()
    
#     #return answer

# if __name__ == "__main__":
#     asyncio.run(main())