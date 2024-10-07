from pydantic import BaseModel, Field
from langchain_aws import ChatBedrock
from langchain import hub
from langchain_core.documents import Document
import logging, json, sys, os
from typing import List, Optional
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from docdb_retriever import DocDBRetriever
from langchain_core.tools import tool
from aind_data_access_api.document_db_ssh import DocumentDbSSHClient, DocumentDbSSHCredentials
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate



from langgraph.graph import END, StateGraph, START

MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
LLM = ChatBedrock(
    model_id= MODEL_ID,
    model_kwargs= {
        "temperature": 0
    }
)

credentials = DocumentDbSSHCredentials()
credentials.database = "metadata_vector_index"
credentials.collection = "curated_assets"
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
    with DocumentDbSSHClient(credentials=credentials) as doc_db_client:

        result = list(doc_db_client.collection.aggregate(
            pipeline=agg_pipeline
        ))
        return result
        
tools = [aggregation_retrieval]

prompt = hub.pull("eden19/entire_db_retrieval")



#Surveying entire database
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
db_surveyor_agent = create_tool_calling_agent(LLM, tools, prompt)
db_surveyor = AgentExecutor(agent=db_surveyor_agent, tools=tools, verbose=False)

from langchain_core.messages import AIMessage, HumanMessage

chat_history = []

result = db_surveyor.invoke(
    {
        "query": "what are the unique modalities in the database",
        "chat_history": chat_history
    }
)

print(result)