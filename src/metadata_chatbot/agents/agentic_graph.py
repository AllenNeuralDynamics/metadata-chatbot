from pydantic import BaseModel, Field
from langchain_aws import ChatBedrock
from langchain import hub
import logging
from typing import Literal
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from aind_data_access_api.document_db_ssh import DocumentDbSSHClient, DocumentDbSSHCredentials
from langchain.agents import AgentExecutor, create_tool_calling_agent

logging.basicConfig(filename='agentic_graph.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode="w")

MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
LLM = ChatBedrock(
    model_id= MODEL_ID,
    model_kwargs= {
        "temperature": 0
    }
)

#determining if entire database needs to be surveyed
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "direct_database"] = Field(
        description="Given a user question choose to route it to the direct database or its vectorstore.",
    )

structured_llm_router = LLM.with_structured_output(RouteQuery)
router_prompt = hub.pull("eden19/query_rerouter")
datasource_router = router_prompt | structured_llm_router


# Queries that require surveying the entire database (like count based questions)
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

db_surveyor_agent = create_tool_calling_agent(LLM, tools, prompt)
db_surveyor = AgentExecutor(agent=db_surveyor_agent, tools=tools, verbose=False)


# Processing query
class ProcessQuery(BaseModel):
    """Binary score to check whether query requires retrieval to be filtered with metadata information to achieve accurate results."""

    binary_score: str = Field(
        description="Query requires further filtering during retrieval process, 'yes' or 'no'"
    )

query_grader = LLM.with_structured_output(ProcessQuery)
query_grade_prompt = hub.pull("eden19/processquery")
query_grader = query_grade_prompt | query_grader
# query_grade = query_grader.invoke({"query": question}).binary_score

# Generating appropriate filter
class FilterGenerator(BaseModel):
    """MongoDB filter to be applied before vector retrieval"""

    filter_query: dict = Field(description="MongoDB filter")
    top_k: int = Field(description="Number of documents to retrieve from the database")

filter_prompt = hub.pull("eden19/filtergeneration")
filter_generator_llm = LLM.with_structured_output(FilterGenerator)

filter_generation_chain = filter_prompt | filter_generator_llm
# filter = filter_generation_chain.invoke({"query": question}).filter_query

# Check if retrieved documents answer question
class RetrievalGrader(BaseModel):
    """Binary score to check whether retrieved documents are relevant to the question"""
    binary_score: str = Field(
        description="Retrieved documents are relevant to the query, 'yes' or 'no'"
    )

retrieval_grader = LLM.with_structured_output(RetrievalGrader)
retrieval_grade_prompt = hub.pull("eden19/retrievalgrader")
doc_grader = retrieval_grade_prompt | retrieval_grader
# doc_grade = doc_grader.invoke({"query": question, "document": doc}).binary_score
# logging.info(f"Retrieved document matched query: {doc_grade}")

# Generating response to documents
answer_generation_prompt = hub.pull("eden19/answergeneration")
rag_chain = answer_generation_prompt | LLM | StrOutputParser()
# generation = rag_chain.invoke({"documents": doc, "query": question})
# logging.info(f"Final answer: {generation}")