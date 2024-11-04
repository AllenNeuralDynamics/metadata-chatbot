from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain import hub
from typing import Literal
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from aind_data_access_api.document_db import MetadataDbClient
from typing_extensions import Annotated, TypedDict

MODEL_ID_SONNET_3 = "anthropic.claude-3-sonnet-20240229-v1:0"
MODEL_ID_SONNET_3_5 = "anthropic.claude-3-sonnet-20240229-v1:0"
SONNET_3_LLM = ChatBedrock(
    model_id= MODEL_ID_SONNET_3,
    model_kwargs= {
        "temperature": 0
    }
)

SONNET_3_5_LLM = ChatBedrock(
    model_id= MODEL_ID_SONNET_3_5,
    model_kwargs= {
        "temperature": 0
    }
)

# Determining if entire database needs to be surveyed
class RouteQuery(TypedDict):
    """Route a user query to the most relevant datasource."""

    reasoning: Annotated[str, ..., "Give a one sentence justification for the chosen method"]
    datasource: Annotated[Literal["vectorstore", "direct_database"], ..., "Given a user question choose to route it to the direct database or its vectorstore."]

structured_llm_router = SONNET_3_LLM.with_structured_output(RouteQuery)
router_prompt = hub.pull("eden19/query_rerouter")
datasource_router = router_prompt | structured_llm_router

# Tool to survey entire database 
API_GATEWAY_HOST = "api.allenneuraldynamics-test.org"
DATABASE = "metadata_vector_index"
COLLECTION = "curated_assets"

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

    result = docdb_api_client.aggregate_docdb_records(
        pipeline=agg_pipeline
    )
    return result
        
tools = [aggregation_retrieval]
db_prompt = hub.pull("eden19/entire_db_retrieval")
db_surveyor_agent = create_tool_calling_agent(SONNET_3_LLM, tools, db_prompt)
query_retriever = AgentExecutor(agent=db_surveyor_agent, tools=tools, return_intermediate_steps = True, verbose=False)


# Generating appropriate filter
class FilterGenerator(TypedDict):
    """MongoDB filter to be applied before vector retrieval"""

    filter_query: Annotated[dict, ..., "MongoDB filter"]
    top_k: int = Annotated[dict, ..., "MongoDB filter"]

filter_prompt = hub.pull("eden19/filtergeneration")
filter_generator_llm = SONNET_3_LLM.with_structured_output(FilterGenerator)
filter_generation_chain = filter_prompt | filter_generator_llm


# Check if retrieved documents answer question
class RetrievalGrader(TypedDict):
    """Relevant material in the retrieved document + Binary score to check relevance to the question"""

    relevant_context:Annotated[str, ..., "Relevant context extracted from document that helps directly answer the question"]
    binary_score: Annotated[Literal["yes", "no"], ..., "Retrieved documents are relevant to the query, 'yes' or 'no'"]

retrieval_grader = SONNET_3_5_LLM.with_structured_output(RetrievalGrader)
retrieval_grade_prompt = hub.pull("eden19/retrievalgrader")
doc_grader = retrieval_grade_prompt | retrieval_grader

# Generating response to documents retrieved from the vector index
answer_generation_prompt = hub.pull("eden19/answergeneration")
rag_chain = answer_generation_prompt | SONNET_3_5_LLM | StrOutputParser()

# Generating response to documents retrieved from the database
db_answer_generation_prompt = hub.pull("eden19/db_answergeneration")
db_rag_chain = db_answer_generation_prompt | SONNET_3_5_LLM | StrOutputParser()

