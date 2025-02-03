"""LLM agents used in GAMER"""

from typing import Literal

from langchain import hub
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import Annotated, TypedDict

MODEL_ID_SONNET_3 = "anthropic.claude-3-sonnet-20240229-v1:0"
MODEL_ID_SONNET_3_5 = "anthropic.claude-3-5-sonnet-20240620-v1:0"
MODEL_ID_HAIKU_3_5 = "anthropic.claude-3-5-haiku-20241022-v1:0"

SONNET_3_LLM = ChatBedrock(
    model_id=MODEL_ID_SONNET_3, model_kwargs={"temperature": 0}, streaming=True
)

SONNET_3_5_LLM = ChatBedrock(
    model_id=MODEL_ID_SONNET_3_5,
    model_kwargs={"temperature": 0},
    streaming=True,
)

HAIKU_3_5_LLM = ChatBedrock(
    model_id=MODEL_ID_HAIKU_3_5,
    model_kwargs={"temperature": 0},
    streaming=True,
)


# Determining if entire database needs to be surveyed
class RouteQuery(TypedDict):
    """Route a user query to the most relevant datasource."""

    datasource: Annotated[
        Literal["vectorstore", "direct_database", "claude"],
        ...,
        (
            "Given a user question choose to route it to the \
         direct database or its vectorstore."
            "If a question can be answered without retrieval, route to claude"
        ),
    ]


structured_llm_router = HAIKU_3_5_LLM.with_structured_output(RouteQuery)
router_prompt = hub.pull("eden19/query_rerouter")
datasource_router = router_prompt | structured_llm_router


# Check if retrieved documents answer question
class QueryRewriter(TypedDict):
    """Rewrite ambiguous queries"""

    binary_score: Annotated[
        Literal["yes", "no"], ..., "Query is ambiguous, 'yes' or 'no'"
    ]
    rewritten_query: Annotated[
        str, ..., "user's query, rewritten to be more specific"
    ]


query_rewriter = HAIKU_3_5_LLM.with_structured_output(QueryRewriter)
query_rewriter_prompt = hub.pull("eden19/query_rewriter")
query_rewriter_chain = query_rewriter_prompt | query_rewriter


# Generating appropriate filter
class FilterGenerator(TypedDict):
    """MongoDB filter to be applied before vector retrieval"""

    filter_query: Annotated[dict, ..., "MongoDB filter"]
    top_k: int = Annotated[dict, ..., "MongoDB filter"]


filter_prompt = hub.pull("eden19/filtergeneration")
filter_generator_llm = HAIKU_3_5_LLM.with_structured_output(FilterGenerator)
filter_generation_chain = filter_prompt | filter_generator_llm


# Check if retrieved documents answer question
class RetrievalGrader(TypedDict):
    """Relevant material in the retrieved document +
    Binary score to check relevance to the question"""

    binary_score: Annotated[
        Literal["yes", "no"],
        ...,
        "Retrieved documents are relevant to the query, 'yes' or 'no'",
    ]


retrieval_grader = SONNET_3_5_LLM.with_structured_output(RetrievalGrader)
retrieval_grade_prompt = hub.pull("eden19/retrievalgrader")
doc_grader = retrieval_grade_prompt | retrieval_grader

# Generating response to documents retrieved from the vector index
answer_generation_prompt = hub.pull("eden19/answergeneration")
rag_chain = answer_generation_prompt | SONNET_3_5_LLM | StrOutputParser()

# Generating response to documents retrieved from the database
db_answer_generation_prompt = hub.pull("eden19/db_answergeneration")
db_rag_chain = db_answer_generation_prompt | SONNET_3_5_LLM | StrOutputParser()

# Generating response from previous context
prompt = ChatPromptTemplate.from_template(
    "Answer {query} based on the following texts: {chat_history}"
)
prev_context_chain = prompt | HAIKU_3_5_LLM | StrOutputParser()
