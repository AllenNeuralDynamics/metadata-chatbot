from typing import Annotated, Literal, TypedDict

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from metadata_chatbot.tools import tools
from metadata_chatbot.utils import (
    HAIKU_3_5_LLM,
    SONNET_3_5_LLM,
    SONNET_3_7_LLM,
)


# Determining if entire database needs to be surveyed
class RouteQuery(TypedDict):
    """Route a user query to the most relevant datasource."""

    datasource: Annotated[
        Literal["vectorstore", "direct_database", "claude", "data_schema"],
        ...,
        (
            "Given a user question choose to route it to the direct database"
            "or its vectorstore. If a question can be answered without"
            "retrieval, route to claude. If a question is about the"
            "schema/structure/definitions, route to data schema"
        ),
    ]


structured_llm_router = SONNET_3_5_LLM.with_structured_output(RouteQuery)
router_prompt = hub.pull("eden19/query_rerouter")
datasource_router = router_prompt | structured_llm_router

# Generating response from previous context
prompt = ChatPromptTemplate.from_template(
    "Answer {query} based on the following texts: {context}"
)
summary_chain = prompt | HAIKU_3_5_LLM | StrOutputParser()

# Aggregation pipeline/ Filter/Projection constructor
template = hub.pull("eden19/entire_db_retrieval")
sonnet_model = SONNET_3_7_LLM.bind_tools(tools)
sonnet_agent = template | sonnet_model


# Tool summarizer model
class ToolSummarizer(TypedDict):
    """Check if tool output answers user query"""

    query_tool_match: Annotated[
        Literal["yes", "no"],
        ...,
        (
            "Given a user's query and information retrieved from an external "
            "database determine whether the query can be answered with the "
            "information. If true answer 'yes', else, 'no'."
        ),
    ]


structured_tool_summarizer = HAIKU_3_5_LLM.with_structured_output(
    ToolSummarizer
)
prompt = hub.pull("eden19/tool_summarizer")
tool_summarizer_agent = prompt | structured_tool_summarizer

# DB retrieval summary model
prompt = hub.pull("eden19/mongodb_summary")
mongodb_summary_agent = prompt | HAIKU_3_5_LLM | StrOutputParser()


# Filter generator model
class FilterGenerator(TypedDict):
    """MongoDB filter to be applied before vector retrieval"""

    filter_query: Annotated[dict, ..., "MongoDB match filter"]
    top_k: int = Annotated[dict, ..., "Number of documents"]


filter_prompt = hub.pull("eden19/filtergeneration")
filter_generator_llm = SONNET_3_7_LLM.with_structured_output(FilterGenerator)
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


retrieval_grader = HAIKU_3_5_LLM.with_structured_output(RetrievalGrader)
retrieval_grade_prompt = hub.pull("eden19/retrievalgrader")
doc_grader = retrieval_grade_prompt | retrieval_grader

# Generating response to documents retrieved from the vector index
answer_generation_prompt = hub.pull("eden19/answergeneration")
rag_chain = answer_generation_prompt | SONNET_3_7_LLM | StrOutputParser()

# Schema generation
schema_prompt = hub.pull("eden19/data-schema-summary")
schema_chain = schema_prompt | SONNET_3_5_LLM | StrOutputParser()
