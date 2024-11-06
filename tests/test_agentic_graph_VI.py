import pytest
from langsmith import unit
from metadata_chatbot.agents.agentic_graph import datasource_router, query_grader, filter_generation_chain, doc_grader, rag_chain

@pytest.fixture
def user_query():
    return "What are the injections for SmartSPIM_675387_2023-05-23_23-05-56?"


@unit
def test_datasource_router_vector_index():
    datasource = datasource_router.invoke({"query": user_query}).datasource
    assert datasource == "vectorstore"


