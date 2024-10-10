from pydantic import BaseModel, Field
from langchain_aws import ChatBedrock
from langchain import hub
import logging, json, sys, os, asyncio
from typing import List, Optional, Literal, Iterator
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from docdb_retriever import DocDBRetriever
from langchain_core.tools import tool
from aind_data_access_api.document_db_ssh import DocumentDbSSHClient, DocumentDbSSHCredentials
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.outputs import GenerationChunk
from langchain_core.documents import Document

from langgraph.graph import END, StateGraph, START

sys.path.append(os.path.abspath("C:/Users/sreya.kumar/Documents/GitHub/metadata-chatbot"))
from utils import LANGCHAIN_COLLECTION, ResourceManager

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
        ...,
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

chat_history = []

# result = db_surveyor.invoke(
#     {
#         "query": "what are the unique modalities in the database",
#         "chat_history": chat_history
#     }
# )

# generated_answer = result['output']['text']

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

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        query: question asked by user
        generation: LLM generation
        documents: list of documents
    """

    query: str
    generation: str
    documents: List[str]
    filter: Optional[dict]
    #k: Optional[int] = 10

def route_question(state):
    """
    Route question to database or vectorstore
    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    query = state["query"]

    source = datasource_router.invoke({"query": query})
    if source.datasource == "direct_database":
        logging.info("Entire database needs to be queried.")
        return "direct_database"
    elif source.datasource == "vectorstore":
        logging.info("Querying against vector embeddings...")
        return "vectorstore"
    
# async def route_question(state):
#     """
#     Route question to database or vectorstore
#     Args:
#         state (dict): The current graph state

#     Returns:
#         str: Next node to call
#     """
#     query = state["query"]

#     source = await datasource_router.ainvoke({"query": query})
#     if source.datasource == "direct_database":
#         logging.info("Entire database needs to be queried.")
#         return "direct_database"
#     elif source.datasource == "vectorstore":
#         logging.info("Querying against vector embeddings...")
#         return "vectorstore"

def generate_for_whole_db(state):
    """
    Filter database
    
    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key may be added to state, generation, which contains the answer for query asked
    """

    query = state["query"]
    chat_history = []

    logging.info("Generating answer...")

    generation = db_surveyor.invoke({'query': query, 'chat_history': chat_history})
    return {"query": query, "generation": generation}

# async def generate_for_whole_db(state):
#     """
#     Filter database
    
#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): New key may be added to state, generation, which contains the answer for query asked
#     """

#     query = state["query"]
#     chat_history = []

#     logging.info("Generating answer...")

#     generation = await db_surveyor.ainvoke({'query': query, 'chat_history': chat_history})
#     return {"query": query, "generation": generation}


def filter_generator(state):
    """
    Filter database

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key may be added to state, filter, which contains the MongoDB query that will be applied before retrieval
    """
    logging.info("Determining whether filter is required...")

    query = state["query"]

    query_grade = query_grader.invoke({"query": query}).binary_score
    logging.info(f"Database needs to be further filtered: {query_grade}")

    if query_grade == "yes":
        filter = filter_generation_chain.invoke({"query": query}).filter_query
        logging.info(f"Database will be filtered using: {filter}")
        return {"filter": filter, "query": query}
    else:
        return {"filter": None, "query": query}
    
# async def filter_generator(state):
#     """
#     Filter database

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): New key may be added to state, filter, which contains the MongoDB query that will be applied before retrieval
#     """
#     logging.info("Determining whether filter is required...")

#     query = state["query"]

#     result = await query_grader.ainvoke({"query": query})
#     query_grade = result.binary_score
#     logging.info(f"Database needs to be further filtered: {query_grade}")

#     if query_grade == "yes":
#         result = await filter_generation_chain.ainvoke({"query": query})
#         filter = result.filter_query
#         logging.info(f"Database will be filtered using: {filter}")
#         return {"filter": filter, "query": query}
#     else:
#         return {"filter": None, "query": query}


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    logging.info("Retrieving documents...")
    query = state["query"]
    filter = state["filter"]

    # Retrieval
    with ResourceManager() as RM:
        collection = RM.client['metadata_vector_index']['LANGCHAIN_ALL_curated_assets']
        retriever = DocDBRetriever(collection = collection, k = 10)
        documents = retriever.get_relevant_documents(query = query, query_filter = filter)
    return {"documents": documents, "query": query}

# async def retrieve(state):
#     """
#     Retrieve documents

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): New key added to state, documents, that contains retrieved documents
#     """
#     logging.info("Retrieving documents...")
#     query = state["query"]
#     filter = state["filter"]

#     # Retrieval
#     with ResourceManager() as RM:
#         collection = RM.client['metadata_vector_index']['LANGCHAIN_ALL_curated_assets']
#         retriever = DocDBRetriever(collection = collection, k = 10)
#         documents = await retriever.aget_relevant_documents(query = query, query_filter = filter)
#     return {"documents": documents, "query": query}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    logging.info("Checking relevance of documents to question asked...")
    query = state["query"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for doc in documents:
        score = doc_grader.invoke({"query": query, "document": doc.page_content})
        grade = score.binary_score
        logging.info(f"Retrieved document matched query: {grade}")
        if grade == "yes":
            logging.info("Document is relevant to the query")
            filtered_docs.append(doc)
        else:
            logging.info("Document is not relevant and will be removed")
            continue
    return {"documents": filtered_docs, "query": query}

# async def grade_doc(doc: Document):
#     score = await doc_grader.ainvoke({"query": query, "document": doc.page_content})
#     grade = score.binary_score
#     logging.info(f"Retrieved document matched query: {grade}")
#     if grade == "yes":
#         logging.info("Document is relevant to the query")
#         return doc
#     else:
#         logging.info("Document is not relevant and will be removed")
#         return None
        

# async def grade_documents(state):
#     """
#     Determines whether the retrieved documents are relevant to the question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Updates documents key with only filtered relevant documents
#     """

#     logging.info("Checking relevance of documents to question asked...")
#     query = state["query"]
#     documents = state["documents"]

#     filtered_docs = await asyncio.gather(*[grade_doc(doc) for doc in documents])
#     filtered_docs = [doc for doc in filtered_docs if doc is not None]
#     return {"documents": filtered_docs, "query": query}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    logging.info("Generating answer...")
    query = state["query"]
    documents = state["documents"]

    doc_text = "\n\n".join(doc.page_content for doc in documents)

    # RAG generation
    generation = rag_chain.invoke({"documents": doc_text, "query": query})
    return {"documents": documents, "query": query, "generation": generation, "filter": state.get("filter", None)}

# async def generate(state):
#     """
#     Generate answer

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): New key added to state, generation, that contains LLM generation
#     """
#     logging.info("Generating answer...")
#     query = state["query"]
#     documents = state["documents"]

#     doc_text = "\n\n".join(doc.page_content for doc in documents)

#     # RAG generation
#     generation = await rag_chain.ainvoke({"documents": doc_text, "query": query})
#     return {"documents": documents, "query": query, "generation": generation, "filter": state.get("filter", None)}

workflow = StateGraph(GraphState) 
workflow.add_node("database_query", generate_for_whole_db)  
workflow.add_node("filter_generation", filter_generator)  
workflow.add_node("retrieve", retrieve)  
workflow.add_node("document_grading", grade_documents)  
workflow.add_node("generate", generate)  

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "direct_database": "database_query",
        "vectorstore": "filter_generation",
    },
)
workflow.add_edge("filter_generation", "retrieve")
workflow.add_edge("retrieve", "document_grading")
workflow.add_edge("document_grading","generate")
workflow.add_edge("generate", END)

# query = "What were the injections performed on subject 608551"
# app = workflow.compile()
# inputs = {"query" : query}
# async def main():
#     await app.ainvoke(inputs)

# if __name__ == "__main__":
#     asyncio.run(main())
    
#     result = await model.acall(query)
#     print(result)

class GAMER:
    def __init__(self):
        # Initialize your LangGraph workflow components here
        self.app = workflow.compile()

    def __call__(self, query:str) -> str:
        # Process the input through your workflow
        inputs = {"query" : query}
        result = self.app.stream(inputs)
        for output in result:
            for key, value in output.items():
                logging.info(f"Currently on node '{key}':")
        return value
    
#     @property
#     def _llm_type(self) -> str:
#         """Get the type of language model used by this chat model. Used for logging purposes only."""
#         return "Claude 3 Sonnet"
    
    # async def acall(self, query:str) -> str:
    #     inputs = {"query" : query}
    #     result = self.app.astream(inputs)
    #     async for output in result:
    #         for key, value in output.items():
    #             logging.info(f"Currently on node '{key}':")
    #     return value['generation'] if value else None
    
query = "How old was the subject in SmartSPIM_675388_2023-05-24_04-10-19_stitched_2023-05-28_18-07-46"

model = GAMER()
result = model(query)
print(result['generation'])

# import asyncio

# async def main():
#     result = await model.acall(query)
#     print(result)

# asyncio.run(main())

