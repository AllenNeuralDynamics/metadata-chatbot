import asyncio
from typing import List, Optional, Annotated
from typing_extensions import TypedDict
from pprint import pprint
import uuid

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import AIMessage, HumanMessage

from metadata_chatbot.agents.docdb_retriever import DocDBRetriever
from metadata_chatbot.agents.react_agent import react_agent
from metadata_chatbot.agents.agentic_graph import datasource_router,  filter_generation_chain, doc_grader, rag_chain

# from docdb_retriever import DocDBRetriever
# from react_agent import react_agent
# from agentic_graph import datasource_router,  filter_generation_chain, doc_grader, rag_chain, db_rag_chain

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

import warnings
warnings.filterwarnings('ignore')


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        query: question asked by user
        generation: LLM generation
        documents: list of documents
    """
    messages: Annotated[list[AnyMessage], add_messages]
    generation: Optional[str]
    documents: Optional[List[str]]
    filter: Optional[dict]
    top_k: Optional[int] 

async def route_question_async(state):
    """
    Route question to database or vectorstore
    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    query = state['messages'][0].content

    source = await datasource_router.ainvoke({"query": query})

    if source['datasource'] == "direct_database":
        return "direct_database"
    elif source['datasource'] == "vectorstore":
        return "vectorstore"
    
def print_stream(stream):
    message_list = []
    for s in stream:
        message_list.append(s)
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

    for message in message_list[-1]['messages']:
        if isinstance(message, AIMessage):
            final_answer = message.content
    return final_answer
    
async def retrieve_DB_async(state):
    """
    Retrieves from data asset collection in prod DB after constructing a MongoDB query
    
    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key may be added to state, generation, which contains the answer for query asked
    """

    query = state['messages'][0].content
    inputs = {"messages": [("user", query)]}

    try:
        message_list = []
        async for s in react_agent.astream(inputs, stream_mode="values"):
            message = s["messages"][-1]
            state['messages'] = state.get('messages', []) + [message]
            message_list.append(message.content)

        return {"messages": state.get("messages", []),
                "generation": state['messages'][-1].content}
        
        #print(message_list)

        #generation = print_stream(react_agent.stream(inputs, stream_mode="values"))
        answer = ''
    except:
        answer = "An error has occured with the retrieval from DocDB, try structuring your query another way!"

        return {"messages": [
                    AIMessage(answer)
                ],
                "generation": answer, 
                "filter": None, 
                "documents": None,
                "top_k": None}

async def filter_generator_async(state):
    """
    Filter database

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key may be added to state, filter, which contains the MongoDB query that will be applied before retrieval
    """
    query = state['messages'][0].content

    try:
        result = await filter_generation_chain.ainvoke({"query": query})
        filter = result['filter_query']
        top_k = result['top_k']
    except:
        filter = None
        top_k = None
        
    return {"filter": filter, 
            "top_k": top_k, 
            "messages": [
                AIMessage(str(result))
            ],
            "documents": None,
            "generation": None}
 
async def retrieve_VI_async(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    query = state['messages'][0].content
    filter = state["filter"]
    top_k = state["top_k"]

    try:
        retriever = DocDBRetriever(k = top_k)


        #print("Retrieving relevant documents from vector index...")
        documents = await retriever.aget_relevant_documents(query = query, query_filter = filter)
            
    except:
        documents = "No documents were returned"

    return {"documents": documents, 
            "filter": state.get("filter", None),
            "top_k": state.get("top_k", None),
            "messages": [AIMessage("Retrieving relevant documents from vector index...")],
            "generation" : None}

async def grade_doc_async(query, doc: Document):
    score = await doc_grader.ainvoke({"query": query, "document": doc.page_content})
    grade = score['binary_score']

    try:
        if grade == "yes":
            return doc.page_content
        else:
            return None
    except:
        return "There was an error processing this document."
        

async def grade_documents_async(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    query = state['messages'][0].content
    documents = state["documents"]

    #print("Checking document relevancy to your query...")

    filtered_docs = await asyncio.gather(
        *[grade_doc_async(query, doc) for doc in documents],
        return_exceptions = True)
    filtered_docs = [doc for doc in filtered_docs if doc is not None]
    return {"documents": filtered_docs, 
            "top_k": state.get("top_k", None), 
            "filter": state.get("filter", None),
            "messages": [AIMessage("Checking document relevancy to your query...")],
            "generation" : None}

async def generate_VI_async(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    query = state['messages'][0].content
    documents = state["documents"]

    try:
        generation = await rag_chain.ainvoke({"documents": documents, "query": query})
    except:
        generation = "Apologies, would you mind reframing the query in another way?"

    return {"documents": documents, 
            "messages": [AIMessage(str(generation))],
            "generation": generation, 
            "top_k": state.get("top_k", None), 
            "filter": state.get("filter", None)}

async_workflow = StateGraph(GraphState) 
async_workflow.add_node("database_query", retrieve_DB_async)  
async_workflow.add_node("filter_generation", filter_generator_async)  
async_workflow.add_node("retrieve", retrieve_VI_async)  
async_workflow.add_node("document_grading", grade_documents_async)  
#async_workflow.add_node("generate_db", generate_DB_async)  
async_workflow.add_node("generate_vi", generate_VI_async)  

async_workflow.add_conditional_edges(
    START,
    route_question_async,
    {
        "direct_database": "database_query",
        "vectorstore": "filter_generation",
    },
)
async_workflow.add_edge("database_query", END) 
#async_workflow.add_edge("generate_db", END)
async_workflow.add_edge("filter_generation", "retrieve")
async_workflow.add_edge("retrieve", "document_grading")
async_workflow.add_edge("document_grading","generate_vi")
async_workflow.add_edge("generate_vi", END)

memory = MemorySaver()
async_app = async_workflow.compile(checkpointer=memory)
unique_id =  str(uuid.uuid4())
config = {"configurable":{"thread_id": unique_id}}

outputs = []

async def main():
    query = "Give me a list of sessions for subject 740955?"
    #query = "What is the mongod query to find How many records are in the database?"
    inputs = {
        "messages": [HumanMessage(query)], 
    }

    async for output in async_app.astream(inputs, config):
        for key, value in output.items():
            if "generation" in value:  # Check if 'generation' exists in the value
                yield value["generation"]

async def collect_main():
    result = []
    async for item in main():  # main is an async generator
        result.append(item)  # or do something with each item
    return result[-1]
    



    # result = await async_app.ainvoke(inputs)
#Run the async function
print(asyncio.run(collect_main()))
