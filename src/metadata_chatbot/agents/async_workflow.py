import asyncio
from typing import List, Optional, Annotated
from typing_extensions import TypedDict
from pprint import pprint
import uuid

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import AIMessage, HumanMessage

from metadata_chatbot.agents.docdb_retriever import DocDBRetriever
from metadata_chatbot.agents.react_agent import astream_input
from metadata_chatbot.agents.agentic_graph import datasource_router, filter_generation_chain, doc_grader, rag_chain, prev_context_chain, query_rewriter_chain

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
    query: Optional[str]
    generation: str
    documents: Optional[List[str]]
    filter: Optional[dict]
    top_k: Optional[int] 

async def route_question_async(state: dict) -> dict:
    """
    Route question to database or vectorstore
    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    query = state['messages'][-1].content
    chat_history = state['messages']

    source = await datasource_router.ainvoke({"query": query, "chat_history": chat_history})

    if source['datasource'] == "direct_database":
        return "direct_database"
    elif source['datasource'] == "vectorstore":
        return "vectorstore"
    elif source['datasource'] == "claude":
        return "claude"
    
    
async def retrieve_DB_async(state: dict) -> dict:
    """
    Retrieves from data asset collection in prod DB after constructing a MongoDB query
    """


    query = state['messages'][-1].content
    chat_history = state['messages']
    message_iterator = []
    # try:

    #     async for result in astream_input(query):
    #         response = result['type']
    #         if response == 'intermediate_steps':
    #             message_iterator.append(result['content'])
    #         if response == 'agg_pipeline':
    #             message_iterator.append(f"The MongoDB pipeline used to on the database is: {result['content']}")
    #         if response == 'tool_response':
    #             message_iterator.append(f"Retrieved output from MongoDB: {result['content']}")
    #         if response == 'final_answer':
    #             answer = result['content']
    # except Exception as e:
    #     answer = f"An error occured: {e}"


    return {"messages": message_iterator,
            "generation": ''
            }

async def filter_generator_async(state: dict) -> dict:
    """
    Filter database by constructing basic MongoDB match filter and determining number of documents to retrieve

    """
    query = state['messages'][-1].content
    chat_history = state['messages']

    try:
        result = await filter_generation_chain.ainvoke({"query": query, "chat_history": chat_history})
        filter = result['filter_query']
        top_k = result['top_k']
    except:
        filter = None
        top_k = None
        
    return {"query": query,
            "filter": filter, 
            "top_k": top_k, 
            "messages": [AIMessage(f"Using MongoDB filter: {filter} on the database and retrieving {top_k} documents")]
            }
 
async def retrieve_VI_async(state: dict) -> dict:
    """
    Retrieve documents
    """
    query = state['query']
    filter = state["filter"]
    top_k = state["top_k"]

    try:
        retriever = DocDBRetriever(k = top_k)
        documents = await retriever.aget_relevant_documents(query = query, query_filter = filter)
            
    except:
        documents = "No documents were returned"

    return {"documents": documents, 
            "messages": [AIMessage("Retrieving relevant documents from vector index...")]
            }

async def grade_doc_async(query: str, doc: Document):
    score = await doc_grader.ainvoke({"query": query, "document": doc.page_content})
    grade = score['binary_score']

    try:
        if grade == "yes":
            return doc.page_content
        else:
            return None
    except:
        return "There was an error processing this document."
        

async def grade_documents_async(state: dict) -> dict:
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    query = state['query']
    documents = state["documents"]

    filtered_docs = await asyncio.gather(
        *[grade_doc_async(query, doc) for doc in documents],
        return_exceptions = True)
    filtered_docs = [doc for doc in filtered_docs if doc is not None]

    return {"documents": filtered_docs, 
            "messages": [AIMessage("Checking document relevancy to your query...")]
            }

async def generate_VI_async(state: dict) -> dict:
    """
    Generate answer
    """
    query = state['query']
    documents = state["documents"]

    try:
        generation = await rag_chain.ainvoke({"documents": documents, "query": query})
    except:
        generation = "Apologies, would you mind reframing the query in another way?"

    return {"messages": [AIMessage(str(generation))],
            "generation": generation, 
            }

async def generate_claude_async(state: dict) -> dict:
    """
    Generate answer
    """
    query = state['messages'][-1].content
    chat_history = state['messages']

    try:

        generation = await prev_context_chain.ainvoke({"query": query, "chat_history": chat_history})
    except:
        generation = "Apologies, the chat history doesn't answer your question."

    return {"messages": [AIMessage(str(generation))],
            "generation": generation, 
            }

async_workflow = StateGraph(GraphState) 
async_workflow.add_node("database_query", retrieve_DB_async)  
async_workflow.add_node("filter_generation", filter_generator_async)  
async_workflow.add_node("retrieve", retrieve_VI_async)  
async_workflow.add_node("document_grading", grade_documents_async)  
async_workflow.add_node("generate_vi", generate_VI_async)  
async_workflow.add_node("generate_claude", generate_claude_async)  

async_workflow.add_conditional_edges(
    START,
    route_question_async,
    {
        "direct_database": "database_query",
        "vectorstore": "filter_generation",
        "claude" : "generate_claude"
    },
)
async_workflow.add_edge("generate_claude", END) 
async_workflow.add_edge("database_query", END) 
async_workflow.add_edge("filter_generation", "retrieve")
async_workflow.add_edge("retrieve", "document_grading")
async_workflow.add_edge("document_grading","generate_vi")
async_workflow.add_edge("generate_vi", END)

memory = MemorySaver()
async_app = async_workflow.compile(checkpointer=memory)

query = "What are the unique modalities in the database??"

#query = "Give me a list of sessions for subject 740955?"

    
# async def new_astream(query):
#     async def main(query):
    
#         unique_id =  str(uuid.uuid4())
#         config = {"configurable":{"thread_id": unique_id}}
#         inputs = {
#             "messages": [HumanMessage(query)], 
#         }
#         async for output in async_app.astream(inputs, config):
#             for key, value in output.items():
#                 if key != "database_query":
#                     yield value['messages'][0].content 
#                 else:
#                     for message in value['messages']:
#                         yield message
#                     yield value['generation']
    
#     async for result in main(query):
#         print(result) # Process the yielded results

# #Run the main coroutine with asyncio
# asyncio.run(new_astream(query))

