import streamlit as st
import asyncio
import uuid

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metadata_chatbot.agents.async_workflow import async_workflow
from metadata_chatbot.agents.react_agent import astream_input

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from streamlit_feedback import streamlit_feedback

import uuid
import warnings
warnings.filterwarnings('ignore')

@st.cache_resource
def load_checkpointer():
    return MemorySaver()

async def answer_generation(query: str, chat_history: list, config:dict, model):
    inputs = {
        "messages": chat_history, 
    }
    async for output in model.astream(inputs, config):
        for key, value in output.items():
            if key != "database_query":
                yield value['messages'][0].content 
            else:
                try:
                    query = str(chat_history) + query
                    async for result in astream_input(query = query):
                        response = result['type']
                        if response == 'intermediate_steps':
                            yield result['content']
                        if response == 'agg_pipeline':
                            yield "The MongoDB pipeline used to on the database is:" 
                            yield f"`{result['content']}`"
                        if response == 'tool_response':
                            yield "Retrieved output from MongoDB:" 
                            yield f"""```json
                                    {result['content']}
                                    ```"""
                        if response == 'final_answer':
                            yield result['content']
                except Exception as e:
                    yield f"An error has occured with the retrieval from DocDB: {e}. Try structuring your query another way."

def set_query(query):
    st.session_state.query = query

async def main():
    st.title("GAMER: Generative Analysis of Metadata Retrieval")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = ''
    st.session_state.thread_id = str(uuid.uuid4())
    
    checkpointer = load_checkpointer()
    model = async_workflow.compile(checkpointer=checkpointer)

    if 'query' not in st.session_state:
        st.session_state.query = ''

    st.info(
        "Ask a question about the AIND metadata! Please note that it will take a couple of seconds to generate an answer. Type a query to start or pick one of these suggestions:"
    )

    examples = [
        "What are the modalities that exist in the database? What are the least and most common ones?",
        "What is the MongoDB query to find the injections used in SmartSPIM_675387_2023-05-23_23-05-56",
        "Can you list all the procedures performed on 662616, including their start and end dates?"
    ]

    columns = st.columns(len(examples))
    for i, column in enumerate(columns):
        with column:
            st.button(examples[i], on_click = set_query, args=[examples[i]])


    message = st.chat_message("assistant")
    message.write("Hello! How can I help you?")

    user_query = st.chat_input("Message GAMER")

    if user_query:
        st.session_state.query = user_query 

    if "messages" not in st.session_state:
        st.session_state.messages = []


    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)

    query = st.session_state.query
    if query is not None and query != '':
        st.session_state.messages.append(HumanMessage(query))

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            prev = None
            generation = None
            chat_history = st.session_state.messages
            with st.status("Generating answer...", expanded = True) as status:
                async for result in answer_generation(query, chat_history, config, model):
                    if prev != None:
                        # if type(prev) == list:
                        #     st.markdown("[")
                        #     for i in prev:
                        #         st.markdown(f'`{i}`,')
                        #     st.markdown("]")
                        # else:
                        st.markdown(prev)
                    prev = result
                    generation = prev
                status.update(label = "Answer generation successful.")

            final_response = st.empty()
            final_response.markdown(generation)
        
            feedback = streamlit_feedback(feedback_type="thumbs",
                                          optional_text_label="[Optional] Please provide an explanation for your choice",)
            print(feedback)
        st.session_state.messages.append(AIMessage(generation))
            # response =  await llm.streamlit_astream(query, unique_id = unique_id)
            # st.markdown(response)
    st.session_state.query = ''    


if __name__ == "__main__":
    asyncio.run(main())