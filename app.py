# Import the Streamlit library
import streamlit as st
import asyncio
import uuid

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metadata_chatbot.agents.async_workflow import async_workflow
from metadata_chatbot.agents.react_agent import astream_input

from langchain_core.messages import HumanMessage, AIMessage

import warnings
warnings.filterwarnings('ignore')

#run on terminal with streamlit run c:/Users/sreya.kumar/Documents/GitHub/metadata-chatbot/app.py [ARGUMENTS]

unique_id =  str(uuid.uuid4())

async def main():
    st.title("GAMER: Generative Analysis of Metadata Retrieval")

    message = st.chat_message("assistant")
    message.write("Hello! How can I help you?")

    query = st.chat_input("Ask a question about the AIND Metadata!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    model = async_workflow.compile()

    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)

    if query is not None and query != '':
        st.session_state.messages.append(HumanMessage(query))

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            async def main(query: str):
                chat_history = st.session_state.messages
                #config = {"configurable":{"thread_id": st.session_state.unique_id}}
                inputs = {
                    "messages": chat_history, 
                }
                async for output in model.astream(inputs):
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
                                        yield f"The MongoDB pipeline used to on the database is: {result['content']}"
                                    if response == 'tool_response':
                                        yield f"Retrieved output from MongoDB: {result['content']}"
                                    if response == 'final_answer':
                                        yield result['content']

                            except Exception as e:
                                yield f"An error has occured with the retrieval from DocDB: {e}. Try structuring your query another way."
                            # for response in value['messages']:
                            #     yield response.content
                            # yield value['generation']

            prev = None
            generation = None
            async for result in main(query):
                if prev != None:
                    st.markdown(prev)
                prev = result
                generation = prev
            st.markdown(generation)
        st.session_state.messages.append(AIMessage(generation))
            # response =  await llm.streamlit_astream(query, unique_id = unique_id)
            # st.markdown(response)
            


if __name__ == "__main__":
    asyncio.run(main())