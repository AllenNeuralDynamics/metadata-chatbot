# Import the Streamlit library
import streamlit as st
import asyncio

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import metadata_chatbot.agents.docdb_retriever
import metadata_chatbot.agents.agentic_graph
from metadata_chatbot.agents.async_workflow import astream


#run on terminal with streamlit run <FILE PATH> [ARGUMENTS]

async def main():
# Write a simple message to the app's webpage
    llm = GAMER()
    message = st.chat_message("assistant")
    message.write("Hello!")

    prompt = st.chat_input("Ask a question about the AIND Metadata!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt:
    # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = await llm.ainvoke(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    asyncio.run(main())