# Import the Streamlit library
import streamlit as st
import sys
import os

import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from metadata_chatbot.agents.GAMER import GAMER
import uuid

#run on terminal with streamlit run c:/Users/sreya.kumar/Documents/GitHub/metadata-chatbot/app.py [ARGUMENTS]


async def main():

    llm = GAMER()
    unique_id =  str(uuid.uuid4())
    
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
        #response = await llm.ainvoke(prompt)

        with st.chat_message("assistant"):
            response =  await llm.streamlit_astream(prompt, unique_id = unique_id)
            st.markdown(response)
            
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    asyncio.run(main())