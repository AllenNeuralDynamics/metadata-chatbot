# Import the Streamlit library
import streamlit as st
import asyncio
import uuid

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metadata_chatbot.agents.GAMER import GAMER
from langchain_core.messages import HumanMessage, AIMessage

#run on terminal with streamlit run c:/Users/sreya.kumar/Documents/GitHub/metadata-chatbot/app.py [ARGUMENTS]

unique_id =  str(uuid.uuid4())

async def main():
    st.title("GAMER: Generative Analysis of Metadata Retrieval")

    llm = GAMER()

    message = st.chat_message("assistant")
    message.write("Hello! How can I help you?")

    prompt = st.chat_input("Ask a question about the AIND Metadata!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)

    if prompt is not None and prompt != '':
        st.session_state.messages.append(HumanMessage(prompt))

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response =  await llm.streamlit_astream(prompt, unique_id = unique_id)
            st.markdown(response)
            
        st.session_state.messages.append(AIMessage(response))


if __name__ == "__main__":
    asyncio.run(main())