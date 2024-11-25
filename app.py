# Import the Streamlit library
import streamlit as st
from metadata_chatbot.agents.GAMER import GAMER
import asyncio

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