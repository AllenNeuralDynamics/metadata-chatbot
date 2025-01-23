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
from langsmith import Client
from langchain_core.tracers.context import collect_runs



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

    client = Client()

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
            with collect_runs() as cb:
                with st.status("Generating answer...", expanded = True) as status:
                    async for result in answer_generation(query, chat_history, config, model):
                        if prev != None:
                            st.markdown(prev)
                        prev = result
                        generation = prev
                    status.update(label = "Answer generation successful.")
                st.session_state.run_id = cb.traced_runs[0].id
                
    if st.session_state.get("run_id"):
        run_id = st.session_state.run_id
        feedback = streamlit_feedback(
            feedback_type="faces",
            optional_text_label="[Optional] Please provide an explanation",
            key=f"feedback_{run_id}",
        )

        # Define score mappings for both "thumbs" and "faces" feedback systems
        score_mappings = {
            "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
        }

        # Get the score mapping based on the selected feedback option
        scores = score_mappings["faces"]

        if feedback:
            # Get the score from the selected feedback option's score mapping
            score = scores.get(feedback["score"])

            if score is not None:
                # Formulate feedback type string incorporating the feedback option
                # and score value
                feedback_type_str = f"faces: {feedback['score']}"

                # Record the feedback with the formulated feedback type string
                # and optional comment
                feedback_record = client.create_feedback(
                    run_id,
                    feedback_type_str,
                    score=score,
                    comment=feedback.get("text"),
                )
                st.session_state.feedback = {
                    "feedback_id": str(feedback_record.id),
                    "score": score,
                }
            else:
                st.warning("Invalid feedback score.")

            # final_response = st.empty()
            # final_response.markdown(generation)
        
            # feedback = streamlit_feedback(feedback_type="thumbs",
            #                               optional_text_label="[Optional] Please provide an explanation for your choice",)
            # print(feedback)
        st.session_state.messages.append(AIMessage(generation))
            # response =  await llm.streamlit_astream(query, unique_id = unique_id)
            # st.markdown(response)
    st.session_state.query = ''    


if __name__ == "__main__":
    asyncio.run(main())