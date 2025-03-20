"""Streamlit app for GAMER"""

import asyncio
import json
import os
import uuid
import warnings

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.context import collect_runs
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client
from streamlit_feedback import streamlit_feedback

from metadata_chatbot.GAMER.workflow import stream_response, workflow

warnings.filterwarnings("ignore")

load_dotenv()


@st.cache_resource
def load_checkpointer():
    """Load langchain persistence"""
    return MemorySaver()


@st.cache_resource
def load_app():
    """Cache the compiled workflow model"""
    checkpointer = load_checkpointer()
    return workflow.compile(checkpointer=checkpointer)


@st.cache_resource
def get_langchain_client():
    """Cache the LangChain client"""
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
    return Client(api_url=langchain_endpoint, api_key=langchain_api_key)


@st.cache_resource
def get_langchain_tracer(project):
    """Cache the LangChain tracer for a specific project"""
    client = get_langchain_client()
    return LangChainTracer(project_name=project, client=client)


@st.cache_data
def get_example_questions():
    """Cache example questions"""
    return [
        (
            "What are the unique instrument ids for SmartSPIM experiments?"

        ),
        (
            "What is the MongoDB query to find the injections used in "
            "SmartSPIM_675387_2023-05-23_23-05-56"
        ),
        (
            "Can you list all the procedures performed on 662616, "
            "including their start and end dates?"
        ),
    ]


async def answer_generation(
    chat_history: list, config: dict, app, prev_generation
):
    """Streams GAMERS' node responses"""
    inputs = {
        "messages": chat_history,
    }

    try:
        async for result in stream_response(
            inputs, config, app, prev_generation
        ):
            yield result

    except Exception as e:
        yield (
            "An error has occured with the retrieval from DocDB: "
            f"{e}. Try structuring your query another way."
        )


def set_query(query):
    """Set query in session state, for buttons"""
    st.session_state.query = query


def initialize_session_state():
    """Initialize st session states"""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "run_id" not in st.session_state:
        st.session_state.run_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        st.session_state.model = load_app()
    if "generation" not in st.session_state:
        st.session_state.generation = None


async def typewriter_stream(result, container):
    """Enables streaming effect on st text content"""
    full_response = ""
    text_content = result["content"]

    if result["type"] == "tool_output":
        try:
            text_content = json.loads(text_content)
        except:
            text_content = json.loads(text_content[0])
    # stream = text_content

    if isinstance(text_content, str):
        for word in text_content.split():
            full_response += word + " "
            container.write(full_response + " ")
            await asyncio.sleep(0.05)
    container.write(text_content)


async def main():
    """Main script to launch Streamlit UI"""
    # st.title("GAMER: Generative Analysis of Metadata Retrieval")

    project = os.getenv("LANGSMITH_PROJECT")

    ls_tracer = get_langchain_tracer(project)
    run_collector = RunCollectorCallbackHandler()

    cfg = RunnableConfig()
    cfg["callbacks"] = [ls_tracer, run_collector]

    initialize_session_state()

    with st.sidebar:

        st.header("GAMER: Generative Analysis of Metadata Retrieval")
        "Please note that it will take a few seconds to generate an answer."

        with st.popover(
            "Configurations :material/settings:", use_container_width=True
        ):
            data_routes = st.selectbox(
                "Ask a question about the", options=("Metadata", "Data schema")
            )

            developer_mode = st.toggle("Developer mode")

        with st.popover(
            "Prompt engineering guide :memo:", use_container_width=True
        ):
            st.markdown(
                "If you are experiencing issues related to latency or robustness, "
                "it is likely that the model in the background is overwhelmed with "
                "the amount of information it has to retrieve or synthesize. "
                "Here are some prompt optimization tips you can try: "
            )
            st.markdown(
                "- Ensure that your query clearly labels the information "
                "you seek (e.g. writing out the full project name to prevent ambiguity). "
            )
            st.markdown(
                "- Explicitly specify a limit for the model to retrieve "
                "(e.g. limit the search to 10 documents). "
            )
            st.markdown(
                "- Break up complex queries. Ask queries one at a time, "
                "ideally starting with a simple, broad query and increasing complexity."
            )
            st.markdown(
                "- The model is relatively poor at fetching one random asset and applying "
                "the specified task to the asset. In this case, ask it to fetch a random "
                "data asset meeting a requirement (e.g. ophys experiment) and then ask it "
                "to apply the task."
            )
            st.markdown(
                "- The model does not know today's date. When asking temporal queries "
                "(i.e. Retrieve all the assets uploaded to the database in the past week) "
                "specify the date."
            )
            st.markdown(
                "Prompt it to return python code using the AIND data access api."
            )
            st.markdown(
                "If the chat history starts to become fuzzy, please refresh the tab. "
                "Note that GAMER will not retain previous contexts if this action is taken."
            )
            st.markdown(
                "Please leave feedback through the faces you see after a response is generated!"
            )

        (
            "[Model architecture repository]"
            "(https://github.com/AllenNeuralDynamics/metadata-chatbot)"
        )
        (
            "[Streamlit app repository]"
            "(https://github.com/sreyakumar/aind-GAMER-app)"
        )

    st.info("Type a query to start or pick one of these suggestions:")

    examples = get_example_questions()

    columns = st.columns(len(examples))
    for i, column in enumerate(columns):
        with column:
            st.button(examples[i], on_click=set_query, args=[examples[i]])

    message = st.chat_message("assistant")
    message.write("Hello! How can I help you?")

    user_query = st.chat_input("Ask a question about the AIND metadata!")

    if user_query:
        st.session_state.query = user_query

    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)

    query = st.session_state.query
    if query is not None and query != "":
        st.session_state.messages.append(HumanMessage(query))

        # with st.chat_message("user"):
        #     st.markdown(query)

        st.chat_message("user").write(query)

        with st.chat_message("assistant"):
            config = {
                "configurable": {"thread_id": st.session_state.thread_id}
            }
            # prev = None
            generation = None
            message_stream = []
            prev_generation = st.session_state.generation

            chat_history = st.session_state.messages
            with collect_runs() as cb:

                if developer_mode:
                    async for result in answer_generation(
                        chat_history,
                        config,
                        st.session_state.model,
                        prev_generation,
                    ):
                        with st.spinner("Generating answer..."):
                            try:
                                if result["type"] == "final_response":
                                    generation = result
                                else:
                                    temp_container = st.empty()
                                    await typewriter_stream(
                                        result, temp_container
                                    )
                                    message_stream.append(result)

                            except Exception as ex:
                                template = (
                                    "An exception of type {0} occurred. "
                                    "Arguments:\n{1!r}"
                                )
                                message = template.format(
                                    type(ex).__name__, ex.args
                                )
                                st.error(message)

                else:
                    try:
                        with st.status(
                            "Generating answer...", expanded=True
                        ) as status:
                            async for result in answer_generation(
                                chat_history,
                                config,
                                st.session_state.model,
                                prev_generation,
                            ):
                                if result["type"] == "final_response":
                                    generation = result
                                else:
                                    temp_container = st.empty()
                                    await typewriter_stream(
                                        result, temp_container
                                    )

                                    message_stream.append(result)

                            status.update(
                                label="Answer generation successful."
                            )

                    except Exception as ex:
                        template = (
                            "An exception of type {0} occurred. "
                            "Arguments:\n{1!r}"
                        )
                        message = template.format(type(ex).__name__, ex.args)
                        st.error(message)

                with st.spinner("Generating answer..."):
                    st.session_state.run_id = cb.traced_runs[-1].id
                    st.session_state.messages.append(
                        AIMessage(generation["content"])
                    )
                    st.session_state.generation = generation["content"]
                    final_response = st.empty()
                    await typewriter_stream(generation, final_response)
            # final_response.write(generation)

    if st.session_state.get("run_id"):
        run_id = st.session_state.run_id
        feedback = streamlit_feedback(
            feedback_type="faces",
            optional_text_label="[Optional] Please provide an explanation",
            key=f"feedback_{run_id}",
        )

        score_mappings = {
            "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
        }

        scores = score_mappings["faces"]

        if feedback:
            score = scores.get(feedback["score"])

            if score is not None:
                feedback_type_str = f"FACES: {feedback['score']}"

                client = get_langchain_client()

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

                st.toast("Feedback logged!", icon=":material/reviews:")

            else:
                st.warning("Invalid feedback score.")

    st.session_state.query = ""


if __name__ == "__main__":
    asyncio.run(main())
