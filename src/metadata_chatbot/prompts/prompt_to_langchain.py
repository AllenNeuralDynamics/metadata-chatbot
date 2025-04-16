"""Script to convert prompts to LC templates"""

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Datasource router
router_prompt = Path(
    "src/metadata_chatbot/prompts/query_router.txt"
).read_text()
router_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=[
                {
                    "text": router_prompt,
                    "type": "text",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        ),
        ("human", "{query}"),
    ]
)

# Aggregation pipeline/ Filter/Projection constructor
mongodb_prompt = Path(
    "src/metadata_chatbot/prompts/entire_db_retrieval.txt"
).read_text()
mongodb_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=[
                {
                    "text": mongodb_prompt,
                    "type": "text",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        ),
        ("human", "{query}"),
    ]
)

# Shortened aggregation pipeline/ Filter/Projection constructor
shortened_mongodb_prompt = Path(
    "src/metadata_chatbot/prompts/shortened_entire_db_retrieval.txt"
).read_text()
shortened_mongodb_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=[
                {
                    "text": shortened_mongodb_prompt,
                    "type": "text",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        ),
        ("human", "{query}"),
    ]
)


# Tool summarizer model
tool_summary_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=[
                {
                    "text": (
                        "You are a neuroscientist whos main goal is to grade whether "
                        "retrieved documents from a scientific database answers the user's query."
                    ),
                    "type": "text",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        ),
        (
            "human",
            "Check if the retrieved tool output: {tool_output} matches the query: {query}",
        ),
    ]
)

# DB retrieval summary model
mongodb_summary_prompt = Path(
    "src/metadata_chatbot/prompts/mongodb_summary.txt"
).read_text()
mongodb_summary_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=[
                {
                    "text": mongodb_summary_prompt,
                    "type": "text",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        ),
        (
            "human",
            (
                "This is the query: {query}, MongoDB tool call used: {tool_call} "
                "and retrieved documents as a result of the tool call: "
                "{documents}"
            ),
        ),
    ]
)


# Filter generator model
filter_gen_prompt = Path(
    "src/metadata_chatbot/prompts/filtergeneration.txt"
).read_text()
filter_gen_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=[
                {
                    "text": filter_gen_prompt,
                    "type": "text",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        ),
        (
            "human",
            (
                "This is the query: {query}."
                "Use the chat history: {chat_history} for more context"
            ),
        ),
    ]
)


# Check if retrieved documents answer question
retrieval_grader_prompt = Path(
    "src/metadata_chatbot/prompts/retrievalgrader.txt"
).read_text()
retrieval_grader_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=[
                {
                    "text": retrieval_grader_prompt,
                    "type": "text",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        ),
        (
            "human",
            (
                "This is the query: {query}."
                "This is the document: {document}."
            ),
        ),
    ]
)

# Generating response to documents retrieved from the vector index
answer_generation_prompt = Path(
    "src/metadata_chatbot/prompts/answergeneration.txt"
).read_text()
answer_generation_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=[
                {
                    "text": answer_generation_prompt,
                    "type": "text",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        ),
        (
            "human",
            (
                "This is the query: {query}."
                "Answer the query based on the following context: {documents}."
            ),
        ),
    ]
)

# Schema generation
schema_generation_prompt = Path(
    "src/metadata_chatbot/prompts/answergeneration.txt"
).read_text()
schema_generation_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=[
                {
                    "text": answer_generation_prompt,
                    "type": "text",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        ),
        (
            "human",
            (
                "This is the query: {query}."
                "Answer the query based on the following context: {context}."
            ),
        ),
    ]
)
