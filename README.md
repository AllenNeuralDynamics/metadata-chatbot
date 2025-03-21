# GAMER: Generative Analysis for Metadata Retrieval

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-94.1%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-0%25-red?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.11-blue?logo=python)

## Installation

Install a virtual environment with python 3.11 (install a version of python 3.11 that's compatible with your operating system).

```bash
py -3.11 -m venv .venv
```

On Windows, activate the environment with

```bash
.venv\Scripts\Activate.ps1
```

You will need access to the AWS Bedrock service in order to access the model. Once you've configured the AWS CLI, and granted access to Anthropic's Claude Sonnet 3 and 3.5, proceed to the following steps.

Install the chatbot package -- ensure virtual environment is running.

```bash
pip install metadata-chatbot
```

## Usage

To stream results from the model,

```bash
from langchain_core.messages import HumanMessage
import asyncio

query = "What was the refractive index of the chamber immersion medium used in this experiment SmartSPIM_675387_2023-05-23_23-05-56"

async def new_astream(query):

    inputs = {"messages": [HumanMessage(query)]}

    config = {}

    async for result in stream_response(inputs,config,app):
        print(result)  # Process the yielded results

asyncio.run(new_astream(query))
```

## Relevant repositories
[Vector embeddings generation script for metadata assets](https://github.com/AllenNeuralDynamics/aind-metadata-embeddings)
[Vector embeddings generation script for AIND data schema repository](https://github.com/AllenNeuralDynamics/aind_data_schema_embeddings)
[Streamlit app respository](https://github.com/sreyakumar/aind-GAMER-app)

## High Level Overview

The project's main goal is to developing a chat bot that is able to ingest, analyze and query metadata. Metadata is accumulated in lieu with experiments and consists of information about the data description, subject, equipment and session. To maintain reproducibility standards, it is important for metadata to be documented well. GAMER is designed to streamline the querying process for neuroscientists and other users.

## Model Overview

The current chat bot model uses Anthropic's Claude Sonnet 3 and 3.5, hosted on AWS' Bedrock service. Since the primary goal is to use natural language to query the database, the user will provide queries about the metadata specifically. The framework is hosted on Langchain. Claude's system prompt has been configured to understand the metadata schema format and craft MongoDB queries based on the prompt. Given a natural language query about the metadata, the model will produce a MongoDB query, thought reasoning and answer. This method of answering follows chain of thought reasoning, where a complex task is broken up into manageable chunks, allowing logical thinking through of a problem.

The main framework used by the model is Retrieval Augmented Generation (RAG), a process in which the model consults an external database to generate information for the user's query. This process doesn't interfere with the model's training process, but rather allows the model to successfully query unseen data with few shot learning (examples of queries and answers) and tools (e.g. API access) to examine these databases.

### Multi-Agent graph framework

A multi-agent workflow is created using Langgraph, allowing for parallel execution of tasks, like document retrieval from the vector index, and increased developer control over the the RAG process.

![Worfklow](GAMER_workflow.jpeg)

This model uses a multi agent framework on Langraph to retrieve and summarize metadata information based on a user's natural language query. This workflow consists of 6 agents, or nodes, where a decision is made and there is new context provided to either the model or the user. Here are some decisions incorporated into the framework:
1. To best answer the query, which data source should the model refer to?
    - Input: `x (query)`
    - Decides best data to query against
    - Output: `entire_database, vector_embeddings, claude, data_schema`
2. If querying against the vector embeddings, does the index need to be filtered further with metdata tags, to improve optimization of retrieval?
    - Input: `x (query)`
    - Decides whether database can be further filtered by applying a MongoDB query
    - Output: `MongoDB query, None`
3. Are the documents retrieved during retrieval relevant to the question?
    - Input: `x (query), y (documents)`
    - Decides whether document should be kept or tossed during summarization
    - Output: `yes, no`
4. Is the tool output retrieved through tool calling relevant to the question?
    - Input: `x (query), y (tool output)`
    - Decides whether MongoDB queries need to be reconstructed to retrieve more relevant output
    - Output: `yes, no`
5. Does the conversation need to be summarized?
    - Input: `x (message list)`
    - If the conversation list exceeds 6 messages, the chat history will be summarized, and earlier messages will be deleted
    - Output: `yes, no`

## Data Retrieval

### Vector Embeddings

To improve retrieval accuracy and decrease hallucinations, we use vector embeddings to access relevant chunks of information found across the database. This process starts with accessing assets, and chunking each json file to chunks of around 8000 tokens (10 chunks per file)-- each chunk preserves the hierarchy found in json files. These chunks are converted to vector arrays of size 1024, through an embedding model (Amazon's Titan 2.0 Embedding). The user's query is converted to a vector and projected onto the latent space. The chunks that contain the most relevant information will be accessed through a cosine similarity search.

### AIND-data-schema-access REST API

For queries that require accessing the entire database, like count based questions, information is accessed through an aggregation pipeline, provided by one of the constructed LLM agents, and the API connection.

## Current specifications

* The model can query the fields for a specified asset.
* The model can query metadata documents from the document database.
* The model is able to return a list of unique values for a given field.
* The model is able to answer count based questions.
