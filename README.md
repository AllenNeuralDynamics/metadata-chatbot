[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-49.4%25-red)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.11-blue?logo=python)

## Usage

## Installation

Install a virtual environment with python 3.11 (install a python 3.11 that's compatible with your operating system). Check if download was successful by runninng

```bash
py -3.11 -m venv .venv
```

On Windows, activate the environment with

```bash
.venv\Scripts\Activate.ps1
```

Install the chatbot package.

```bash
pip install -e .
```

To develop the code, run

```bash
pip install -e .[dev]
```

Or simply,

```bash
pip install metadata-chatbot
```

## High Level Overview

The project's main goal is to developing a chat bot that is able to ingest, analyze and query metadata. Metadata is accumulated in lieu with experiments and consists of information about the data description, subject, equipment and session. To maintain reproducibility standards, it is important for metadata to be documented well.

## Model Overview

The current chat bot model uses Anthropic's Claude Sonnet 3 hosted on AWS' Bedrock service. Since the primary goal is to use natural language to query the database, the user will provide prompts about the metadata specifically. The framework is hosted on Langchain. Claude's system prompt has been configured to understand the metadata schema format and craft MongoDB queries based on the prompt. Given a natural language query about the metadata, the model will produce a MongoDB query, thought reasoning and answer. This method of answering follows chain of thought reasoning, where a complex task is broken up into manageable chunks, allowing logical thinking through of a problem. 

The main framework used by the model is Retrieval Augmented Generation, a process in which the model consults an external database to generate information for the user's query. This process doesn't interfere with the model's training process, but rather allows the model to successfully query unseen data with few shot learning (examples of queries and answers) and tools (e.g. API access) to examine these databases.

## Data Retrieval

### Vector Embeddings

To improve retrieval accuracy and decrease hallucinations, we use vector embeddings to access relevant chunks of information found across the database. This process starts with accessing assets, and chunking each json file to chunks of around 8000 tokens (10 chunks per file)-- each chunk preserves the hierarchy found in json files. These chunks are converted to vector arrays of size 1024, through an embedding model (Amazon's Titan 2.0 Embedding). The user's query is converted to a vector and projected onto the latent space. The chunks that contain the most relevant information will be accessed through a cosine similarity search.

### AIND-data-schema-access REST API

For queries that require accessing the entire database, like count based questions, information is accessed through an aggregation pipeline, provided by one of the constructed LLM agents, and the API connection.

## Multi-Agent graph framework
A multi-agent workflow is created using Langgraph, allowing for parallel execution of tasks, like document retrieval from the vector index, and control over the the RAG process.

![Worfklow](multi-agent-workflow-11-01.jpeg)

