# metadata-chatbot

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-57.3%25-red)
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

## Data Retrieval

### Vector Embeddings

To improve retrieval accuracy and decrease hallucinations, we use vector embeddings to access relevant chunks of information found across the database. This process starts with accessing assets, and chunking each json file to chunks of around 8000 tokens (10 chunks per file)-- each chunk preserves the hierarchy found in json files. These chunks are converted to vector arrays of size 1024, through an embedding model (Amazon's Titan 2.0 Embedding). The user's query is converted to a vector and projected onto the latent space. The chunks that contain the most relevant information will be accessed through a cosine similarity search.

### AIND-data-schema-access REST API

For queries that require accessing the entire database, like count based questions, information is accessed through an aggregation pipeline, provided by one of the constructed LLM agents, and the API connection.

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```bash
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```bash
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):

```bash
flake8 .
```

- Use **black** to automatically format the code into PEP standards:

```bash
black .
```

- Use **isort** to automatically sort import statements:

```bash
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:

```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Semantic Release

The table below, from [semantic release](https://github.com/semantic-release/semantic-release), shows which commit message gets you which release type when `semantic-release` runs (using the default configuration):

| Commit message                                                                                                                                                                                     | Release type                                                                                                       |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `fix(pencil): stop graphite breaking when too much pressure applied`                                                                                                                             | ~~Patch~~ Fix Release, Default release                                                                            |
| `feat(pencil): add 'graphiteWidth' option`                                                                                                                                                       | ~~Minor~~ Feature Release                                                                                         |
| `perf(pencil): remove graphiteWidth option<br>``<br>BREAKING CHANGE: The graphiteWidth option has been removed.``<br>The default graphite width of 10mm is always used for performance reasons.` | ~~Major~~ Breaking Release <br /> (Note that the `BREAKING CHANGE: ` token must be in the footer of the commit) |

### Documentation

To generate the rst files source files for documentation, run

```bash
sphinx-apidoc -o doc_template/source/ src 
```

Then to create the documentation HTML files, run

```bash
sphinx-build -b html doc_template/source/ doc_template/build/html
```

More info on sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).
