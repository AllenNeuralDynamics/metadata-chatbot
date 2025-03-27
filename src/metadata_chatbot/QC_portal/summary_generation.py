"""REST API to generate summaries for given asset name"""

import uvicorn
from fastapi import FastAPI

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

from aind_data_access_api.document_db import MetadataDbClient

from metadata_chatbot.utils import SONNET_3_7_LLM, HAIKU_3_5_LLM

API_GATEWAY_HOST = "api.allenneuraldynamics.org"
DATABASE = "metadata_index"
COLLECTION = "data_assets"

docdb_api_client = MetadataDbClient(
    host=API_GATEWAY_HOST,
    database=DATABASE,
    collection=COLLECTION,
)

prompt = hub.pull("eden19/qc_portal_summary")
summary_generator = prompt | HAIKU_3_5_LLM | StrOutputParser()

app = FastAPI()


@app.get("/summary/{name}")
async def REST_summary(name: str):
    """Invoking GAMER to generate summary of asset"""
    filter = {"name": name}
    records = docdb_api_client.retrieve_docdb_records(
        filter_query=filter,
    ) # type = list

    result = await summary_generator.ainvoke({"data_asset": records})
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
