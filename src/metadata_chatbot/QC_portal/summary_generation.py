"""REST API to generate summaries for given asset name"""

import json

import panel as pn
import uvicorn
from aind_data_access_api.document_db import MetadataDbClient
from fastapi import FastAPI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from metadata_chatbot.utils import HAIKU_3_5_LLM, SONNET_3_7_LLM

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


from typing import Any, Dict, List

import httpx
import panel as pn
from fastapi import FastAPI
from panel.io.fastapi import add_application
from pydantic import BaseModel

pn.extension()


app = FastAPI()


@app.get("/summary/{name}")
async def REST_summary(name: str):
    """Invoking GAMER to generate summary of asset
    Args:
        name: The name of the asset to summarize

    Returns:
        A generated summary of the asset
    """
    try:
        filter = {"name": name}
        records = docdb_api_client.retrieve_docdb_records(
            filter_query=filter,
        )  # type = list

        if not records:
            return {"error": f"No asset found with name: {name}"}

        result = await summary_generator.ainvoke({"data_asset": records})
        return result
    except Exception as e:
        return {"error": f"Error generating summary: {str(e)}"}


# Add Panel application at /panel path
@add_application("/panel", app=app, title="GAMER Asset Summary Interface")
def create_panel_app():
    # Create Panel components
    name_input = pn.widgets.TextInput(
        name="Asset Name", placeholder="Enter asset name"
    )
    submit_button = pn.widgets.Button(
        name="Generate Summary", button_type="primary"
    )
    result_pane = pn.pane.JSON(height=500, depth=3)
    status = pn.pane.Markdown("")

    # Create a function that will be called when the button is clicked
    async def get_summary(event):
        status.object = "⏳ Generating summary..."
        try:
            # Use httpx to make an async HTTP request to our own API endpoint
            async with httpx.AsyncClient() as client:
                response = await client.get(f"/summary/{name_input.value}")
                if response.status_code == 200:
                    result = response.json()
                    result_pane.object = result
                    status.object = "✅ Summary generated successfully!"
                else:
                    status.object = f"❌ Error: Received status code {response.status_code}"
        except Exception as e:
            status.object = f"❌ Error: {str(e)}"

    # Bind the function to the button click event
    submit_button.on_click(get_summary)

    # Layout the components
    header = pn.pane.Markdown("# GAMER Asset Summary Tool")
    description = pn.pane.Markdown(
        """
        Enter the name of an asset to generate a summary using GAMER.
        The summary will be displayed below after processing.
        """
    )

    input_row = pn.Row(name_input, submit_button)

    # Return the complete layout
    return pn.Column(
        header,
        description,
        input_row,
        status,
        pn.pane.Markdown("## Summary Results"),
        result_pane,
    )


# If you want to run this directly with uvicorn (not necessary if importing elsewhere)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
