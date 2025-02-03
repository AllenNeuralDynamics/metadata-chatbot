"""REST API to generate summaries for given asset name"""

import uvicorn
from fastapi import FastAPI

from metadata_chatbot.agents.GAMER import GAMER

app = FastAPI()


@app.get("/summary/{name}")
async def REST_summary(name: str):
    """Invoking GAMER to generate summary of asset"""
    query = (
        f"Provide a detailed 3 sentence summary of the asset name:{name}."
        "Get straight to the point in your answer"
    )
    model = GAMER()
    result = await model.ainvoke(query)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
