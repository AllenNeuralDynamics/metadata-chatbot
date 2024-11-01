from fastapi import FastAPI
import uvicorn
from metadata_chatbot.agents.GAMER import GAMER

app = FastAPI()

@app.get("/summary/{name}")
async def REST_summary(name: str):
    query = f"Give me a detailed 3 sentence summary of the asset name: {name}. Do not include a starting phrase like here is a 3 sentence summary of this asset."
    model = GAMER()
    result = await model.ainvoke(query)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)