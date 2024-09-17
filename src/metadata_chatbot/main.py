from fastapi import FastAPI
import uvicorn
from metadata_chatbot.chat import get_summary

app = FastAPI()

@app.get("/summary/{_id}")
def REST_summary(_id: str):
    return get_summary(_id)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)