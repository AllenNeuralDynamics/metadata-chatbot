from fastapi import FastAPI
import uvicorn
from metadata_chatbot.bedrock_model.chat import get_summary

app = FastAPI()

@app.get("/summary/{_id}")
def REST_summary(_id: str):
    result = get_summary(_id)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)