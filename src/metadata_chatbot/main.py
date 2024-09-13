from fastapi import FastAPI
from .chat import get_summary

app = FastAPI()

@app.get("/summary/{_id}")
def REST_Summary(_id: str):
    return get_summary(_id)