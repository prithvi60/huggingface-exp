from fastapi import FastAPI, Request
from pydantic import BaseModel
from agent_code import run_agent_task
import uvicorn

app = FastAPI()

class Query(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"msg": "SmolAgent API is running. Use POST /run with {'prompt': 'your task'}"}

@app.post("/run")
def run_query(data: Query):
    try:
        result = run_agent_task(data.prompt)
        return {"response": result}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # uvicorn.run(app, host="127.0.0.1", port=8000)

