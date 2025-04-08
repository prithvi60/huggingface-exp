import os
from fastapi import FastAPI
from pydantic import BaseModel
from agent_code import run_agent_task
import gradio as gr
from fastapi.middleware.cors import CORSMiddleware

# FastAPI app
app = FastAPI()

# Enable CORS for cross-origin support (important if frontend is separate)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Gradio interface
def gradio_interface(prompt):
    return run_agent_task(prompt)

gradio_app = gr.Interface(fn=gradio_interface, inputs="text", outputs="text", title="SmolAgent Gradio")

# ✅ Mount Gradio inside FastAPI
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

# ✅ Uvicorn entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
