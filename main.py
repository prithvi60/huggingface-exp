import os
import base64
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from time import time
# from agent_code import raw_agent_run
from agent_code_local import raw_agent_run

# Load environment variables
load_dotenv()

# -------------------------------
# 🌐 Langfuse + OpenTelemetry Setup
# -------------------------------
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
host = os.getenv("LANGFUSE_HOST")

LANGFUSE_AUTH = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host}/api/public/otel"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry import trace

trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(trace_provider)
tracer = trace.get_tracer(__name__)

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

# -------------------------------
# 🔁 Traced Task Function
# -------------------------------
def run_agent_task(user_input: str) -> str:
    with tracer.start_as_current_span("smolagent_openai_run") as span:
        start = time()
        try:
            result = raw_agent_run(user_input)

            prompt_tokens = len(user_input.split())
            completion_tokens = len(result.split())
            total_tokens = prompt_tokens + completion_tokens
            cost_usd = total_tokens * 0.000002  # gpt-3.5-turbo estimated cost

            span.set_attribute("agent.name", "smolagent")
            span.set_attribute("model.id", "gpt-3.5-turbo")
            span.set_attribute("tokens.prompt", prompt_tokens)
            span.set_attribute("tokens.completion", completion_tokens)
            span.set_attribute("tokens.total", total_tokens)
            span.set_attribute("cost.usd", round(cost_usd, 6))
            span.set_attribute("execution_time_ms", int((time() - start) * 1000))
            span.set_attribute("status", "success")

            return result
        except Exception as e:
            span.set_attribute("status", "error")
            span.set_attribute("error.message", str(e))
            raise

# -------------------------------
# 🚀 FastAPI Setup
# -------------------------------
app = FastAPI()

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

# -------------------------------
# 🎛️ Gradio Interface
# -------------------------------
def gradio_interface(prompt):
    return run_agent_task(prompt)

gradio_app = gr.Interface(fn=gradio_interface, inputs="text", outputs="text", title="SmolAgent Gradio")
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

# -------------------------------
# ✅ Uvicorn entry point Use http://localhost:8080/ for local browser not postman
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
