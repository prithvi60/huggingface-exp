from dotenv import load_dotenv
import os
import base64

# Load from .env file
load_dotenv()

# Langfuse keys from environment
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
host = os.getenv("LANGFUSE_HOST")

# Create Basic Auth header
LANGFUSE_AUTH = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

# Set OTEL exporter environment variables
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host}/api/public/otel"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

# Optional: Load Hugging Face token if needed elsewhere
hf_token = os.getenv("HF_TOKEN")

from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
 
# Create a TracerProvider for OpenTelemetry
trace_provider = TracerProvider()

# Add a SimpleSpanProcessor with the OTLPSpanExporter to send traces
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

# Set the global default tracer provider
from opentelemetry import trace
trace.set_tracer_provider(trace_provider)
tracer = trace.get_tracer(__name__)

# Instrument smolagents with the configured provider
SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

# agent logic
from smolagents import (CodeAgent, DuckDuckGoSearchTool, HfApiModel)

search_tool = DuckDuckGoSearchTool()
agent = CodeAgent(tools=[search_tool], model=HfApiModel(model_id="HuggingFaceH4/zephyr-7b-beta"))

agent.run("How many Rubik's Cubes could you fit inside the Notre Dame Cathedral?")