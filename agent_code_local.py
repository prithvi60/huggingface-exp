from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    DuckDuckGoSearchTool,
)
from agent_model_setup import SimpleLlamaCppModel 
import os

# 1️⃣  load llama.cpp model
mistral_llm = SimpleLlamaCppModel(
    model_path=os.getenv("LLAMACPP_MODEL_PATH", "models/mistral.q4.gguf")
)

# 2️⃣  keep your existing search tool agent
search_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool()],
    model=mistral_llm,
    max_steps=10,
    name="search",
    description="Runs web searches for you.",
)

# 3️⃣  manager orchestrates everything
manager_agent = CodeAgent(
    tools=[],
    model=mistral_llm,     # ← swap‑in here too
    managed_agents=[search_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)

def raw_agent_run(user_input: str) -> str:
    return manager_agent.run(user_input)
