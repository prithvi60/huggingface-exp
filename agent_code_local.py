import os
from dotenv import load_dotenv
from smolagents import CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool
from agent_model_setup import LlamaCppModel

load_dotenv()  # if you later add env-based overrides

# 1️⃣ Load local Mistral model
mistral_llm = LlamaCppModel()

# 2️⃣ Define your search tool agent
search_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool()],
    model=mistral_llm,
    max_steps=10,
    name="search",
    description="Runs web searches for you. Provide the query as the first argument.",
)

# 3️⃣ Manager orchestrates everything
manager_agent = CodeAgent(
    tools=[],                # no direct tools here; handled by search_agent
    model=mistral_llm,
    managed_agents=[search_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)

# 4️⃣ Entry point for your application
def raw_agent_run(user_input: str) -> str:
    """Run the manager agent on user input and return its final answer."""
    result = manager_agent.run(user_input)
    return result.final_answer
