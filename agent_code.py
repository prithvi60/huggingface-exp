from smolagents import (
CodeAgent,
ToolCallingAgent,
DuckDuckGoSearchTool,
LiteLLMModel,
)
import os
from dotenv import load_dotenv

load_dotenv()


# model = HfApiModel()
model = LiteLLMModel(model_id="gpt-4o-mini",api_key=os.getenv("OPENAI_API_KEY"))
# model = LiteLLMModel(model_id="gemini/gemini-2.0-flash-exp")

search_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    max_steps=10,
    name="search",
    description="Runs web searches for you. Give it your query as an argument.",
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)

# answer = manager_agent.run("What year was the movie 'Rebel Without a Cause' released and who was the star of it?")
# answer = manager_agent.run("What movie was released in 1955 that stars James Dean and what is it's Rotten Tomatoes score?")

# print(answer)

# -------------------------------
# 🌐 The function we use in main.py to provide to api
# -------------------------------
def raw_agent_run(user_input: str) -> str:
    return manager_agent.run(user_input)

