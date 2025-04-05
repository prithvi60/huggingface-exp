import os

from smolagents import (CodeAgent, DuckDuckGoSearchTool,OpenAIServerModel)


# Load Ollama/OpenAI-compatible config
openai_base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
openai_model = os.getenv("OPENAI_MODEL", "mistral-7b:latest")
openai_key = os.getenv("OPENAI_API_KEY", "ollama")

model = OpenAIServerModel(
    model_id=openai_model,
    api_base=openai_base_url,
    api_key=openai_key
)

# Tools
search_tool = DuckDuckGoSearchTool()

# Agent
agent = CodeAgent(
    tools=[search_tool],
    model=model
)

# Run agent
agent.run("whats 1 + 2 ?")
