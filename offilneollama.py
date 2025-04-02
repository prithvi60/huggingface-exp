from smolagents import (CodeAgent, DuckDuckGoSearchTool, OpenAIModel)

# Load Ollama/OpenAI-compatible config
openai_base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
openai_model = os.getenv("OPENAI_MODEL", "mistral")
openai_key = os.getenv("OPENAI_API_KEY", "ollama")

model = OpenAIModel(
    model=openai_model,
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
agent.run("How many Rubik's Cubes could you fit inside the Notre Dame Cathedral?")
