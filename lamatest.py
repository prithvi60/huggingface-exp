# test_tool_call.py
from smolagents import ToolCallingAgent, DuckDuckGoSearchTool
from agent_model_setup import LlamaCppModel


def main():
    model = LlamaCppModel()
    agent = ToolCallingAgent(
        tools=[DuckDuckGoSearchTool()],
        model=model,
        max_steps=3,
        name="searcher",
        description="Use search('…') to look things up on the web.",
    )
    result = agent.run("search('capital of France')")
    print("Final answer:", result)

if __name__ == "__main__":
    main()
