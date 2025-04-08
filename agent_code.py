import os
from smolagents import CodeAgent, LiteLLMModel, tool
from huggingface_hub import list_models

@tool
def model_download_tool(task: str) -> str:
    """
    Returns the most downloaded model of a given task on the Hugging Face Hub.
    
    Args:
        task: The task to evaluate.
    """
    most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
    return most_downloaded_model.id

# Create a model instance using LiteLLMModel
openai_model = LiteLLMModel(
    model_id="gpt-3.5-turbo",  # or any desired model id
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize the CodeAgent with the LiteLLMModel and the tool
agent = CodeAgent(
    tools=[model_download_tool],
    model=openai_model
)

def run_agent_task(user_input: str) -> str:
    return agent.run(user_input)

