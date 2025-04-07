# We are using open ai api instead of inference to test deployment
import os
from smolagents import CodeAgent, tool
from huggingface_hub import list_models
from openai import OpenAI

# --- 1. TOOL: Hugging Face Model Downloader ---
@tool
def model_download_tool(task: str) -> str:
    """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint.

    Args:
        task: The task for which to get the download count.
    """
    most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
    
    return most_downloaded_model.id

# --- 2. PATCHED OpenAI MODEL ---
class PatchedOpenAIServerModel:
    def __init__(self, api_key=None, model_id="gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_id = model_id
        self.client = OpenAI(api_key=self.api_key)

    def run(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def __call__(self, prompt: str) -> str:
        return self.run(prompt)

# --- 3. MODEL + AGENT SETUP ---
openai_model = PatchedOpenAIServerModel(model_id="gpt-3.5-turbo")

agent = CodeAgent(
    tools=[model_download_tool],
    model=openai_model
)

# --- 4. MAIN FUNCTION TO RUN ---
def run_agent_task(user_input: str) -> str:
    return agent.run(user_input)

# --- 5. EXAMPLE CALL ---
if __name__ == "__main__":
    query = "What's the most downloaded model for text-classification?"
    print(run_agent_task(query))
