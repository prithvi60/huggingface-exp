import re
import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool

from dotenv import load_dotenv
load_dotenv()


@tool
def visit_webpage(url: str) -> str:
    """
    Visits a webpage and returns its content as Markdown.

    Args:
        url (str): The full URL of the webpage to visit.

    Returns:
        str: The content of the webpage converted to Markdown.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        markdown_content = markdownify(response.text).strip()
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

# --- Updated Agent Setup ---
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    HfApiModel,
    DuckDuckGoSearchTool,
)
# --- LLM used ---
model = HfApiModel(model_id="mistralai/Mistral-7B-Instruct-v0.2")

# Create a ToolCallingAgent for search + webpage tools
web_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), visit_webpage],
    model=model,
    max_steps=10,
    name="search",
    description="Runs web searches and reads web pages. Give it a query or URL."
)

# Now CodeAgent directly accepts this ToolCallingAgent in managed_agents
manager_agent = CodeAgent(
    tools=[],  # No direct tools here
    model=model,
    managed_agents=[web_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)

# Run the task
answer = manager_agent.run("What do you think will happen in 2025 with AI Agents? Compare the usage in production in the past 12 months.")
print(answer)
