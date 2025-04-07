import gradio as gr
from agent_code import run_agent_task

def inference(prompt):
    try:
        return run_agent_task(prompt)
    except Exception as e:
        return str(e)

gr.Interface(fn=inference, 
             inputs=gr.Textbox(label="Enter your query"),
             outputs=gr.Text(label="Response"),
             title="SmolAgent HuggingFace Tool"
            ).launch(server_name="0.0.0.0", server_port=7860)
            # ).launch(server_name="127.0.0.1", server_port=7860)

