from llama_cpp import Llama

class SimpleLlamaCppModel:
    def __init__(self, model_path, n_ctx=4096, n_gpu_layers=0, temperature=0.7, max_tokens=1024):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            use_mlock=True,
            logits_all=False,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, messages, stop_sequences=None, **kwargs):
        prompt = self.convert_messages_to_prompt(messages)
        print("📨 Prompt sent to LLM:", prompt[:300])  # Print only first 300 chars

        try:
            output = self.llm(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=stop_sequences or ["</s>"],
            )
            print("✅ LLM output raw:", output)
            return {"content": output["choices"][0]["text"].strip()}
        except Exception as e:
            print("❌ Error running LLM:", e)
            return {"content": "[Error generating response]"}

def convert_messages_to_prompt(self, messages):
    prompt = ""

    # Add default system message if not provided
    if not any(m.get("role") == "system" for m in messages):
        prompt += (
            "You are an AI agent with access to tools. "
            "You can think step-by-step and call tools like:\n"
            "tool_name('argument')\n"
            "You will receive observations and can continue from there.\n\n"
        )

    for m in messages:
        role = m.get("role")
        content = m.get("content")

        # These lines ensure system messages and dynamic tool info show up
        if role == "system":
            prompt += f"{content.strip()}\n\n"

        elif role == "user":
            prompt += f"Task: {content.strip()}\n\n"

        elif role == "assistant":
            prompt += f"{content.strip()}\n\n"

        elif role == "tool":
            prompt += f"[Observation] {content.strip()}\n\n"

    prompt += "Next step:\n"
    return prompt
