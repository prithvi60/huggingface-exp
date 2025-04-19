from llama_cpp import Llama

llm = Llama(model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_ctx=4096)

prompt = "[User] What can you do?\n[Assistant]"
output = llm(prompt=prompt, max_tokens=256, temperature=0.7)
print("🔁 Result:", output["choices"][0]["text"])
