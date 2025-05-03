# agent_model_setup.py
import os
import re
import json
import uuid
from copy import deepcopy
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from llama_cpp import Llama
from smolagents import Model, ChatMessage, MessageRole, Tool

# ─── Configuration (high‑quality & low‑latency on <8GB RAM) ─────────────────────────────────────
# Retain medium quant variant (Q4_K_M) for best quality vs footprint (~4.8GB)
MODEL_PATH   = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
# MODEL_PATH   = "models/ministral-3b-instruct.Q4_0.gguf"
# Mistal 7B
# Full context window for richer responses
N_CTX        = 4096
# CPU-only; set >0 if GPU offload available
N_GPU_LAYERS = 0
# Sampling parameters
TEMPERATURE  = 0.7
# Allow up to 1024 tokens per response
MAX_TOKENS   = 1024
# Threading: cap threads to balance CPU utilization on VPS
import os
N_THREADS    = min(4, max(1, os.cpu_count() or 1))
# llama.cpp flags: use mmap for faster load, disable mlock to reduce memory pressure

# ─── Vendored Utility Functions ─────────────────────────────────────────────────

def _remove_stop_sequences(text: str, stops: List[str]) -> str:
    """Strip any trailing stop sequences from the model output."""
    for stop in stops:
        if text.endswith(stop):
            return text[:-len(stop)]
    return text


def _parse_json_blob(text: str) -> Any:
    """Extract first JSON object from a string."""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in text")
    return json.loads(match.group(0))


dataclass
class ChatMessageToolCallDefinition:
    name: str
    arguments: Any
    description: Optional[str] = None

dataclass
class ChatMessageToolCall:
    id: str
    type: str
    function: ChatMessageToolCallDefinition


def _get_tool_call_from_text(
    text: str,
    tool_name_key: str = "name",
    tool_arguments_key: str = "arguments",
) -> ChatMessageToolCall:
    """Parse a JSON blob into a ChatMessageToolCall, unwrapping schema-only dicts."""
    json_obj = _parse_json_blob(text)
    name = json_obj.get(tool_name_key)
    args = json_obj.get(tool_arguments_key, {})
    # If args is just the input schema description, extract the human string
    if isinstance(args, dict) and set(args.keys()) == {"type", "description"}:
        args = args.get("description")
    return ChatMessageToolCall(
        id=str(uuid.uuid4()),
        type="function",
        function=ChatMessageToolCallDefinition(
            name=name,
            arguments=args,
        ),
    )(
        id=str(uuid.uuid4()),
        type="function",
        function=ChatMessageToolCallDefinition(
            name=json_obj.get(tool_name_key),
            arguments=json_obj.get(tool_arguments_key, {}),
        ),
    )

# ─── Llama.cpp Model Adapter ───────────────────────────────────────────────────
class LlamaCppModel(Model):
    """Adapter: use a local GGUF quantized model via llama.cpp for smolagents."""
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        n_ctx: int = N_CTX,
        n_gpu_layers: int = N_GPU_LAYERS,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        **kwargs,
    ):
        super().__init__(flatten_messages_as_text=True, **kwargs)
        self.model_id = "llama-cpp"
        # Initialize llama.cpp with threading and optional GPU
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=N_THREADS,
            use_mlock=True,
            use_mmap=True,
            logits_all=False,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        # Prepare model kwargs (tools, stops, etc.)
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        msgs = completion_kwargs.pop("messages")
        stops = completion_kwargs.pop("stop", []) or []

        # Build simple prompt
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)

        # Run inference (eager model eval)
        out = self.llm(
            prompt=prompt,
            max_tokens=completion_kwargs.get("max_tokens", self.max_tokens),
            temperature=completion_kwargs.get("temperature", self.temperature),
            stop=stops,
        )
        text = out.get("choices", [{}])[0].get("text", "")
        if stops:
            text = _remove_stop_sequences(text, stops)

        # Wrap response
        chat_msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=text,
            raw={"llama_out": out, **completion_kwargs},
        )
        # Ensure final answer tool is called if needed
        return chat_msg
