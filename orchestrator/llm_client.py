from __future__ import annotations

import json
from typing import Optional

from orchestrator.llm_config import LLMConfig, load_config


class LLMClient:
    """Minimal LLM client using Microsoft Foundry / Azure Responses endpoint.

    This client sends a system+user prompt pair and returns the model's
    text completion.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or load_config()

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self.config.is_valid():
            raise RuntimeError("LLM configuration invalid; set AZURE_OPENAI_ENDPOINT (or AZURE_ENDPOINT), AZURE_OPENAI_API_KEY (or AZURE_KEY), and AZURE_DEPLOYMENT")

        # Prefer using the OpenAI Python SDK (works with Azure Foundry/OpenAI-compatible Responses API)
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            raise RuntimeError("OpenAI SDK not installed. Install with: pip install openai") from exc

        # Build base_url for SDK: ensure it's the service root (no trailing /responses)
        ep = (self.config.endpoint or "").rstrip("/")
        # If endpoint contains /responses, strip it to the root
        if ep.endswith("/responses"):
            ep = ep[: -len("/responses")]

        client = OpenAI(api_key=self.config.api_key, base_url=ep)

        # Combine system + user into a single input string (Azure Responses accepts a string model param)
        combined = system_prompt + "\n\n" + user_prompt

        resp = client.responses.create(model=self.config.deployment, input=combined)

        # Try common response locations
        # 1) modern SDK exposes `output_text`
        try:
            if hasattr(resp, "output_text") and resp.output_text:
                return resp.output_text
        except Exception:
            pass

        # 2) fallback to parsing dict-like structure
        try:
            data = resp if isinstance(resp, dict) else resp.__dict__
        except Exception:
            data = None

        if isinstance(data, dict):
            # Try nested 'output' or 'choices'
            if "output" in data and isinstance(data["output"], list):
                parts = [p for p in data["output"] if isinstance(p, str)]
                if parts:
                    return "\n".join(parts)
            choices = data.get("choices")
            if choices and isinstance(choices, list):
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
                    if "text" in first and isinstance(first["text"], str):
                        return first["text"]

        # As last resort, convert response to JSON string
        return json.dumps(resp)

