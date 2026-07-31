from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class LLMConfig:
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    deployment: Optional[str] = None

    def is_valid(self) -> bool:
        return bool(self.endpoint and self.api_key and self.deployment)

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"LLMConfig(endpoint={'set' if self.endpoint else 'unset'}, deployment={self.deployment})"


def load_config(path: Optional[str] = None) -> LLMConfig:
    """Load LLM configuration.

    Priority:
      - If `path` provided and exists, load env vars from it (dotenv file)
      - Else, load environment variables from process environment
    """
    if path:
        load_dotenv(path)
    else:
        # load .env in repo root if present
        load_dotenv()

    # Support multiple env var names used across examples and SDKs
    endpoint = (
        os.getenv("AZURE_OPENAI_ENDPOINT")
        or os.getenv("AZURE_ENDPOINT")
        or os.getenv("LLM_ENDPOINT")
    )
    api_key = (
        os.getenv("AZURE_OPENAI_API_KEY")
        or os.getenv("AZURE_KEY")
        or os.getenv("LLM_API_KEY")
    )
    deployment = (
        os.getenv("AZURE_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or os.getenv("LLM_DEPLOYMENT")
    )

    return LLMConfig(endpoint=endpoint, api_key=api_key, deployment=deployment)
