from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    client_id: str
    secret_key: str
    redirect_uri: str
    access_token: str


def load_config() -> Config:
    return Config(
        client_id=os.getenv("FYERS_CLIENT_ID", ""),
        secret_key=os.getenv("FYERS_SECRET_KEY", ""),
        redirect_uri=os.getenv("FYERS_REDIRECT_URI", ""),
        access_token=os.getenv("FYERS_ACCESS_TOKEN", ""),
    )