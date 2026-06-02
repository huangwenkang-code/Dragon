"""Centralised settings via pydantic-settings / env."""

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from env / .env file."""

    # LLM
    llm_provider: str = Field(default="openai")
    llm_backend_url: str = Field(default="https://api.openai.com/v1")
    llm_api_key: str = Field(default="")
    deep_think_llm: str = Field(default="gpt-4o")
    quick_think_llm: str = Field(default="gpt-4o-mini")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")

    # Database
    db_url: str = Field(default="")  # 优先; 空则回退到 postgres_url
    postgres_url: str = Field(
        default="postgresql://dragon:dragon@localhost:5432/dragon_engine"
    )

    # ChromaDB
    chroma_persist_dir: str = Field(default="./data/chroma")

    # Data
    tushare_token: str = Field(default="")
    akshare_enabled: bool = Field(default=True)

    # Graph
    max_recur_limit: int = Field(default=100)
    max_debate_rounds: int = Field(default=2)

    # Data source switches
    data_source_realtime: str = Field(default="mootdx")
    data_source_fundamentals: str = Field(default="tx_finance")
    data_source_sector: str = Field(default="ths_hot")
    data_source_lhb: str = Field(default="akshare")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # allow env vars not defined in this model (e.g. EM_COOKIE)
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
