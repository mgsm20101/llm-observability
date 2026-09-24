from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Tracing — a local JSONL sink, see tracer.py. The hosted Langfuse client
    # this project was written against needed an account and two required keys,
    # which meant the project could not run at all without them. It now runs
    # with no credentials and no network; the keys below are optional and unused
    # by default, kept only so a hosted backend can be wired back in later.
    trace_dir: Path = ROOT / "runs"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Ollama — `gemma3:4b` is what the local server actually serves. The former
    # default `qwen3:4b` lives in a second model store the running server cannot
    # see, so every call failed on a model-not-found path.
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "gemma3:4b"

    # Retrieval — an in-process store over a small local corpus, not Qdrant.
    # Qdrant was a fourth service to stand up for a project whose subject is
    # observability, not vector search.
    corpus_path: Path = ROOT / "data" / "corpus.jsonl"
    embed_model: str = "intfloat/multilingual-e5-base"
    top_k: int = 3

    # Cost tracking (USD per 1M tokens). Local inference is billed at zero;
    # the hosted prices exist to express what the same traffic would have cost.
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
