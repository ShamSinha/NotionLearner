from __future__ import annotations

from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    notion_api_key: str
    notion_database_id: str
    openai_api_key: str = "ollama"
    openai_base_url: Optional[str] = "http://localhost:11434/v1"
    api_secret: str

    # Dual models: fast categorize vs deeper analysis
    openai_model: str = "qwen3:8b"  # legacy alias → analyze model
    categorize_model: str = "qwen3:4b"
    analyze_model: str = "qwen3:8b"
    embedding_model: str = "nomic-embed-text"

    # Unload Ollama models after this many seconds idle (0 = never)
    ollama_idle_unload_seconds: int = 300

    # Whisper fallback when YouTube captions missing
    enable_whisper_fallback: bool = True
    whisper_model: str = "small"  # tiny|base|small|medium — small is good on M4 16GB

    # Keep Notion pages clean by default. Enable to append the full extracted source.
    notion_include_source_content: bool = False

    learning_categories: str = (
        "Information Theory,"
        "Convex Optimization,"
        "Linear Algebra,"
        "Probability & Statistics,"
        "Machine Learning,"
        "Deep Learning,"
        "Computer Vision,"
        "NLP / LLMs,"
        "Reinforcement Learning,"
        "Signal Processing,"
        "Algorithms,"
        "Other"
    )

    @property
    def category_list(self) -> List[str]:
        return [c.strip() for c in self.learning_categories.split(",") if c.strip()]


settings = Settings()
