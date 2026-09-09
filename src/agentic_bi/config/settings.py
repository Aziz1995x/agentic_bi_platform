from pathlib import Path
from typing import Literal
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str
    postgres_port: int
    environment: Literal["dev", "staging", "prod"]
    log_level: str = "INFO"

    llm_provider: Literal["anthropic", "openai", "groq", "local"] = "openai"
    llm_model: str = "gpt-4o-mini"    # "qwen/qwen3.6-27b"
    llm_temperature: float = 0.0

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    ollama_base_url: str = "http://localhost:11434"
    # Raw data paths — defaults assume the documented download step has
    # placed Olist CSVs at data/raw/olist/. Override via .env if needed.
    raw_data_dir: Path = Path("data/raw/olist")

    documents_dir: Path = Path("documents")

    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    faiss_index_dir: Path = Path("data/vectorstore/faiss_index")

    @property
    def customers_csv(self) -> Path:
        return self.raw_data_dir / "olist_customers_dataset.csv"

    @property
    def category_translation_csv(self) -> Path:
        return self.raw_data_dir / "product_category_name_translation.csv"

    @property
    def products_csv(self) -> Path:
        return self.raw_data_dir / "olist_products_dataset.csv"

    @property
    def sellers_csv(self) -> Path:
        return self.raw_data_dir / "olist_sellers_dataset.csv"

    @property
    def orders_csv(self) -> Path:
        return self.raw_data_dir / "olist_orders_dataset.csv"

    @property
    def order_items_csv(self) -> Path:
        return self.raw_data_dir / "olist_order_items_dataset.csv"

    @property
    def order_payments_csv(self) -> Path:
        return self.raw_data_dir / "olist_order_payments_dataset.csv"

    @property
    def order_reviews_csv(self) -> Path:
        return self.raw_data_dir / "olist_order_reviews_dataset.csv"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
