"""Typed application configuration.

Settings are read from environment variables (case-insensitive) and, when
present, a local `.env` file. See `.env.example` for the full list. Cached so
the settings object is constructed once per process.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Runtime
    environment: str = "development"
    log_level: str = "info"
    service_name: str = "chat-service"
    host: str = "0.0.0.0"
    port: int = 8002

    # Gateway trust.
    internal_service_token: str = ""

    # Background warm-up after startup (Qdrant bootstrap, embedding model,
    # product-help index — see app.main._warm_up). Tests turn it off.
    startup_warmup: bool = True

    # Shared infrastructure (clients live in app/core; wired into readiness).
    mongo_uri: str = ""
    redis_url: str = "redis://red-d7v2vglb910c73akocvg:6379"

    # MongoDB pooling / server selection.
    mongo_db: str = ""
    mongo_max_pool_size: int = 20
    mongo_min_pool_size: int = 0
    mongo_server_selection_timeout_ms: int = 5000

    # Qdrant (vector DB). Client + collection bootstrap live in app/rag.
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str = ""  # required for Qdrant Cloud; leave empty for local
    qdrant_collection_public: str = "content_public"
    qdrant_collection_private: str = "content_private"
    qdrant_vector_size: int = 384  # all-MiniLM-L6-v2 dimensionality
    qdrant_distance: str = "Cosine"

    # Embeddings (engine lives in app/rag/embedder.py).
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_cache_ttl: int = 86400  # 24 hours (chat:embed:* cache)
    embedding_batch_size: int = 32

    # AI — LLM generation (provider-agnostic facade in app/llm/provider.py).
    llm_provider: str = "groq"

    # Groq generation (app/llm/groq_client.py, lazy-loaded on first use).
    # Model names are NEVER hardcoded in code — they come from these settings.
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    # Ordered fallbacks tried if the primary model is unavailable (comma-separated).
    groq_model_fallbacks: str = "qwen/qwen3-32b,llama-3.1-8b-instant"
    groq_temperature: float = 0.1        # low for factual, grounded answers
    groq_max_output_tokens: int = 2048   # cap to contain cost + latency
    groq_timeout_seconds: float = 30.0   # per-request budget (prioritise responsiveness)
    groq_max_retries: int = 1            # transient-error retries per model (SDK-level)
    # Groq's Whisper transcription (audio.transcriptions.create) — a distinct
    # API surface from chat completions above, so it gets its own model
    # setting rather than reusing groq_model/groq_model_chain (those are
    # fallback-chained chat models, not applicable here). The full model
    # (not "-turbo", the distilled/pruned variant) — code-switched speech
    # (e.g. Hindi transliterated to Roman script, "Hinglish") is exactly the
    # kind of harder edge case a distilled model tends to lose accuracy on
    # first; Groq's inference is fast enough regardless that the full
    # model's extra latency is still fine for ~4s live-caption chunks.
    groq_whisper_model: str = "whisper-large-v3"

    @property
    def groq_model_chain(self) -> list[str]:
        """Primary model followed by configured fallbacks, de-duplicated, in order."""
        chain = [self.groq_model, *self.groq_model_fallbacks.split(",")]
        seen: set[str] = set()
        ordered: list[str] = []
        for raw in chain:
            model = raw.strip()
            if model and model not in seen:
                seen.add(model)
                ordered.append(model)
        return ordered


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
