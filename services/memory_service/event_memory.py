"""
Event memory service for dragon-engine cognitive engine.

Stores and retrieves event embeddings via ChromaDB so the engine can
recall semantically similar past events.  Supports DashScope, OpenAI,
DeepSeek, Google, and Ollama embedding providers.

Adapted from TradingAgents-CN FinancialSituationMemory.
"""
import hashlib
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from shared.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports — the module is usable even when optional deps are
# missing.  Embedding generation degrades gracefully to zero-vectors.
# ---------------------------------------------------------------------------
_chromadb_available = False
_dashscope_available = False
_openai_available = False
try:
    import chromadb
    _chromadb_available = True
except ImportError:
    pass

try:
    import dashscope
    from dashscope import TextEmbedding  # noqa: F401
    _dashscope_available = True
except ImportError:
    pass

try:
    from openai import OpenAI  # noqa: F401
    _openai_available = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# ChromaDBManager — thread-safe singleton for collection access
# ---------------------------------------------------------------------------


class ChromaDBManager:
    """Thread-safe singleton ChromaDB manager to avoid concurrent collection
    creation conflicts."""

    _instance: Optional["ChromaDBManager"] = None
    _lock = threading.Lock()
    _collections: Dict[str, Any] = {}
    _client: Optional[Any] = None

    def __new__(cls) -> "ChromaDBManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        try:
            from .chromadb_config import get_optimal_chromadb_client, is_windows_11
            import platform

            self._client = get_optimal_chromadb_client()

            system = platform.system()
            if system == "Windows":
                if is_windows_11():
                    logger.info(
                        "ChromaDB Windows 11 optimized init (build %s)",
                        platform.version(),
                    )
                else:
                    logger.info("ChromaDB Windows 10 compatible init")
            else:
                logger.info("ChromaDB %s standard init", system)

            self._initialized = True
        except Exception as e:
            logger.error("ChromaDB init failed: %s", e)
            # Fallback: simplest possible config
            try:
                if _chromadb_available:
                    settings = chromadb.config.Settings(
                        allow_reset=True,
                        anonymized_telemetry=False,
                        is_persistent=False,
                    )
                    self._client = chromadb.Client(settings)
                    logger.info("ChromaDB fallback init succeeded")
                else:
                    self._client = None
                    logger.warning("ChromaDB unavailable — memory disabled")
            except Exception as backup_error:
                logger.warning("ChromaDB minimal init failed: %s", backup_error)
                self._client = None
            self._initialized = True

    def get_or_create_collection(self, name: str) -> Any:
        """Thread-safe get-or-create of a named ChromaDB collection."""
        if self._client is None:
            return _NoOpCollection(name)

        with self._lock:
            if name in self._collections:
                logger.debug("Using cached collection: %s", name)
                return self._collections[name]

            try:
                collection = self._client.get_collection(name=name)
                logger.info("Reusing existing collection: %s", name)
            except Exception:
                try:
                    collection = self._client.create_collection(name=name)
                    logger.info("Created new collection: %s", name)
                except Exception as e:
                    try:
                        collection = self._client.get_collection(name=name)
                        logger.info("Retrieved concurrently created collection: %s", name)
                    except Exception as final_error:
                        logger.error("Collection operation failed for %s: %s", name, final_error)
                        raise final_error

            self._collections[name] = collection
            return collection


class _NoOpCollection:
    """Stub collection that returns empty results when ChromaDB is unavailable."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._count = 0

    def count(self) -> int:
        return self._count

    def add(self, **kwargs: Any) -> None:
        self._count += len(kwargs.get("ids", []))

    def query(self, **kwargs: Any) -> Dict[str, List]:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


# ---------------------------------------------------------------------------
# Lifecycle stage enum-like values
# ---------------------------------------------------------------------------

class LifecycleStage:
    EMERGING = "emerging"       # Just detected, low confidence
    ACTIVE = "active"           # Confirmed, actively influencing
    PEAKING = "peaking"         # At maximum impact
    DECAYING = "decaying"       # Influence fading
    RESOLVED = "resolved"       # Event concluded


# ---------------------------------------------------------------------------
# EventMemory
# ---------------------------------------------------------------------------


class EventMemory:
    """Stores and queries event embeddings for cognitive recall.

    Each stored event tracks its lifecycle stage, creation timestamp,
    activation timestamp, and exponential decay factor so the engine can
    filter for active (still-relevant) events.
    """

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        """
        Args:
            name: ChromaDB collection name.
            config: Dict with keys:
                - llm_provider: one of "dashscope", "openai", "deepseek",
                  "google", "ollama" (default "openai").
                - backend_url: OpenAI-compatible base URL (optional).
        """
        self.config = config
        self.llm_provider = config.get("llm_provider", "openai").lower()
        self.name = name

        # Embedding length guard (default 50000 chars).
        self.max_embedding_length = int(
            os.getenv("MAX_EMBEDDING_CONTENT_LENGTH", "50000")
        )
        self.enable_embedding_length_check = (
            os.getenv("ENABLE_EMBEDDING_LENGTH_CHECK", "true").lower() == "true"
        )

        self.fallback_available = False
        self.fallback_client: Optional[Any] = None
        self.fallback_embedding: Optional[str] = None

        # Last-text metadata for debugging.
        self._last_text_info: Optional[Dict[str, Any]] = None

        # ---- Provider selection ----
        self.embedding: str = ""
        self.client: Any = None  # OpenAI client, None for DashScope, or "DISABLED"

        self._init_provider()

        # ChromaDB collection.
        self.chroma_manager = ChromaDBManager()
        self._collection = self.chroma_manager.get_or_create_collection(name)

    # ------------------------------------------------------------------
    # Provider initialization (mirrors TradingAgents-CN logic)
    # ------------------------------------------------------------------

    def _init_provider(self) -> None:
        provider = self.llm_provider

        if provider in ("dashscope", "alibaba"):
            self._init_dashscope("DashScope")

        elif provider == "qianfan":
            self._init_dashscope_with_fallback("Qianfan")

        elif provider == "deepseek":
            self._init_deepseek()

        elif provider == "google":
            self._init_google()

        elif provider == "openrouter":
            self._init_dashscope_with_fallback("OpenRouter")

        elif provider == "ollama":
            self.embedding = "nomic-embed-text"
            self.client = OpenAI(base_url=self.config.get("backend_url", "http://localhost:11434/v1"))
            logger.info("Using Ollama embedding: %s", self.embedding)

        else:
            # Default: OpenAI
            self.embedding = "text-embedding-3-small"
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                self.client = OpenAI(
                    api_key=openai_key,
                    base_url=self.config.get("backend_url", "https://api.openai.com/v1"),
                )
                logger.info("Using OpenAI embedding: %s", self.embedding)
            else:
                self.client = "DISABLED"
                logger.warning("OPENAI_API_KEY not set — embedding disabled")

    def _init_dashscope(self, label: str) -> None:
        self.embedding = "text-embedding-v3"
        self.client = None
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        if dashscope_key:
            try:
                import dashscope  # noqa: F811
                dashscope.api_key = dashscope_key
                logger.info("%s embedding configured (DashScope)", label)
            except ImportError:
                logger.error("DashScope package not installed")
                self.client = "DISABLED"
            except Exception as e:
                logger.error("%s DashScope init failed: %s", label, e)
                self.client = "DISABLED"
        else:
            self.client = "DISABLED"
            logger.warning("DASHSCOPE_API_KEY not set — %s embedding disabled", label)

    def _init_dashscope_with_fallback(self, label: str) -> None:
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        if dashscope_key:
            try:
                import dashscope  # noqa: F811
                dashscope.api_key = dashscope_key
                self.embedding = "text-embedding-v3"
                self.client = None
                logger.info("%s using DashScope embedding", label)
            except ImportError:
                logger.error("DashScope not installed for %s", label)
                self.client = "DISABLED"
            except Exception as e:
                logger.error("%s DashScope init failed: %s", label, e)
                self.client = "DISABLED"
        else:
            self.client = "DISABLED"
            logger.warning("DASHSCOPE_API_KEY not set — %s embedding disabled", label)

    def _init_deepseek(self) -> None:
        force_openai = os.getenv("FORCE_OPENAI_EMBEDDING", "false").lower() == "true"
        dashscope_key = os.getenv("DASHSCOPE_API_KEY") if not force_openai else None

        if dashscope_key:
            try:
                import dashscope  # noqa: F811
                dashscope.api_key = dashscope_key
                self.embedding = "text-embedding-v3"
                self.client = None
                logger.info("DeepSeek using DashScope embedding")
                return
            except (ImportError, Exception) as e:
                logger.warning("DashScope unavailable for DeepSeek: %s", e)

        self.embedding = "text-embedding-3-small"
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.client = OpenAI(
                api_key=openai_key,
                base_url=self.config.get("backend_url", "https://api.openai.com/v1"),
            )
        else:
            deepseek_key = os.getenv("DEEPSEEK_API_KEY")
            if deepseek_key:
                self.client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
                logger.info("DeepSeek using native embedding")
            else:
                self.client = "DISABLED"
                logger.warning("No embedding provider available for DeepSeek")

    def _init_google(self) -> None:
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if dashscope_key:
            try:
                import dashscope  # noqa: F811
                dashscope.api_key = dashscope_key
                self.embedding = "text-embedding-v3"
                self.client = None
                if openai_key:
                    self.fallback_available = True
                    self.fallback_client = OpenAI(
                        api_key=openai_key,
                        base_url=self.config.get("backend_url", "https://api.openai.com/v1"),
                    )
                    self.fallback_embedding = "text-embedding-3-small"
                    logger.info("Google using DashScope embedding (OpenAI fallback)")
                else:
                    logger.info("Google using DashScope embedding (no fallback)")
            except ImportError:
                logger.error("DashScope not installed for Google")
                self.client = "DISABLED"
            except Exception as e:
                logger.error("Google DashScope init failed: %s", e)
                self.client = "DISABLED"
        else:
            self.client = "DISABLED"
            logger.warning("DASHSCOPE_API_KEY not set — Google embedding disabled")

    # ------------------------------------------------------------------
    # Smart text truncation
    # ------------------------------------------------------------------

    def _smart_text_truncation(self, text: str, max_length: int = 8192) -> Tuple[str, bool]:
        """Truncate text intelligently at sentence/paragraph boundaries.

        Returns:
            (truncated_text, was_truncated)
        """
        if len(text) <= max_length:
            return text, False

        # Prefer sentence boundaries.
        sentences = text.split('。')  # Chinese full stop
        if len(sentences) > 1:
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + '。') <= max_length - 50:
                    truncated += sentence + '。'
                else:
                    break
            if len(truncated) > max_length // 2:
                logger.info("Truncated at sentence boundary: %d/%d chars", len(truncated), len(text))
                return truncated, True

        # Fallback: paragraph boundaries.
        paragraphs = text.split('\n')
        if len(paragraphs) > 1:
            truncated = ""
            for paragraph in paragraphs:
                if len(truncated + paragraph + '\n') <= max_length - 50:
                    truncated += paragraph + '\n'
                else:
                    break
            if len(truncated) > max_length // 2:
                logger.info("Truncated at paragraph boundary: %d/%d chars", len(truncated), len(text))
                return truncated, True

        # Last resort: keep head + tail.
        front = text[:max_length // 2]
        back = text[-(max_length // 2 - 100):]
        truncated = front + "\n...[content truncated]...\n" + back
        logger.warning("Hard truncation: %d -> %d chars", len(text), len(truncated))
        return truncated, True

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for *text* using the configured provider.

        Returns a zero-vector of length 1024 when embedding is disabled,
        the provider is unavailable, or an error occurs.
        """
        if self.client == "DISABLED":
            return [0.0] * 1024

        if not text or not isinstance(text, str) or len(text) == 0:
            return [0.0] * 1024

        text_length = len(text)

        # Length guard.
        if self.enable_embedding_length_check and text_length > self.max_embedding_length:
            logger.warning("Text too long (%s chars > %s) — skipping embedding", text_length, self.max_embedding_length)
            self._last_text_info = {
                "original_length": text_length,
                "processed_length": 0,
                "was_truncated": False,
                "was_skipped": True,
                "provider": self.llm_provider,
                "strategy": "length_limit_skip",
                "max_length": self.max_embedding_length,
            }
            return [0.0] * 1024

        self._last_text_info = {
            "original_length": text_length,
            "processed_length": text_length,
            "was_truncated": False,
            "was_skipped": False,
            "provider": self.llm_provider,
            "strategy": "direct",
        }

        # Route to the right provider.
        dashscope_providers = {"dashscope", "alibaba", "qianfan", "google", "deepseek", "openrouter"}
        if self.llm_provider in dashscope_providers and self.client is None:
            return self._embed_via_dashscope(text)
        else:
            return self._embed_via_openai_compat(text)

    def _embed_via_dashscope(self, text: str) -> List[float]:
        try:
            import dashscope  # noqa: F811
            from dashscope import TextEmbedding  # noqa: F811

            if not getattr(dashscope, 'api_key', None):
                logger.warning("DashScope API key not set")
                return [0.0] * 1024

            response = TextEmbedding.call(model=self.embedding, input=text)

            if response.status_code == 200:
                embedding = response.output['embeddings'][0]['embedding']
                logger.debug("DashScope embedding OK, dim=%d", len(embedding))
                return embedding

            error_msg = f"{response.code} - {response.message}"
            if any(kw in error_msg.lower() for kw in ('length', 'token', 'limit', 'exceed')):
                logger.warning("DashScope length limit: %s", error_msg)
                if self.fallback_available and self.fallback_client:
                    logger.info("Attempting OpenAI fallback for long text")
                    try:
                        resp = self.fallback_client.embeddings.create(
                            model=self.fallback_embedding,
                            input=text,
                        )
                        embedding = resp.data[0].embedding
                        logger.info("OpenAI fallback OK, dim=%d", len(embedding))
                        return embedding
                    except Exception as fb_err:
                        logger.error("OpenAI fallback failed: %s", fb_err)
                return [0.0] * 1024
            else:
                logger.error("DashScope API error: %s", error_msg)
                return [0.0] * 1024

        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ('length', 'token', 'limit', 'exceed', 'too long')):
                logger.warning("DashScope length exception: %s", e)
                if self.fallback_available and self.fallback_client:
                    try:
                        resp = self.fallback_client.embeddings.create(
                            model=self.fallback_embedding,
                            input=text,
                        )
                        return resp.data[0].embedding
                    except Exception as fb_err:
                        logger.error("OpenAI fallback failed: %s", fb_err)
                return [0.0] * 1024
            elif 'import' in error_str:
                logger.error("DashScope import error: %s", e)
            elif 'connection' in error_str:
                logger.error("DashScope connection error: %s", e)
            elif 'timeout' in error_str:
                logger.error("DashScope timeout: %s", e)
            else:
                logger.error("DashScope embedding exception: %s", e)
            return [0.0] * 1024

    def _embed_via_openai_compat(self, text: str) -> List[float]:
        if self.client is None or self.client == "DISABLED":
            return [0.0] * 1024

        try:
            response = self.client.embeddings.create(model=self.embedding, input=text)
            embedding = response.data[0].embedding
            logger.debug("%s embedding OK, dim=%d", self.llm_provider, len(embedding))
            return embedding

        except Exception as e:
            error_str = str(e).lower()
            length_keywords = (
                'token', 'length', 'too long', 'exceed', 'maximum', 'limit',
                'context', 'input too large', 'request too large',
            )
            if any(kw in error_str for kw in length_keywords):
                logger.warning("%s length limit — embedding degraded", self.llm_provider)
            else:
                logger.error("%s embedding error: %s", self.llm_provider, e)
            return [0.0] * 1024

    # ------------------------------------------------------------------
    # Add events (renamed from add_situations)
    # ------------------------------------------------------------------

    def add_events(self, events: List[Tuple[str, str, Optional[Dict[str, Any]]]]) -> None:
        """Store events with their vector embeddings and lifecycle metadata.

        Args:
            events: List of (event_text, summary, extra_meta) tuples.
                    extra_meta is an optional dict that can contain custom
                    lifecycle fields (stage, decay_factor, etc.).
        """
        documents: List[str] = []
        summaries: List[str] = []
        ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []

        offset = self._collection.count()
        now_iso = datetime.now(timezone.utc).isoformat()

        for i, (event_text, summary, extra_meta) in enumerate(events):
            doc_id = str(offset + i)

            # Lifecycle metadata.
            meta: Dict[str, Any] = {
                "summary": summary,
                "created_at": now_iso,
                "activated_at": extra_meta.get("activated_at", now_iso) if extra_meta else now_iso,
                "decay_factor": extra_meta.get("decay_factor", 1.0) if extra_meta else 1.0,
                "lifecycle_stage": extra_meta.get("lifecycle_stage", LifecycleStage.ACTIVE) if extra_meta else LifecycleStage.ACTIVE,
            }
            if extra_meta:
                # Carry through any custom fields.
                for key in ("source", "source_url", "confidence", "tags", "symbols"):
                    if key in extra_meta:
                        meta[key] = extra_meta[key]

            documents.append(event_text)
            summaries.append(summary)
            ids.append(doc_id)
            embeddings.append(self.get_embedding(event_text))
            metadatas.append(meta)

        self._collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids,
        )
        logger.info("Added %d event(s) to collection '%s'", len(events), self.name)

    # ------------------------------------------------------------------
    # Query events (renamed from get_memories)
    # ------------------------------------------------------------------

    def query_events(
        self,
        query_text: str,
        n_matches: int = 1,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Find semantically similar events.

        Args:
            query_text: The search query text.
            n_matches: Max number of results.
            min_similarity: Minimum similarity score (0.0-1.0) to include.

        Returns:
            List of dicts with keys: event_text, summary, similarity,
            distance, and any stored metadata fields.
        """
        query_embedding = self.get_embedding(query_text)

        if all(v == 0.0 for v in query_embedding):
            logger.debug("Query embedding is a zero vector — returning empty results")
            return []

        collection_count = self._collection.count()
        if collection_count == 0:
            logger.debug("Event memory is empty")
            return []

        actual_n = min(n_matches, collection_count)

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_n,
            )

            events: List[Dict[str, Any]] = []
            if results and 'documents' in results and results['documents']:
                documents = results['documents'][0]
                metadatas_list = results.get('metadatas', [[]])[0]
                distances = results.get('distances', [[]])[0]

                for i, doc in enumerate(documents):
                    md = metadatas_list[i] if i < len(metadatas_list) else {}
                    distance = distances[i] if i < len(distances) else 1.0
                    similarity = 1.0 - distance

                    if similarity < min_similarity:
                        continue

                    events.append({
                        "event_text": doc,
                        "summary": md.get("summary", ""),
                        "similarity": similarity,
                        "distance": distance,
                        "created_at": md.get("created_at"),
                        "activated_at": md.get("activated_at"),
                        "decay_factor": md.get("decay_factor", 1.0),
                        "lifecycle_stage": md.get("lifecycle_stage", LifecycleStage.ACTIVE),
                        "source": md.get("source"),
                        "source_url": md.get("source_url"),
                        "confidence": md.get("confidence"),
                        "tags": md.get("tags"),
                        "symbols": md.get("symbols"),
                    })

            logger.debug("Event query returned %d match(es)", len(events))
            return events

        except Exception as e:
            logger.error("Event query failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Query active events
    # ------------------------------------------------------------------

    def query_active_events(
        self,
        query_text: str,
        n_matches: int = 5,
    ) -> List[Dict[str, Any]]:
        """Query events that are still in an active lifecycle stage.

        Filters out events with stage 'resolved' or 'decaying' (unless
        decay_factor is still above 0.3).

        Args:
            query_text: Search query text.
            n_matches: Max results to return.

        Returns:
            Filtered list of active events.
        """
        # Fetch more than needed and post-filter.
        raw_results = self.query_events(query_text, n_matches=n_matches * 3)

        active_stages = {LifecycleStage.EMERGING, LifecycleStage.ACTIVE, LifecycleStage.PEAKING}
        active: List[Dict[str, Any]] = []

        for evt in raw_results:
            stage = evt.get("lifecycle_stage", "")
            decay = float(evt.get("decay_factor", 1.0))
            if stage in active_stages or (stage == LifecycleStage.DECAYING and decay > 0.3):
                active.append(evt)
            if len(active) >= n_matches:
                break

        logger.debug("Active events query: %d raw -> %d active", len(raw_results), len(active))
        return active

    # ------------------------------------------------------------------
    # Update event lifecycle
    # ------------------------------------------------------------------

    def update_event_lifecycle(
        self,
        doc_id: str,
        stage: Optional[str] = None,
        decay_factor: Optional[float] = None,
        activated_at: Optional[str] = None,
    ) -> None:
        """Update lifecycle metadata for an existing event.

        Note: ChromaDB metadata updates are limited; this re-adds the
        document with the same ID, which overwrites the existing entry.
        """
        try:
            results = self._collection.get(ids=[doc_id])
            if not results or not results['documents']:
                logger.warning("Event %s not found for lifecycle update", doc_id)
                return

            existing_doc = results['documents'][0]
            existing_meta = results['metadatas'][0] if results['metadatas'] else {}

            new_meta = dict(existing_meta)
            if stage is not None:
                new_meta["lifecycle_stage"] = stage
            if decay_factor is not None:
                new_meta["decay_factor"] = decay_factor
            if activated_at is not None:
                new_meta["activated_at"] = activated_at

            # Upsert by ID (overwrites).
            self._collection.upsert(
                documents=[existing_doc],
                metadatas=[new_meta],
                ids=[doc_id],
            )
            logger.info("Updated lifecycle for event %s: stage=%s, decay=%s", doc_id, stage, decay_factor)

        except Exception as e:
            logger.error("Failed to update lifecycle for event %s: %s", doc_id, e)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_embedding_config_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enable_embedding_length_check,
            "max_embedding_length": self.max_embedding_length,
            "max_embedding_length_formatted": f"{self.max_embedding_length:,} chars",
            "provider": self.llm_provider,
            "client_status": "DISABLED" if self.client == "DISABLED" else "ENABLED",
        }

    def get_last_text_info(self) -> Optional[Dict[str, Any]]:
        return getattr(self, '_last_text_info', None)

    def get_cache_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "collection_count": self._collection.count(),
            "client_status": "enabled" if self.client != "DISABLED" else "disabled",
            "embedding_model": self.embedding,
            "provider": self.llm_provider,
        }
        if self._last_text_info:
            info["last_text_processing"] = self._last_text_info
        return info

    @property
    def count(self) -> int:
        return self._collection.count()


# ---------------------------------------------------------------------------
# Module test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== EventMemory smoke test ===\n")

    config = {
        "llm_provider": "openai",
        "backend_url": "https://api.openai.com/v1",
    }

    memory = EventMemory(name="test_events", config=config)

    # Add sample events.
    test_events = [
        (
            "央行宣布降准0.5个百分点，释放长期流动性约1万亿元",
            "央行降准释放流动性",
            {
                "source": "PBOC",
                "confidence": 0.95,
                "tags": ["货币政策", "降准", "流动性"],
                "symbols": ["000001", "600036"],
                "lifecycle_stage": LifecycleStage.ACTIVE,
                "decay_factor": 1.0,
            },
        ),
        (
            "新能源汽车销量同比增长45%，渗透率突破40%",
            "新能源汽车销量大增",
            {
                "source": "CAAM",
                "confidence": 0.9,
                "tags": ["新能源汽车", "销量"],
                "symbols": ["300750", "002594"],
                "lifecycle_stage": LifecycleStage.ACTIVE,
                "decay_factor": 0.9,
            },
        ),
        (
            "美联储维持利率不变，点阵图显示年内降息预期",
            "美联储暂停加息释放降息信号",
            {
                "source": "FOMC",
                "confidence": 0.85,
                "tags": ["美联储", "利率", "降息"],
                "symbols": ["600519", "000858"],
                "lifecycle_stage": LifecycleStage.EMERGING,
                "decay_factor": 1.0,
            },
        ),
    ]
    memory.add_events(test_events)

    # Query.
    query = "货币政策宽松，央行可能继续降准降息"
    print(f"Query: {query}\n")
    results = memory.query_events(query, n_matches=2)

    for i, r in enumerate(results, 1):
        print(f"Match {i}:")
        print(f"  Similarity: {r['similarity']:.3f}")
        print(f"  Summary:    {r['summary']}")
        print(f"  Stage:      {r['lifecycle_stage']}")
        print(f"  Decay:      {r['decay_factor']}")
        print()

    # Active events.
    print("--- Active events ---")
    active = memory.query_active_events(query, n_matches=3)
    for i, r in enumerate(active, 1):
        print(f"  {i}. [{r['lifecycle_stage']}] {r['summary']} (sim={r['similarity']:.2f})")

    # Cache info.
    info = memory.get_cache_info()
    print(f"\nCache info: {info}")
    print(f"Embedding status: {memory.get_embedding_config_status()}")

    print("\n=== Smoke test complete ===")
