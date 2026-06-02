"""Monster report text-embedding store backed by ChromaDB + sentence-transformers.

Provides semantic similarity search over historical monster markdown reports.
Lazy-loads the embedding model and ChromaDB collection on first use.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import yaml

from services.monster_matcher.matcher import CandidateFeatures
from shared.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "monster_reports"
CHROMA_PERSIST_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data_lake", "chroma_db"
)
MONSTERS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data_lake", "monsters"
)

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_model = None
_model_lock = threading.Lock()

_chroma_client = None
_collection = None
_chroma_lock = threading.Lock()

_chroma_available = False
_st_available = False


def _ensure_chroma():
    """Lazy-import and initialize ChromaDB persistent client."""
    global _chroma_client, _collection, _chroma_available

    if _chroma_client is not None:
        return

    with _chroma_lock:
        if _chroma_client is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

            _chroma_client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIR,
                settings=Settings(anonymized_telemetry=False),
            )
            _chroma_available = True
            logger.info("ChromaDB persistent client ready at %s", CHROMA_PERSIST_DIR)
        except Exception as e:
            logger.warning("ChromaDB unavailable: %s", e)
            _chroma_client = None
            _chroma_available = False


def _ensure_model():
    """Lazy-load the sentence-transformers model (thread-safe)."""
    global _model, _st_available

    if _model is not None:
        return

    with _model_lock:
        if _model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(MODEL_NAME)
            _st_available = True
            logger.info("sentence-transformers model loaded: %s", MODEL_NAME)
        except Exception as e:
            logger.warning("sentence-transformers unavailable: %s", e)
            _model = None
            _st_available = False


def _get_collection():
    """Get or create the monster_reports ChromaDB collection."""
    global _collection

    _ensure_chroma()
    if not _chroma_available:
        return None

    if _collection is not None:
        return _collection

    with _chroma_lock:
        if _collection is not None:
            return _collection

        try:
            _collection = _chroma_client.get_collection(name=COLLECTION_NAME)
            logger.info("Reusing existing collection '%s' (%d docs)",
                        COLLECTION_NAME, _collection.count())
        except Exception:
            _collection = _chroma_client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Created new collection '%s'", COLLECTION_NAME)

        return _collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _parse_markdown_file(filepath: str) -> tuple[Optional[str], dict]:
    """Read a monster markdown file, extracting body text and YAML frontmatter metadata.

    Reads the file once. Returns ``(body_text, metadata_dict)``.
    ``body_text`` is ``None`` on read failure.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        logger.warning("Failed to read %s: %s", filepath, e)
        return None, {}

    parts = content.split("---", 2)
    if len(parts) >= 3:
        body = parts[2].strip()
        try:
            meta = yaml.safe_load(parts[1])
        except Exception:
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        return body, meta
    # No frontmatter -- use whole file
    return content.strip(), {}


def _extract_stock_code(filename: str) -> str:
    """Extract stock code from filename like '000062_深圳华强.md'."""
    base = os.path.splitext(os.path.basename(filename))[0]
    return base.split("_")[0] if "_" in base else base



def build_index(force: bool = False) -> int:
    """Build the ChromaDB index from all markdown reports.

    Always re-processes all files and rebuilds the collection.
    Idempotent-once behaviour is provided by ``ensure_index()``.

    Args:
        force: If True, delete existing collection before rebuilding.

    Returns:
        Number of documents indexed.
    """
    _ensure_chroma()
    _ensure_model()

    if not _chroma_available:
        logger.warning("ChromaDB not available -- cannot build index")
        return 0
    if not _st_available:
        logger.warning("sentence-transformers not available -- cannot build index")
        return 0

    global _collection

    with _chroma_lock:
        if force and _collection is not None:
            try:
                _chroma_client.delete_collection(name=COLLECTION_NAME)
            except Exception:
                pass
            _collection = None

    coll = _get_collection()
    if coll is None:
        return 0

    # Scan markdown files
    if not os.path.isdir(MONSTERS_DIR):
        logger.warning("Monsters directory not found: %s", MONSTERS_DIR)
        return 0

    md_files = sorted(
        os.path.join(MONSTERS_DIR, f)
        for f in os.listdir(MONSTERS_DIR)
        if f.endswith(".md")
    )

    documents = []
    ids = []
    metadatas = []

    for filepath in md_files:
        body, meta = _parse_markdown_file(filepath)
        if not body:
            continue

        stock_code = _extract_stock_code(filepath)

        documents.append(body)
        ids.append(stock_code)
        metadatas.append({
            "stock_code": stock_code,
            "stock_name": meta.get("stock_name", ""),
            "primary_type": meta.get("primary_type", ""),
            "sector": meta.get("sector", ""),
            "max_gain_pct": meta.get("max_gain_pct", 0),
            "trading_days": meta.get("trading_days", 0),
            "filepath": filepath,
        })

    if not documents:
        logger.warning("No markdown documents found to index")
        return 0

    # Generate embeddings
    embeddings = _model.encode(documents, show_progress_bar=False).tolist()

    # Delete existing docs with same IDs (upsert)
    for stock_code in ids:
        try:
            coll.delete(ids=[stock_code])
        except Exception:
            pass

    coll.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
    logger.info("Indexed %d monster reports into '%s'", len(documents), COLLECTION_NAME)
    return len(documents)


def query_similar(
    query_text: str,
    top_k: int = 5,
) -> list[dict]:
    """Query the monster report index for semantically similar reports.

    Args:
        query_text: Natural-language description of the candidate stock.
        top_k: Max number of results.

    Returns:
        List of dicts with keys: stock_code, stock_name, similarity, distance,
        primary_type, sector, max_gain_pct, trading_days, filepath.
    """
    _ensure_chroma()
    _ensure_model()

    if not _chroma_available or not _st_available:
        return []

    coll = _get_collection()
    if coll is None or coll.count() == 0:
        # Try building index on demand
        count = build_index()
        if count == 0:
            return []
        coll = _get_collection()

    if coll is None or coll.count() == 0:
        return []

    query_embedding = _model.encode([query_text], show_progress_bar=False).tolist()
    actual_k = min(top_k, coll.count())

    try:
        results = coll.query(query_embeddings=query_embedding, n_results=actual_k)
    except Exception as e:
        logger.warning("ChromaDB query failed: %s", e)
        return []

    matches = []
    if results and results.get("documents") and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            distance = results["distances"][0][i] if results.get("distances") else 1.0
            similarity = 1.0 - distance
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            matches.append({
                "stock_code": meta.get("stock_code", results["ids"][0][i]),
                "stock_name": meta.get("stock_name", ""),
                "similarity": round(similarity, 4),
                "distance": round(distance, 4),
                "primary_type": meta.get("primary_type", ""),
                "sector": meta.get("sector", ""),
                "max_gain_pct": meta.get("max_gain_pct", 0),
                "trading_days": meta.get("trading_days", 0),
                "filepath": meta.get("filepath", ""),
            })

    return matches


def build_candidate_query_text(features: CandidateFeatures) -> str:
    """Serialize CandidateFeatures into a Chinese query string.

    Args:
        features: CandidateFeatures dataclass instance.
    """
    parts = [
        f"候选股: {features.stock_name}({features.symbol})",
    ]
    if features.market_cap > 0:
        parts.append(f"市值{features.market_cap:.0f}亿")
    if features.sector:
        parts.append(f"板块{features.sector}")
    if features.turnover_pct > 0:
        parts.append(f"换手率{features.turnover_pct:.1f}%")
    if features.price > 0:
        parts.append(f"当前价格{features.price:.2f}元")
    if features.main_force_net != 0:
        direction = "流入" if features.main_force_net > 0 else "流出"
        parts.append(f"主力资金{direction}{abs(features.main_force_net):.0f}万")

    return "，".join(parts)


# ---------------------------------------------------------------------------
# Module init: trigger index build on first import in production
# ---------------------------------------------------------------------------

_initialized = False


def ensure_index() -> int:
    """Idempotent: ensure the index is built. Safe to call multiple times."""
    global _initialized
    if _initialized:
        coll = _get_collection()
        return coll.count() if coll else 0
    _initialized = True
    return build_index()
