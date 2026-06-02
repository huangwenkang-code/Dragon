# Monster Matcher L3 + 妖股信号打分器 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ChromaDB text-embedding similarity layer (L3) to monster matcher and replace zero-valued ml_sub with rules-based monster signal scorer.

**Architecture:** Two independent NEW modules (`embedding_store.py`, `signal_scorer.py`), MODIFY `matcher.py` to fuse L2+L3, MODIFY `generate_candidates.py` to wire signal_scorer into ml_sub and pass L3 into monster matching.

**Tech Stack:** sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2), ChromaDB (persistent, local), existing SQLAlchemy + asyncio

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `services/monster_matcher/embedding_store.py` | CREATE | ChromaDB + sentence-transformers singleton |
| `services/monster_matcher/signal_scorer.py` | CREATE | Rules-based 5-dim monster signal scorer |
| `services/monster_matcher/matcher.py` | MODIFY | Add L3 optional integration to find_similar() |
| `services/graph_service/nodes/generate_candidates.py` | MODIFY | Wire signal_scorer → ml_sub, pass L3 to matcher |
| `requirements.txt` | MODIFY | Add sentence-transformers |

---

### Task 1: Create `embedding_store.py` — ChromaDB text embedding store

**Files:**
- Create: `services/monster_matcher/embedding_store.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add sentence-transformers to requirements.txt**

Read `requirements.txt` first, then append:

```
sentence-transformers>=3.0.0
```

- [ ] **Step 2: Install the dependency**

Run: `source venv/Scripts/activate && pip install sentence-transformers`
Expected: installs sentence-transformers + torch + transformers + the model will be downloaded on first use

- [ ] **Step 3: Write the embedding_store.py module**

```python
"""Monster report text-embedding store backed by ChromaDB + sentence-transformers.

Provides semantic similarity search over historical monster markdown reports.
Lazy-loads the embedding model and ChromaDB collection on first use.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

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

            _chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=CHROMA_PERSIST_DIR,
                anonymized_telemetry=False,
                is_persistent=True,
            ))
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


def _read_markdown_body(filepath: str) -> Optional[str]:
    """Read a monster markdown file, stripping YAML frontmatter.

    Returns the body text after the second '---', or None on failure.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        logger.warning("Failed to read %s: %s", filepath, e)
        return None

    # Split on YAML frontmatter delimiters
    parts = content.split("---", 2)
    if len(parts) >= 3:
        return parts[2].strip()
    # No frontmatter — use whole file
    return content.strip()


def _extract_stock_code(filename: str) -> str:
    """Extract stock code from filename like '000062_深圳华强.md'."""
    base = os.path.splitext(os.path.basename(filename))[0]
    return base.split("_")[0] if "_" in base else base


def _extract_metadata(filepath: str) -> dict:
    """Extract key metadata from the YAML frontmatter of a monster report."""
    import re
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return {}

    meta = {}
    # Extract known YAML keys via regex (avoid PyYAML dependency for simplicity)
    patterns = {
        "stock_name": r"stock_name:\s*\"(.+?)\"",
        "primary_type": r"primary_type:\s*\"(.+?)\"",
        "sector": r"sector:\s*\"(.+?)\"",
        "max_gain_pct": r"max_gain_pct:\s*(\d+)",
        "trading_days": r"trading_days:\s*(\d+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, content)
        if m:
            val = m.group(1)
            if key in ("max_gain_pct", "trading_days"):
                val = int(val)
            meta[key] = val
    return meta


def build_index(force: bool = False) -> int:
    """Build/rebuild the ChromaDB index from all markdown reports.

    Args:
        force: If True, delete existing collection and rebuild from scratch.

    Returns:
        Number of documents indexed.
    """
    _ensure_chroma()
    _ensure_model()

    if not _chroma_available:
        logger.warning("ChromaDB not available — cannot build index")
        return 0
    if not _st_available:
        logger.warning("sentence-transformers not available — cannot build index")
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

    if coll.count() > 0 and not force:
        logger.info("Collection already has %d docs, skipping build", coll.count())
        return coll.count()

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
        body = _read_markdown_body(filepath)
        if not body:
            continue

        stock_code = _extract_stock_code(filepath)
        meta = _extract_metadata(filepath)

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

    # Remove existing docs with same IDs (upsert)
    existing_ids = set(ids)
    try:
        existing = coll.get()
        for existing_id in existing.get("ids", []):
            if existing_id in existing_ids and existing_id not in ids:
                pass  # keep non-conflicting
    except Exception:
        pass

    # Delete then add (simpler upsert for ChromaDB)
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


def build_candidate_query_text(features) -> str:
    """Serialize CandidateFeatures into a Chinese query string.

    Args:
        features: CandidateFeatures dataclass instance.
    """
    from services.monster_matcher.matcher import CandidateFeatures
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
```

- [ ] **Step 4: Test import and index build**

```bash
cd D:/K/dragon-engine && source venv/Scripts/activate && python -c "
from services.monster_matcher.embedding_store import build_index, query_similar, build_candidate_query_text
count = build_index(force=True)
print(f'Indexed {count} reports')
# Test query
results = query_similar('候选股: 测试股(000001), 市值55亿, 板块消费电子, 换手率4.2%, 当前价格12.5元')
print(f'Got {len(results)} results')
for r in results:
    print(f'  {r[\"stock_name\"]} ({r[\"stock_code\"]}) sim={r[\"similarity\"]:.3f} type={r[\"primary_type\"]}')
"
```

Expected: downloads model (~420MB, one-time), builds index with 11 docs, query returns results.

---

### Task 2: Create `signal_scorer.py` — rules-based monster signal scorer

**Files:**
- Create: `services/monster_matcher/signal_scorer.py`

- [ ] **Step 1: Write the signal_scorer module**

```python
"""Rules-based monster signal scorer.

Transparent 5-dimension scoring that replaces the zero-valued ml_sub.
No training — uses known monster stock patterns: small cap, trader
participation, sentiment/hype anomalies, concept heat, limit-up streaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.utils.logging import get_logger

logger = get_logger(__name__)

# Speculative keywords for limit-up streak estimation
SPEC_KW = {"连板", "龙头", "妖股", "打板", "接力", "涨停", "热点"}

# Default weights (adjusted: turnover 0.20 redistributed to other 5 dims)
DEFAULT_WEIGHTS = {
    "small_cap": 0.24,
    "trader_participation": 0.24,
    "sentiment_anomaly": 0.19,
    "concept_heat": 0.19,
    "limit_up_streak": 0.14,
}


@dataclass
class SignalBreakdown:
    """Per-dimension scores for auditability."""
    small_cap: float = 0.0
    trader_participation: float = 0.0
    sentiment_anomaly: float = 0.0
    concept_heat: float = 0.0
    limit_up_streak: float = 0.0

    def as_dict(self) -> dict:
        return {
            "small_cap": round(self.small_cap, 3),
            "trader_participation": round(self.trader_participation, 3),
            "sentiment_anomaly": round(self.sentiment_anomaly, 3),
            "concept_heat": round(self.concept_heat, 3),
            "limit_up_streak": round(self.limit_up_streak, 3),
        }


def score_stock(
    market_cap: float = 0.0,
    famous_traders: list[str] | None = None,
    trader_signal: str = "",
    lhb_score: float = 0.0,
    avg_sentiment: float = 0.0,
    avg_hype: float = 0.0,
    matched_concepts: list[str] | None = None,
    sentiment_keywords: list[str] | None = None,
    event_keywords: list[str] | None = None,
) -> tuple[float, SignalBreakdown]:
    """Compute the monster signal score (0-1) for a single stock.

    Args:
        market_cap: Market cap in 亿.
        famous_traders: List of famous trader names from dragon tiger board.
        trader_signal: Trader signal string (e.g. "合力做多").
        lhb_score: Raw dragon tiger score.
        avg_sentiment: Average sentiment score from FinBERT.
        avg_hype: Average hype score.
        matched_concepts: List of concept names matched for this stock.
        sentiment_keywords: Keywords from sentiment analysis.
        event_keywords: Keywords from related events.

    Returns:
        (total_score, SignalBreakdown) — both 0-1.
    """
    breakdown = SignalBreakdown()
    available = set()

    # --- Small Cap ---
    if market_cap > 0:
        if market_cap < 50:
            breakdown.small_cap = 1.0
        elif market_cap < 100:
            breakdown.small_cap = 0.7
        elif market_cap < 200:
            breakdown.small_cap = 0.3
        else:
            breakdown.small_cap = 0.0
        available.add("small_cap")

    # --- Trader Participation ---
    traders = famous_traders or []
    if traders:
        breakdown.trader_participation = 1.0
        available.add("trader_participation")
    elif lhb_score > 0:
        # On dragon tiger board but no famous trader
        breakdown.trader_participation = 0.5
        available.add("trader_participation")
    else:
        # Not on dragon tiger board at all — this dimension is missing
        breakdown.trader_participation = 0.0

    # --- Sentiment Anomaly ---
    if avg_sentiment > 0.8 and avg_hype > 0.3:
        breakdown.sentiment_anomaly = 1.0
    elif avg_sentiment > 0.5:
        breakdown.sentiment_anomaly = 0.5
    else:
        breakdown.sentiment_anomaly = 0.2  # baseline, not zero — even flat sentiment is a data point
    available.add("sentiment_anomaly")

    # --- Concept Heat ---
    concepts = matched_concepts or []
    n = len(concepts)
    if n >= 3:
        breakdown.concept_heat = 1.0
    elif n == 2:
        breakdown.concept_heat = 0.7
    elif n == 1:
        breakdown.concept_heat = 0.4
    else:
        breakdown.concept_heat = 0.0
    available.add("concept_heat")

    # --- Limit-Up Streak ---
    all_keywords = set(sentiment_keywords or []) | set(event_keywords or [])
    kw_matches = len(all_keywords & SPEC_KW)
    breakdown.limit_up_streak = min(kw_matches / 3.0, 1.0)
    if kw_matches > 0:
        available.add("limit_up_streak")

    # Redistribute weights: if a dimension is unavailable (0 data), drop its
    # weight and re-normalize the rest.
    active_weights = {
        dim: DEFAULT_WEIGHTS[dim]
        for dim in available
        if dim in DEFAULT_WEIGHTS
    }
    total_weight = sum(active_weights.values()) if active_weights else 1.0

    if total_weight == 0:
        return 0.0, breakdown

    score = sum(
        getattr(breakdown, dim) * (active_weights.get(dim, 0) / total_weight)
        for dim in active_weights
    )
    score = round(max(0.0, min(1.0, score)), 4)

    return score, breakdown
```

- [ ] **Step 2: Test the scorer standalone**

```bash
cd D:/K/dragon-engine && source venv/Scripts/activate && python -c "
from services.monster_matcher.signal_scorer import score_stock

# Test: ideal monster signal
score, bd = score_stock(
    market_cap=35.0,
    famous_traders=['宁波桑田路'],
    trader_signal='合力做多',
    lhb_score=0.8,
    avg_sentiment=0.9,
    avg_hype=0.5,
    matched_concepts=['华为汽车', '消费电子', '新能源'],
    sentiment_keywords=['连板', '龙头', '涨停'],
)
print(f'Ideal monster: score={score}')
print(f'  breakdown: {bd.as_dict()}')
assert score > 0.7, f'Expected >0.7, got {score}'

# Test: boring large-cap
score2, bd2 = score_stock(
    market_cap=500.0,
    famous_traders=[],
    trader_signal='',
    lhb_score=0.0,
    avg_sentiment=0.3,
    avg_hype=0.1,
    matched_concepts=[],
)
print(f'Boring large-cap: score={score2}')
print(f'  breakdown: {bd2.as_dict()}')
assert score2 < 0.3, f'Expected <0.3, got {score2}'

# Test: mid-cap with some signals
score3, bd3 = score_stock(
    market_cap=80.0,
    famous_traders=[],
    trader_signal='',
    lhb_score=0.6,
    avg_sentiment=0.7,
    avg_hype=0.4,
    matched_concepts=['光伏'],
    sentiment_keywords=['热点'],
)
print(f'Mid-cap partial: score={score3}')
print(f'  breakdown: {bd3.as_dict()}')

print('All tests passed')
"
```

Expected: ideal >0.7, boring <0.3, mid-cap in between. All assertions pass.

---

### Task 3: Modify `matcher.py` — integrate L3 into find_similar()

**Files:**
- Modify: `services/monster_matcher/matcher.py`

- [ ] **Step 1: Read current matcher.py**

Already read. Key locations:
- `find_similar()` method at line 99-228
- L2 scoring loop at line 116-224

- [ ] **Step 2: Modify find_similar() to accept and use L3**

Add `use_l3: bool = True` parameter to `find_similar()`. After the L2 scoring loop (after line 224, before the sort), insert L3 query and fusion logic.

In the method signature (line 99-104), change:

```python
async def find_similar(
    self,
    features: CandidateFeatures,
    top_k: int = 5,
    filter_types: Optional[list[str]] = None,
) -> list[MatchResult]:
```

To:

```python
async def find_similar(
    self,
    features: CandidateFeatures,
    top_k: int = 5,
    filter_types: Optional[list[str]] = None,
    use_l3: bool = True,
) -> list[MatchResult]:
```

After the L2 scoring loop (after line 224, before the sort at line 227), insert:

```python
            # ---- L3: ChromaDB text embedding similarity ----
            l3_scores: dict[int, float] = {}  # monster id → text_similarity
            if use_l3:
                try:
                    from services.monster_matcher.embedding_store import (
                        build_candidate_query_text,
                        ensure_index,
                        query_similar,
                    )
                    ensure_index()
                    query_text = build_candidate_query_text(features)
                    l3_results = query_similar(query_text, top_k=top_k * 3)
                    for l3r in l3_results:
                        # Match back to monster by stock_code
                        for mi, mr in enumerate(results):
                            if mr.monster.stock_code == l3r["stock_code"]:
                                l3_scores[mi] = l3r["similarity"]
                                break
                except Exception as e:
                    logger.warning("[MonsterMatcher] L3 skipped: %s", e)

            # Fuse L2 + L3: 0.5 * structured + 0.5 * text_similarity
            for mi, mr in enumerate(results):
                text_sim = l3_scores.get(mi, 0.0)
                if text_sim > 0:
                    mr.similarity_score = 0.5 * mr.similarity_score + 0.5 * text_sim
                    mr.feature_scores["text_semantic"] = text_sim
                    if text_sim > 0.6:
                        mr.match_reasons.insert(0, "语义高度相似")
```

Also add `import logging` if not already present, and add the logger:

```python
logger = logging.getLogger(__name__)
```

Wait — check the current imports. The file uses `from shared.utils.logging import get_logger` — actually it doesn't currently. Let me check... no, it doesn't import logger currently. Add at the top after imports:

```python
from shared.utils.logging import get_logger
logger = get_logger(__name__)
```

- [ ] **Step 3: Verify matcher.py has no syntax errors**

```bash
cd D:/K/dragon-engine && source venv/Scripts/activate && python -c "from services.monster_matcher.matcher import MonsterMatcher, CandidateFeatures, MatchResult; print('Import OK')"
```

---

### Task 4: Modify `generate_candidates.py` — wire signal_scorer and L3

**Files:**
- Modify: `services/graph_service/nodes/generate_candidates.py`

- [ ] **Step 1: Read current generate_candidates.py key sections**

Already read. Key locations:
- `_compute_scores()` at line 232-415 — `ml_score = model_scores.get(sym, 0.0)` at line 333
- `_enrich_with_monster_reference()` at line 109-168 — calls `matcher.find_similar(features, top_k=3)` at line 140
- `_try_model_ensemble()` at line 418-453

- [ ] **Step 2: Replace ml_score in _compute_scores() with signal_scorer call**

Replace line 333:
```python
ml_score = model_scores.get(sym, 0.0)
```

With:
```python
# Monster signal scorer (rules-based, replaces ML ensemble)
try:
    from services.monster_matcher.signal_scorer import score_stock as _score_monster
    ml_score, _signal_bd = _score_monster(
        market_cap=float(flow.get("market_cap", 0)),
        famous_traders=lhb.get("famous_traders", []),
        trader_signal=lhb.get("trader_signal", ""),
        lhb_score=lhb_score,
        avg_sentiment=avg_sent,
        avg_hype=avg_hype,
        matched_concepts=list(matched_concepts),
        sentiment_keywords=list(keywords),
        event_keywords=[],  # event keywords not indexed per-symbol yet
    )
except Exception:
    ml_score = 0.0
```

- [ ] **Step 3: Enable L3 in _enrich_with_monster_reference()**

Change line 140:
```python
matches = await matcher.find_similar(features, top_k=3)
```

To:
```python
matches = await matcher.find_similar(features, top_k=3, use_l3=True)
```

- [ ] **Step 4: Mark _try_model_ensemble as deprecated, remove from main flow**

In `generate_candidates()` (around line 37), remove or comment out the call to `_try_model_ensemble()`:

```python
# -- Optional ML model ensemble prediction (DEPRECATED: models untrained) --
# model_scores = _try_model_ensemble(state)
model_scores: dict[str, float] = {}
```

Keep the `_try_model_ensemble()` function definition (don't delete it), but it will no longer be called in normal flow.

- [ ] **Step 5: Verify no syntax errors**

```bash
cd D:/K/dragon-engine && source venv/Scripts/activate && python -c "from services.graph_service.nodes.generate_candidates import generate_candidates; print('Import OK')"
```

---

### Task 5: End-to-end test — run pipeline and verify outputs

- [ ] **Step 1: Clean restart the server**

```bash
taskkill /F /IM python.exe 2>/dev/null; sleep 2
cd D:/K/dragon-engine && source venv/Scripts/activate && python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 &
```

Wait for startup (~15s).

- [ ] **Step 2: Trigger a pipeline run**

```bash
curl -s -X POST http://localhost:8000/run -H "Content-Type: application/json" -d '{}' | python -m json.tool | head -20
```

Wait for pipeline completion (~5 min).

- [ ] **Step 3: Export sample response**

```bash
cd D:/K/dragon-engine && source venv/Scripts/activate && python scripts/export_sample_response.py
```

- [ ] **Step 4: Verify the outputs**

```bash
cd D:/K/dragon-engine && source venv/Scripts/activate && python -c "
import json
with open('dragon-engine-web/public/pipeline_result_sample.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Check ml_sub is non-zero
for c in data['leader_candidates'][:5]:
    print(f'{c[\"stock_name\"]} ({c[\"stock_code\"]}): leader={c[\"leader_score\"]:.3f} ml_sub={c[\"ml_sub\"]:.3f}')

print()
# Check monster_reference has improved similarity
for c in data['leader_candidates'][:3]:
    mr = c.get('monster_reference')
    if mr:
        print(f'{c[\"stock_name\"]}:')
        print(f'  summary: {mr.get(\"summary\", \"\")[:120]}')
        for m in mr.get('top_matches', []):
            print(f'  - {m[\"stock_name\"]}: sim={m[\"similarity\"]:.3f} reasons={m.get(\"match_reasons\", [])}')
    print()
"
```

Expected:
- `ml_sub` values vary across candidates (not all 0 or all identical)
- `monster_reference` top matches have similarity >0.20 (ideally >0.30)
- Some match_reasons include "语义高度相似"
- Summary uses Chinese type labels

- [ ] **Step 5: Verify ChromaDB persistence**

```bash
ls -la D:/K/dragon-engine/data_lake/chroma_db/
```

Expected: directory exists with ChromaDB data files.

- [ ] **Step 6: Verify second run uses cached index**

Trigger another pipeline run or run a quick test:

```bash
cd D:/K/dragon-engine && source venv/Scripts/activate && python -c "
from services.monster_matcher.embedding_store import build_index, query_similar
# Should NOT rebuild (count > 0 already)
count = build_index(force=False)
print(f'Index count (should be 11): {count}')
# Force rebuild
count2 = build_index(force=True)
print(f'After force rebuild: {count2}')
"
```
