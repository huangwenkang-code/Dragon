"""FinBERT inference for Chinese financial sentiment (CPU-friendly).

Model: yiyanghkust/finbert-tone-chinese (~110M params, BERT-base)
  - 3-class: negative / neutral / positive
  - CPU inference: ~200ms per text
  - Lazy-loaded on first call, stays in memory thereafter.
"""

from __future__ import annotations

from shared.utils.logging import get_logger

logger = get_logger(__name__)

_pipeline = None
_load_error: str | None = None
_model_name = "yiyanghkust/finbert-tone-chinese"

# Label mapping — handles both named labels and numeric labels
_LABEL_MAP: dict[str, str] = {}


def _load_model():
    """Lazy-load FinBERT pipeline. Called internally on first predict_sentiment()."""
    global _pipeline, _load_error

    if _pipeline is not None or _load_error is not None:
        return

    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

        logger.info("[finbert] loading model %s (CPU) ...", _model_name)
        tokenizer = AutoTokenizer.from_pretrained(_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(_model_name)

        _pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # CPU
            top_k=None,  # return all class probabilities
        )

        # Build label map from model config
        id2label = getattr(model.config, "id2label", {})
        for idx, label in id2label.items():
            label_lower = str(label).lower()
            _LABEL_MAP[str(idx)] = label_lower
            _LABEL_MAP[f"LABEL_{idx}"] = label_lower

        if not _LABEL_MAP:
            # Fallback: assume standard 3-class ordering
            _LABEL_MAP.update({"0": "neutral", "1": "positive", "2": "negative",
                               "LABEL_0": "neutral", "LABEL_1": "positive", "LABEL_2": "negative"})

        logger.info("[finbert] model loaded successfully (device=CPU)")

    except Exception as exc:
        _load_error = str(exc)
        _pipeline = None
        logger.warning("[finbert] failed to load model: %s", exc)


def predict_sentiment(text: str) -> dict | None:
    """Predict sentiment for a Chinese financial text.

    Returns dict with keys: label, positive, negative, neutral
    Returns None if model unavailable (caller should fall back to keyword scoring).
    """
    _load_model()

    if _pipeline is None:
        return None

    # Truncate to ~2000 chars (approx 512 BERT tokens for Chinese, ~3 chars/token)
    text = text[:2000] if len(text) > 2000 else text
    if not text.strip():
        return {"label": "neutral", "positive": 0.0, "negative": 0.0, "neutral": 1.0}

    try:
        results = _pipeline(text, truncation=True, max_length=512)
        # When top_k=None, results is [[{...}, {...}, {...}]] (batch of 1)
        # First [0] gets the first (and only) input's results
        batch_result = results[0] if isinstance(results[0], list) else results
        scores: dict[str, float] = {}
        for r in batch_result:
            raw_label = str(r["label"]).strip()
            mapped = _LABEL_MAP.get(raw_label, raw_label.lower())
            # Map all variations to our three labels
            if mapped in ("positive", "positive ", "lab_1", "label_1"):
                mapped = "positive"
            elif mapped in ("negative", "negative ", "lab_2", "label_2"):
                mapped = "negative"
            elif mapped in ("neutral", "neutral ", "lab_0", "label_0"):
                mapped = "neutral"
            scores[mapped] = r["score"]

        positive = scores.get("positive", 0.0)
        negative = scores.get("negative", 0.0)
        neutral = scores.get("neutral", 0.0)

        dominant = max(scores, key=scores.get) if scores else "neutral"

        return {
            "label": dominant,
            "positive": round(positive, 4),
            "negative": round(negative, 4),
            "neutral": round(neutral, 4),
        }

    except Exception as exc:
        logger.warning("[finbert] inference failed: %s", exc)
        return None


def is_available() -> bool:
    """Check if FinBERT model is loaded and ready."""
    _load_model()
    return _pipeline is not None
