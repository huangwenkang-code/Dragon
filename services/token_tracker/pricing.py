"""Model pricing table — ¥ per 1K tokens.

DeepSeek V4-Pro pricing (promotional 2.5折, effective until 2026-05-31):
  Input:  ¥3/M  → ¥0.003/1K
  Output: ¥6/M  → ¥0.006/1K
Standard pricing (after promotion):
  Input:  ¥12/M → ¥0.012/1K
  Output: ¥24/M → ¥0.024/1K

DeepSeek V3 pricing:
  Input:  ¥1/M  → ¥0.001/1K
  Output: ¥2/M  → ¥0.002/1K
"""

PRICING = {
    # DeepSeek models — promotional pricing (current)
    "deepseek-v4-pro": {"input": 0.003, "output": 0.006},
    "deepseek-v4":     {"input": 0.003, "output": 0.006},
    "deepseek-v3":     {"input": 0.001, "output": 0.002},
    "deepseek-r1":     {"input": 0.004, "output": 0.016},
    "deepseek-chat":   {"input": 0.001, "output": 0.002},
    "deepseek-reasoner": {"input": 0.004, "output": 0.016},
    # Qwen models
    "qwen-turbo": {"input": 0.0003, "output": 0.0006},
    "qwen-plus":  {"input": 0.0008, "output": 0.002},
    "qwen-max":   {"input": 0.02, "output": 0.06},
    # Fallback — intentionally conservative (uses DeepSeek V3 rates)
    "default": {"input": 0.001, "output": 0.002},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in ¥ based on token counts and model pricing."""
    # Strip provider prefix if present (e.g. "deepseek/deepseek-v4-pro")
    if "/" in model:
        model = model.split("/")[-1]
    p = PRICING.get(model, PRICING["default"])
    cost = (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1000
    return round(cost, 6)
