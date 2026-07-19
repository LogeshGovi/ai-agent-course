"""Shared pricing model, importable from any day's scripts via PYTHONPATH=week01/src.

Mirrors the PRICING/cost() shape built in day2/cost.py — that file stays as-is
(it's the day2 exercise deliverable); this is the reusable copy for day3/day4
and beyond, so those don't have to duplicate request_once.py-style across folders.
"""

# Price per 1,000,000 tokens, in USD. Update these if pricing changes.
PRICING = {
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},   # intro pricing through 2026-08-31
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}


def cost(usage: dict, model: str) -> float:
    if model not in PRICING:
        raise ValueError(f"No pricing entry for model {model!r} — add it to PRICING")

    prices = PRICING[model]
    input_cost = usage["input_tokens"] / 1_000_000 * prices["input"]
    output_cost = usage["output_tokens"] / 1_000_000 * prices["output"]
    return input_cost + output_cost
