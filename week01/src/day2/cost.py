import httpx
import json
from request_once import load_api_key, load_model

MESSAGES_URL = "https://api.anthropic.com/v1/messages"

# Price per 1,000,000 tokens, in USD. Update these if pricing changes.
PRICING = {
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},   # intro pricing through 2026-08-31
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
}


def cost(usage: dict, model: str) -> float:
    if model not in PRICING:
        raise ValueError(f"No pricing entry for model {model!r} — add it to PRICING")

    prices = PRICING[model]
    input_cost = usage["input_tokens"] / 1_000_000 * prices["input"]
    output_cost = usage["output_tokens"] / 1_000_000 * prices["output"]
    return input_cost + output_cost


def send(body: dict, api_key: str) -> dict:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    response = httpx.post(MESSAGES_URL, headers=headers, json=body)
    result = response.json()
    if "error" in result:
        raise RuntimeError(f"API error ({response.status_code}): {result['error']['message']}")
    return result


if __name__ == "__main__":
    api_key = load_api_key()

    model = load_model()
    body = {
        "model": model,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Explain what a token is in one sentence."}],
    }

    result = send(body, api_key)
    usage = result["usage"]
    request_cost = cost(usage, model)

    print(f"response: {result['content'][0]['text']!r}")
    print(f"usage: {usage}")
    print(f"cost: ${request_cost:.6f}")
