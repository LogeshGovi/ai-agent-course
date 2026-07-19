import httpx
import json
from request_once import load_api_key

MESSAGES_URL = "https://api.anthropic.com/v1/messages"

CHARS_PER_TOKEN = 2  # rough English-text heuristic


def estimate_tokens(messages: list, system: str | None = None) -> int:
    total_chars = len(system) if system else 0
    for message in messages:
        total_chars += len(message["content"])
    return total_chars // CHARS_PER_TOKEN


def send_and_get_usage(messages: list, system: str | None, api_key: str) -> dict:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": "claude-sonnet-5",
        "max_tokens": 50,
        "messages": messages,
    }
    if system:
        body["system"] = system

    response = httpx.post(MESSAGES_URL, headers=headers, json=body)
    result = response.json()
    if "error" in result:
        raise RuntimeError(f"API error ({response.status_code}): {result['error']['message']}")
    return result["usage"]


def compare(label: str, messages: list, system: str | None, api_key: str):
    estimated = estimate_tokens(messages, system)
    usage = send_and_get_usage(messages, system, api_key)
    actual = usage["input_tokens"]
    error_pct = (estimated - actual) / actual * 100

    print(f"--- {label} ---")
    print(f"  estimated: {estimated}")
    print(f"  actual (usage.input_tokens): {actual}")
    print(f"  error: {error_pct:+.1f}%")
    print()


if __name__ == "__main__":
    api_key = load_api_key()

    compare(
        "single short message, no system prompt",
        messages=[{"role": "user", "content": "What's the capital of France? and how to get there from Frankfurt?"}],
        system=None,
        api_key=api_key,
    )

    compare(
        "single message with a system prompt",
        messages=[{"role": "user", "content": "What's the capital of France?"}],
        system="You are a helpful, concise geography assistant.",
        api_key=api_key,
    )

    compare(
        "multi-turn conversation with system prompt",
        messages=[
            {"role": "user", "content": "My name is Ada."},
            {"role": "assistant", "content": "Nice to meet you, Ada!"},
            {"role": "user", "content": "What's my name?"},
        ],
        system="You are a helpful, concise assistant.",
        api_key=api_key,
    )
