import httpx
import json
from request_once import load_api_key

URL = "https://api.anthropic.com/v1/messages"


def send(body: dict, api_key: str) -> dict:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    response = httpx.post(URL, headers=headers, json=body)
    result = response.json()
    if "error" in result:
        raise RuntimeError(f"API error ({response.status_code}): {result['error']['message']}")
    return result


def temperature_experiment(api_key: str):
    prompt = "Give me one random word."
    for temperature in [0, 0.5, 1.0]:
        print(f"\n--- temperature={temperature} ---")
        for i in range(5):
            body = {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 20,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            result = send(body, api_key)
            text = result["content"][0]["text"]
            print(f"  run {i + 1}: {text!r}")


def max_tokens_truncation_experiment(api_key: str):
    print("\n--- max_tokens=10 truncation ---")
    body = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Explain photosynthesis in detail."}],
    }
    result = send(body, api_key)
    print(f"  stop_reason: {result['stop_reason']}")
    print(f"  content: {result['content'][0]['text']!r}")


def stop_sequences_experiment(api_key: str):
    print("\n--- stop_sequences ---")
    body = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 100,
        "stop_sequences": ["3."],
        "messages": [{"role": "user", "content": "List five colors, numbered 1. 2. 3. 4. 5."}],
    }
    result = send(body, api_key)
    print(f"  stop_reason: {result['stop_reason']}")
    print(f"  content: {result['content'][0]['text']!r}")


if __name__ == "__main__":
    api_key = load_api_key()
    temperature_experiment(api_key)
    max_tokens_truncation_experiment(api_key)
    stop_sequences_experiment(api_key)
