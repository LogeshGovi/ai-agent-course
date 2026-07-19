import httpx
import json
import uuid
from request_once import load_api_key, load_model
import re

COUNT_URL = "https://api.anthropic.com/v1/messages/count_tokens"
MESSAGES_URL = "https://api.anthropic.com/v1/messages"
MODEL = load_model()

def normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t\n\r\f\v]+", " ", text)

def count_tokens(text: str, api_key: str) -> int:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
    }
    response = httpx.post(COUNT_URL, headers=headers, json=body)
    result = response.json()
    if "error" in result:
        raise RuntimeError(f"API error ({response.status_code}): {result['error']['message']}")
    return result["input_tokens"]


PYTHON_FUNCTION = '''def add(a, b):
    return a + b
'''


def run_samples(api_key: str):
    samples = {
        "plain English sentence": "The quick brown fox jumps over the lazy dog.",
        "same sentence in German": "Der schnelle braune Fuchs springt über den faulen Hund.",
        "a UUID": str(uuid.uuid4()),
        "a Python function": PYTHON_FUNCTION,
        "the word strawberry": "strawberry",
        "an emoji": "🍓",
        "500 spaces": " " * 500 + "x",
    }

    for label, text in samples.items():
        tokens = count_tokens(text, api_key)
        words = len(text.split())
        chars = len(text)
        non_ws_chars = len(normalize_whitespace(text))

        print(f"{label}:")
        print(f"  words: {words}, chars: {chars}, normalized (dedup ws) chars: {non_ws_chars}, tokens: {tokens}")
        print(f"  tokens/word:        {tokens / words:.3f}" if words else "  tokens/word:        n/a")
        print(f"  tokens/char:        {tokens / chars:.3f}")
        print(f"  tokens/norm_char:   {tokens / non_ws_chars:.3f}")
        print()


def strawberry_question(api_key: str):
    print("--- asking the model to count letters ---")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": MODEL,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "How many r's are in the word strawberry?"}],
    }
    response = httpx.post(MESSAGES_URL, headers=headers, json=body)
    result = response.json()
    print(f"  model answer: {result['content'][0]['text']!r}")
    print(f"  input_tokens for this request: {result['usage']['input_tokens']}")


if __name__ == "__main__":
    api_key = load_api_key()
    run_samples(api_key)
    strawberry_question(api_key)
