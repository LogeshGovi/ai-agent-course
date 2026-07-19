import os
import httpx
import json
from dotenv import load_dotenv


def load_api_key() -> str:
    load_dotenv()
    ant_key = os.environ["ANTHROPIC_API_KEY"]
    return ant_key


def main():
    api_key = load_api_key()
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Say hello in one sentence."}
        ],
    }
    response = httpx.post(url, headers=headers, json=body)
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
