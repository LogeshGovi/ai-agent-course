import httpx
import json
from request_once import load_api_key, load_model

URL = "https://api.anthropic.com/v1/messages"
MODEL = load_model()


def send_raw(body: dict, api_key: str) -> httpx.Response:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    return httpx.post(URL, headers=headers, json=body)


def show(response: httpx.Response):
    print(f"  status: {response.status_code}")
    print(f"  body: {json.dumps(response.json(), indent=2)}")


def bad_api_key_experiment():
    print("\n--- bad API key (expect 401) ---")
    body = {
        "model": MODEL,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = send_raw(body, "sk-ant-not-a-real-key")
    show(response)


def nonexistent_model_experiment(api_key: str):
    print("\n--- nonexistent model (expect 404) ---")
    body = {
        "model": "claude-does-not-exist",
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = send_raw(body, api_key)
    show(response)


def malformed_body_experiment(api_key: str):
    print("\n--- missing max_tokens (expect 400) ---")
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = send_raw(body, api_key)
    show(response)

    print("\n--- empty messages array (expect 400) ---")
    body = {
        "model": MODEL,
        "max_tokens": 20,
        "messages": [],
    }
    response = send_raw(body, api_key)
    show(response)


def max_tokens_too_high_experiment(api_key: str):
    print("\n--- max_tokens above model cap (expect 400) ---")
    body = {
        "model": MODEL,
        "max_tokens": 99999999,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = send_raw(body, api_key)
    show(response)


def rate_limit_experiment(api_key: str):
    print("\n--- hammering the API until 429 ---")
    body = {
        "model": MODEL,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    for i in range(50):
        response = send_raw(body, api_key)
        print(f"  request {i + 1}: status {response.status_code}")
        if response.status_code == 429:
            print(f"  retry-after header: {response.headers.get('retry-after')}")
            show(response)
            break
    else:
        print("  never hit a 429 in 50 requests — try raising the range")


if __name__ == "__main__":
    api_key = load_api_key()
    bad_api_key_experiment()
    nonexistent_model_experiment(api_key)
    malformed_body_experiment(api_key)
    max_tokens_too_high_experiment(api_key)
    rate_limit_experiment(api_key)
