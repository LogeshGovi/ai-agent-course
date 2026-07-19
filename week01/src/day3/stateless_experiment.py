"""Day 3, task 2: the stateless experiment — the intellectual core of week 1.

"Memory" in a chat model is a client-side illusion built by resending the
entire conversation on every turn. Nothing is retained server-side between
requests. This script proves it three ways:

1. Two fresh, unrelated requests — the model has no memory across them.
2. Both messages sent in one `messages` array — now it "remembers", because
   the whole exchange was in the single request it just processed.
3. A 20-turn simulated conversation, resending full history every turn,
   showing input_tokens grow turn over turn (roughly quadratic total cost
   across the whole conversation, since turn N resends all N-1 prior turns).
"""

import httpx
from rich.console import Console
from rich.table import Table
from request_once import load_api_key, load_model

URL = "https://api.anthropic.com/v1/messages"
MODEL = load_model()  # small max_tokens — this is a token-count exercise, not a quality one
console = Console()


def send(messages: list, api_key: str, max_tokens: int = 30) -> dict:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    response = httpx.post(URL, headers=headers, json=body)
    result = response.json()
    if "error" in result:
        raise RuntimeError(f"API error ({response.status_code}): {result['error']['message']}")
    return result


def no_memory_demo(api_key: str):
    console.print("[bold]--- fresh requests, no shared history ---[/bold]")
    r1 = send([{"role": "user", "content": "My name is Ada."}], api_key)
    console.print(f"turn 1: {r1['content'][0]['text']!r}")

    r2 = send([{"role": "user", "content": "What's my name?"}], api_key)
    console.print(f"turn 2 (fresh request, no history): {r2['content'][0]['text']!r}\n")


def shared_history_demo(api_key: str):
    console.print("[bold]--- both messages sent in one array ---[/bold]")
    messages = [
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "Nice to meet you, Ada!"},
        {"role": "user", "content": "What's my name?"},
    ]
    result = send(messages, api_key)
    console.print(f"response: {result['content'][0]['text']!r}\n")


def quadratic_cost_demo(api_key: str, turns: int = 20):
    console.print(f"[bold]--- {turns}-turn conversation, resending full history each turn ---[/bold]")
    messages = []
    input_tokens_per_turn = []

    for i in range(turns):
        messages.append({"role": "user", "content": f"Fact {i}: remember that my favorite number is {i}."})
        result = send(messages, api_key)
        reply = result["content"][0]["text"]
        messages.append({"role": "assistant", "content": reply})
        input_tokens_per_turn.append(result["usage"]["input_tokens"])

    table = Table(title="input_tokens per turn (each turn resends everything before it)")
    table.add_column("turn")
    table.add_column("input_tokens", justify="right")
    table.add_column("bar")

    max_tokens = max(input_tokens_per_turn)
    for i, tokens in enumerate(input_tokens_per_turn, start=1):
        bar_len = int(tokens / max_tokens * 40)
        table.add_row(str(i), str(tokens), "█" * bar_len)

    console.print(table)
    total = sum(input_tokens_per_turn)
    console.print(
        f"\nturn 1 input_tokens: {input_tokens_per_turn[0]}, "
        f"turn {turns} input_tokens: {input_tokens_per_turn[-1]} "
        f"({input_tokens_per_turn[-1] / input_tokens_per_turn[0]:.1f}x growth)"
    )
    console.print(f"sum of input_tokens across all {turns} turns: {total}")
    console.print(
        "[dim]compare that sum to turns * input_tokens_per_turn[0] — the gap is the cost\n"
        "of resending history, and it's why cost grows faster than linearly with turn count.[/dim]"
    )


if __name__ == "__main__":
    api_key = load_api_key()
    no_memory_demo(api_key)
    shared_history_demo(api_key)
    quadratic_cost_demo(api_key)
