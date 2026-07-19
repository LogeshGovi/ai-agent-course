"""Day 3, task 1: raw SSE streaming — no SDK stream=True helper, we parse the wire format ourselves.

Wire format: each event arrives as an "event: <type>" line followed by a
"data: <json>" line, separated by blank lines. In practice the JSON payload's
own "type" field always matches the event name, so we only need to read the
"data: " lines and dispatch on payload["type"] — no need to track the
separate "event:" line.
"""

import httpx
import json
from rich.console import Console
from request_once import load_api_key, load_model

URL = "https://api.anthropic.com/v1/messages"
MODEL = load_model()
console = Console()


def stream_message(body: dict, api_key: str) -> tuple[str, dict, str|None]:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {**body, "stream": True}

    accumulated_text = ""
    usage = {"input_tokens": 0, "output_tokens": 0}
    stop_reason = None

    try:
        with httpx.stream("POST", URL, headers=headers, json=body, timeout=60.0) as response:
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue

                payload = json.loads(line[len("data: "):])
                event_type = payload["type"]

                if event_type == "message_start":
                    # usage arrives partially here (input_tokens is final,
                    # output_tokens is not — that only lands at message_delta)
                    usage["input_tokens"] = payload["message"]["usage"]["input_tokens"]

                elif event_type == "content_block_delta":
                    delta = payload["delta"]
                    if delta["type"] == "text_delta":
                        chunk = delta["text"]
                        accumulated_text += chunk
                        console.print(chunk, end="", style="green")

                elif event_type == "message_delta":
                    # usage.output_tokens is only complete/final on this event
                    usage["output_tokens"] = payload["usage"]["output_tokens"]
                    stop_reason = payload["delta"].get("stop_reason")

                elif event_type == "message_stop":
                    console.print()  # newline once the terminal event arrives
                    break

    except httpx.ReadError as e:
        # Simulates what you'd see if you killed your wifi mid-stream:
        # httpx raises here, and everything streamed so far is still in
        # accumulated_text — but the response was never completed, so usage
        # (especially output_tokens) may be incomplete or missing entirely.
        console.print(f"\n[red]stream disconnected mid-response: {e}[/red]")
        console.print(f"[yellow]partial text received before disconnect:[/yellow] {accumulated_text!r}")

    return accumulated_text, usage, stop_reason


if __name__ == "__main__":
    api_key = load_api_key()
    body = {
        "model": MODEL,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": "Write a 4-sentence story about a lighthouse keeper."}],
    }

    text, usage, stop_reason = stream_message(body, api_key)

    console.print(f"\n[bold]stop_reason:[/bold] {stop_reason}")
    console.print(f"[bold]usage:[/bold] {usage}")
