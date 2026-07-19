"""Day 3, task 3: Conversation — the reusable class this week builds toward.

Holds message history, streams responses (same raw-SSE approach as
streaming.py, reimplemented here as part of turning that exploration into a
clean reusable component), and tracks running token/cost totals per turn and
per session. This is the piece the daily plan says "becomes the agent's
message loop" in week 2, and gets wrapped by day4/chat.py.
"""

import httpx
import json
from rich.console import Console
from request_once import load_api_key, load_model
from pricing import cost

URL = "https://api.anthropic.com/v1/messages"


class Conversation:
    def __init__(self, api_key: str, model: str | None = None, system: str | None = None, max_tokens: int = 1024):
        model = model or load_model()
        self.api_key = api_key
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self.messages: list[dict] = []

        self.last_turn_usage: dict = {"input_tokens": 0, "output_tokens": 0}
        self.last_turn_cost: float = 0.0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cost = 0.0

        self.console = Console()

    def send(self, text: str, stream_output: bool = True) -> str:
        self.messages.append({"role": "user", "content": text})

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self.messages,
            "stream": True,
        }
        if self.system:
            body["system"] = self.system

        accumulated_text = ""
        usage = {"input_tokens": 0, "output_tokens": 0}

        with httpx.stream("POST", URL, headers=headers, json=body, timeout=60.0) as response:
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue

                payload = json.loads(line[len("data: "):])
                event_type = payload["type"]

                if event_type == "message_start":
                    usage["input_tokens"] = payload["message"]["usage"]["input_tokens"]

                elif event_type == "content_block_delta":
                    delta = payload["delta"]
                    if delta["type"] == "text_delta":
                        chunk = delta["text"]
                        accumulated_text += chunk
                        if stream_output:
                            self.console.print(chunk, end="")

                elif event_type == "message_delta":
                    usage["output_tokens"] = payload["usage"]["output_tokens"]

                elif event_type == "message_stop":
                    if stream_output:
                        self.console.print()
                    break

        self.messages.append({"role": "assistant", "content": accumulated_text})

        self.last_turn_usage = usage
        self.last_turn_cost = cost(usage, self.model)
        self.session_input_tokens += usage["input_tokens"]
        self.session_output_tokens += usage["output_tokens"]
        self.session_cost += self.last_turn_cost

        return accumulated_text


if __name__ == "__main__":
    api_key = load_api_key()
    convo = Conversation(api_key, system="You are a helpful, concise assistant.")

    convo.send("My name is Ada.")
    convo.send("What's my name?")

    print(f"\nsession input tokens: {convo.session_input_tokens}")
    print(f"session output tokens: {convo.session_output_tokens}")
    print(f"session cost: ${convo.session_cost:.6f}")
