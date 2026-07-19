"""Day 4: CLI chatbot with a context strategy.

Ships two strategies for handling conversations that would otherwise exceed
the context window, selectable with --strategy:

  sliding  — drop oldest turns until the (estimated) history fits. Fast,
             free, forgets hard. No extra API call.
  compact  — once the (estimated) history crosses ~70% of --window, ask the
             model to summarize the oldest half into one paragraph and
             replace those turns with the summary. Costs a call, preserves
             gist, loses detail.

--window defaults to a deliberately small number of tokens (not anything
close to a real model's actual context window) so the strategy is actually
exercised in a normal conversation instead of requiring hundreds of turns.

Stretch goal from the daily plan (token-aware truncation that never splits a
tool-call pair or assistant/user boundary) is not implemented — this chatbot
has no tools, so there's nothing to preserve a boundary around beyond the
user/assistant pairing the compact/sliding logic already respects.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

from rich.console import Console

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "day3"))

from request_once import load_api_key  # noqa: E402
from conversation import Conversation  # noqa: E402

console = Console()

DAY4_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(DAY4_DIR, "logs")
TRANSCRIPT_DIR = os.path.join(DAY4_DIR, "transcripts")
SYSTEM_PROMPT_PATH = os.path.join(DAY4_DIR, "system_prompt.txt")

CHARS_PER_TOKEN = 4  # cheap heuristic for window-management decisions only —
# real cost/usage accounting always comes from Conversation's real API usage.


def estimate_tokens_chars(messages: list, system: str | None) -> int:
    total_chars = len(system) if system else 0
    for m in messages:
        total_chars += len(m["content"])
    return total_chars // CHARS_PER_TOKEN


def load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_PATH) as f:
        return f.read().strip()


def log_turn(log_path: str, user_text: str, response_text: str, usage: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request": {"role": "user", "content": user_text},
        "response": {"role": "assistant", "content": response_text},
        "usage": usage,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def apply_sliding_window(convo: Conversation, window: int):
    dropped_pairs = 0
    while len(convo.messages) > 2 and estimate_tokens_chars(convo.messages, convo.system) > window:
        del convo.messages[:2]
        dropped_pairs += 1
    if dropped_pairs:
        console.print(f"[yellow]sliding-window: dropped {dropped_pairs} oldest turn(s)[/yellow]")


def apply_compact(convo: Conversation, window: int, api_key: str):
    threshold = window * 0.7
    if len(convo.messages) < 4 or estimate_tokens_chars(convo.messages, convo.system) < threshold:
        return

    half = len(convo.messages) // 2
    half -= half % 2  # keep the cut on a user/assistant pair boundary
    if half < 2:
        return

    oldest = convo.messages[:half]
    excerpt = "\n".join(f"{m['role']}: {m['content']}" for m in oldest)

    console.print("[cyan]compact: summarizing oldest half of the conversation...[/cyan]")
    summarizer = Conversation(api_key, model=convo.model)
    summary = summarizer.send(
        "Summarize the following conversation excerpt into a single concise "
        f"paragraph, preserving key facts and names:\n\n{excerpt}",
        stream_output=False,
    )

    # fold the summarizer's own usage/cost into the main session's totals
    convo.session_input_tokens += summarizer.session_input_tokens
    convo.session_output_tokens += summarizer.session_output_tokens
    convo.session_cost += summarizer.session_cost

    del convo.messages[:half]
    convo.messages.insert(0, {"role": "user", "content": f"[Earlier conversation summary]: {summary}"})
    convo.messages.insert(1, {"role": "assistant", "content": "Understood, I have that context."})

    console.print(f"[cyan]compact: replaced {half} messages with a summary[/cyan]")


def print_status(convo: Conversation, window: int):
    tokens_used = estimate_tokens_chars(convo.messages, convo.system)
    console.print(
        f"[dim]tokens: {tokens_used}/{window} | "
        f"turn cost: ${convo.last_turn_cost:.6f} | "
        f"session cost: ${convo.session_cost:.6f}[/dim]"
    )


def save_transcript(convo: Conversation) -> str:
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    path = os.path.join(TRANSCRIPT_DIR, f"transcript_{int(time.time())}.json")
    with open(path, "w") as f:
        json.dump(
            {"model": convo.model, "system": convo.system, "messages": convo.messages},
            f,
            indent=2,
        )
    return path


def load_transcript(convo: Conversation, path: str):
    with open(path) as f:
        data = json.load(f)
    convo.messages = data["messages"]
    if data.get("system"):
        convo.system = data["system"]


def handle_command(user_input: str, convo: Conversation, args) -> bool:
    """Returns True if the REPL loop should quit."""
    command, *rest = user_input[1:].split(maxsplit=1)
    arg = rest[0] if rest else None

    if command == "quit":
        return True

    elif command == "reset":
        convo.messages = []
        convo.session_input_tokens = 0
        convo.session_output_tokens = 0
        convo.session_cost = 0.0
        console.print("[green]conversation reset[/green]")

    elif command == "tokens":
        tokens_used = estimate_tokens_chars(convo.messages, convo.system)
        console.print(
            f"estimated tokens in history: {tokens_used}/{args.window}\n"
            f"session totals — input: {convo.session_input_tokens}, "
            f"output: {convo.session_output_tokens}, cost: ${convo.session_cost:.6f}"
        )

    elif command == "save":
        path = save_transcript(convo)
        console.print(f"[green]saved to {path}[/green]")

    elif command == "load":
        if not arg:
            console.print("[red]usage: /load <path>[/red]")
        else:
            load_transcript(convo, arg)
            console.print(f"[green]loaded {arg}[/green]")

    else:
        console.print(f"[red]unknown command: /{command}[/red] (try /reset /tokens /save /load /quit)")

    return False


def main():
    parser = argparse.ArgumentParser(description="Week 1 Day 4 CLI chatbot")
    parser.add_argument("--strategy", choices=["sliding", "compact"], default="sliding")
    parser.add_argument(
        "--window", type=int, default=3000,
        help="Simulated context window in tokens (small on purpose, to actually exercise the strategy)",
    )
    parser.add_argument("--model", default="claude-sonnet-5")
    args = parser.parse_args()

    api_key = load_api_key()
    system = load_system_prompt()
    convo = Conversation(api_key, model=args.model, system=system)

    session_id = int(time.time())
    log_path = os.path.join(LOG_DIR, f"session_{session_id}.jsonl")

    console.print(f"[bold]week1 chatbot[/bold] — strategy={args.strategy}, window={args.window}, model={args.model}")
    console.print("commands: /reset /tokens /save /load <path> /quit\n")

    while True:
        try:
            user_input = console.input("[bold cyan]you> [/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input.strip():
            continue

        if user_input.startswith("/"):
            if handle_command(user_input, convo, args):
                break
            continue

        if args.strategy == "sliding":
            apply_sliding_window(convo, args.window)
        else:
            apply_compact(convo, args.window, api_key)

        console.print("[bold magenta]claude> [/bold magenta]", end="")
        response_text = convo.send(user_input)

        log_turn(log_path, user_input, response_text, convo.last_turn_usage)
        print_status(convo, args.window)


if __name__ == "__main__":
    main()
