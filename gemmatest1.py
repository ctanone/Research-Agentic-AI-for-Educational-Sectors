"""
Ollama Chat with Memory Management
- ~100MB memory budget for conversation history
- Auto-compresses long conversations via summarization
- Dumps compressed/full history to JSON on exit
"""

import sys
import json
import gzip
import pickle
import datetime
import os
from pathlib import Path
from ollama import chat

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MEMORY_BUDGET_BYTES   = 100 * 1024 * 1024   # 100 MB
COMPRESS_THRESHOLD    = 0.75                 # compress when 75% full
MIN_TURNS_BEFORE_COMP = 6                    # need at least N turns before compressing
SUMMARY_KEEP_RECENT   = 4                    # keep last N turn-pairs verbatim after compress
DUMP_DIR              = Path(".")            # where .json / .gz dumps go
MODEL                 = "lumen:latest"
SUMMARY_MODEL         = "lumen:latest"       # model used to summarize (can differ)


# ─── MEMORY TRACKER ────────────────────────────────────────────────────────────
def messages_bytes(messages: list[dict]) -> int:
    """Rough byte estimate for a list of message dicts."""
    return sum(len(m.get("role", "").encode()) + len(m.get("content", "").encode())
               for m in messages)


def memory_status(messages: list[dict]) -> str:
    used = messages_bytes(messages)
    pct  = used / MEMORY_BUDGET_BYTES * 100
    bar  = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
    return f"[MEM {bar} {pct:5.1f}%  {used/1024:.1f}KB / {MEMORY_BUDGET_BYTES//1024//1024}MB]"


# ─── COMPRESSION ───────────────────────────────────────────────────────────────
def compress_history(messages: list[dict], summary_so_far: str) -> tuple[list[dict], str]:
    """
    Summarize all-but-last SUMMARY_KEEP_RECENT pairs, prepend summary as system
    message, keep recent turns verbatim.
    Returns (new_messages, updated_summary_text).
    """
    # Separate system seed (if any) from conversation turns
    system_msgs = [m for m in messages if m["role"] == "system"]
    convo_msgs  = [m for m in messages if m["role"] != "system"]

    # Turns to summarize = everything except the last N pairs
    keep_count   = SUMMARY_KEEP_RECENT * 2          # each pair = user + assistant
    to_summarize = convo_msgs[:-keep_count] if len(convo_msgs) > keep_count else []
    recent_turns = convo_msgs[-keep_count:]

    if not to_summarize:
        return messages, summary_so_far  # nothing to compress yet

    # Build summarization prompt
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in to_summarize
    )
    prior = f"Previous summary:\n{summary_so_far}\n\n" if summary_so_far else ""
    summary_prompt = (
        f"{prior}Summarize the following conversation concisely, "
        f"preserving key facts, decisions, and context:\n\n{history_text}"
    )

    print("\n  ⟳  Compressing conversation history...", end="", flush=True)
    try:
        resp = chat(model=SUMMARY_MODEL, messages=[
            {"role": "system", "content": "You are a concise summarizer. Return only the summary."},
            {"role": "user",   "content": summary_prompt},
        ])
        new_summary = resp["message"]["content"].strip()
        print(" done.\n")
    except Exception as e:
        print(f" failed ({e}). Keeping history as-is.\n")
        return messages, summary_so_far

    # Rebuild messages: system messages + summary injection + recent turns
    summary_injection = {
        "role":    "system",
        "content": f"[Conversation summary so far]\n{new_summary}",
    }
    new_messages = system_msgs + [summary_injection] + recent_turns
    return new_messages, new_summary


# ─── DUMP ──────────────────────────────────────────────────────────────────────
def dump_session(messages: list[dict], summary: str, compressed: bool = False):
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base     = DUMP_DIR / f"chat_session_{ts}"
    payload  = {
        "timestamp":    ts,
        "summary":      summary,
        "messages":     messages,
        "total_bytes":  messages_bytes(messages),
    }

    # Always write plain JSON
    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # Also write gzip'd pickle for compact archival
    gz_path = base.with_suffix(".pkl.gz")
    with gzip.open(gz_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n  💾  Session saved:")
    print(f"      JSON  → {json_path}  ({json_path.stat().st_size/1024:.1f} KB)")
    print(f"      GZ    → {gz_path}  ({gz_path.stat().st_size/1024:.1f} KB)")


# ─── MAIN CHAT LOOP ────────────────────────────────────────────────────────────
def main():
    messages:      list[dict] = []
    summary_so_far: str       = ""
    compression_count: int    = 0

    print("╔══════════════════════════════════════════════════════╗")
    print("║  Ollama Chat  ·  Memory-managed  ·  100 MB budget    ║")
    print("║  Type 'that's all' / 'exit' / 'quit' to end          ║")
    print("║  Type '/mem'  to see memory usage                    ║")
    print("║  Type '/dump' to manually save session               ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            user_input = "exit"

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if user_input.lower() == "/mem":
            print(f"  {memory_status(messages)}\n")
            continue

        if user_input.lower() == "/dump":
            dump_session(messages, summary_so_far)
            continue

        exit_phrases = {"that's all", "thats all", "that's all.", "exit", "quit", "bye"}
        if user_input.lower() in exit_phrases:
            dump_session(messages, summary_so_far)
            print("\nGoodbye! Have a great day! 👋")
            break

        # ── Append user message ───────────────────────────────────────────────
        messages.append({"role": "user", "content": user_input})

        # ── Memory check → compress if needed ─────────────────────────────────
        used = messages_bytes(messages)
        if (used >= MEMORY_BUDGET_BYTES * COMPRESS_THRESHOLD
                and len(messages) >= MIN_TURNS_BEFORE_COMP):
            messages, summary_so_far = compress_history(messages, summary_so_far)
            compression_count += 1

        # Hard cap: if still over budget after compression, drop oldest non-system turns
        while messages_bytes(messages) > MEMORY_BUDGET_BYTES:
            non_sys = [i for i, m in enumerate(messages) if m["role"] != "system"]
            if len(non_sys) < 2:
                break
            messages.pop(non_sys[0])   # drop oldest user turn
            if non_sys[1] < len(messages):
                messages.pop(non_sys[1] - 1)  # drop its assistant reply

        # ── Call Ollama ────────────────────────────────────────────────────────
        try:
            response = chat(model=MODEL, messages=messages)
            assistant_msg = response["message"]["content"]
        except Exception as e:
            print(f"\n  ⚠  Ollama error: {e}\n")
            messages.pop()  # remove the user turn we just added
            continue

        messages.append({"role": "assistant", "content": assistant_msg})

        # ── Print response + memory bar ───────────────────────────────────────
        print(f"\nAssistant: {assistant_msg}\n")
        print(f"  {memory_status(messages)}"
              + (f"  [compressed ×{compression_count}]" if compression_count else "")
              + "\n")


if __name__ == "__main__":
    main()