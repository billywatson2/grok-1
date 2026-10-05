#!/usr/bin/env python3
"""
Report which optional API keys this checkout can see, without ever printing one.

    python3 tools/check_keys.py

Exits 0 when at least one key is configured, 1 when none are -- handy in a
start-up script. Only lengths are printed, never values.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"

KEYS = [
    ("GROQ_API_KEY", "arena contestant 'groq-llama'; playground --backend groq",
     "export GROQ_API_KEY=...  (console.groq.com/keys)"),
    ("E2B_API_KEY", "examples/groq_code_interpreter.py (runs generated code)",
     "export E2B_API_KEY=...  (e2b.dev/dashboard)"),
    ("ARENA_TOKEN", "locks the arena when it is reachable from other devices",
     "export ARENA_TOKEN=..."),
]


def load_env_file() -> None:
    """Same rules as the servers: existing environment wins, values never logged."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    load_env_file()
    src = f"(.env found at {ENV_FILE.relative_to(ROOT)})" if ENV_FILE.exists() \
        else "(no .env file -- using the environment only)"
    print(f"key status {src}\n")
    found = 0
    for name, purpose, hint in KEYS:
        value = os.environ.get(name)
        if value:
            found += 1
            print(f"  [set]     {name:<14} {len(value)} chars   used by: {purpose}")
        else:
            print(f"  [missing] {name:<14} {'':<11} used by: {purpose}")
            print(f"            how to set: {hint}")
    print()
    if found:
        print(f"{found} key(s) configured. Values are never printed by this tool.")
        return 0
    print("No keys configured. Everything still works locally -- the arena simply "
          "shows the cloud contestant as not ready, and --backend groq is unavailable.")
    print("See .env.example for the format.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
