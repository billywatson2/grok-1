#!/usr/bin/env python3
"""
Groq writes Python, E2B runs it.

Ask a model on Groq for code, then execute that code in a throwaway E2B sandbox
and print what it produced. Nothing runs on this machine, and nothing persists
in the sandbox afterwards.

    export GROQ_API_KEY=...     # https://console.groq.com/keys
    export E2B_API_KEY=...      # https://e2b.dev/dashboard
    python examples/groq_code_interpreter.py "Calculate how many r's are in strawberry"

Useful flags:

    --dry-run        print the plan without calling anything (needs no keys)
    --show-code      print the generated code before running it
    --model ID       any id from https://api.groq.com/openai/v1/models
    --json           machine-readable output
    --template NAME  E2B template (default: the code-interpreter template)

Keys are read from the environment first, then from a `.env` file at the repo
root (gitignored). They are never written to disk, logged, or put in the
printed output. Exit codes: 2 missing key, 3 generation failed, 4 code failed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_PROMPT = "Calculate how many r's are in the word 'strawberry.'"

SYSTEM_PROMPT = (
    "You are a helpful assistant that can execute python code in a Jupyter notebook. "
    "Only respond with the code to be executed and nothing else. "
    "Strip backticks in code blocks."
)

# ```python ... ```  /  ```py ... ```  /  ``` ... ```  -- models wrap code
# anyway, whatever the system prompt says, so unwrap it here instead of trusting
FENCE = re.compile(r"```(?:python|py|python3)?[ \t]*\r?\n(.*?)(?:\r?\n)?```", re.S)


# --------------------------------------------------------------------------- #
# keys
def load_env_file(path: Path | None = None) -> None:
    """Read KEY=VALUE lines into os.environ; real environment variables win."""
    path = path or (HERE.parent / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


def require(name: str, where: str, purpose: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(
            f"error: {name} is not set ({purpose}).\n"
            f"  get one: {where}\n"
            f"  then:    export {name}=...\n"
            f"  or put {name}=... in a .env file at {HERE.parent / '.env'} "
            f"(gitignored, never committed)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return value


def missing_package(package: str) -> None:
    print(f"error: {package} is not installed.\n"
          f"  pip install -r {HERE / 'requirements.txt'}", file=sys.stderr)
    raise SystemExit(2)


# --------------------------------------------------------------------------- #
# generation
def extract_code(text: str) -> str:
    """Pull the code out of a model reply, however it chose to wrap it."""
    text = (text or "").strip()
    blocks = FENCE.findall(text)
    if blocks:
        # the last block is what runs in the notebook-cell pattern
        return blocks[-1].strip()
    return text


def generate_code(prompt: str, model: str, api_key: str | None = None,
                  client=None, temperature: float = 0.0) -> str:
    """Ask Groq to write the code. `client` is injectable so tests can mock it."""
    if client is None:
        try:
            from groq import Groq
        except ImportError:
            missing_package("groq")
        key = api_key or require("GROQ_API_KEY", "https://console.groq.com/keys",
                                 "needed to ask the model for code")
        # GROQ_BASE_URL is only useful to point at a proxy or a test mock
        client = Groq(api_key=key, base_url=os.environ.get("GROQ_BASE_URL") or None)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
    )
    text = response.choices[0].message.content or ""
    return text


# --------------------------------------------------------------------------- #
# execution
def execute_code(code: str, template: str | None = None, timeout: int = 120,
                 sandbox_factory=None) -> dict:
    """Run `code` in an E2B sandbox; return what happened. Nothing is kept."""
    if sandbox_factory is None:
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError:
            missing_package("e2b-code-interpreter")
        sandbox_factory = Sandbox.create

    kwargs: dict = {"timeout": timeout}
    if template:
        kwargs["template"] = template

    with sandbox_factory(**kwargs) as sandbox:
        execution = sandbox.run_code(code)

    results = [getattr(r, "text", None) for r in (execution.results or [])]
    logs = getattr(execution, "logs", None)
    error = getattr(execution, "error", None)
    return {
        "ok": error is None,
        "result": next((r for r in reversed(results) if r), None),
        "results": [r for r in results if r],
        "stdout": (getattr(logs, "stdout", "") or ""),
        "stderr": (getattr(logs, "stderr", "") or ""),
        "error": None if error is None else {
            "name": getattr(error, "name", "Error"),
            "value": getattr(error, "value", str(error)),
            "traceback": getattr(error, "traceback", ""),
        },
    }


# --------------------------------------------------------------------------- #
# cli
def explain_api_error(exc: Exception) -> str:
    """Turn SDK failures into something actionable, especially retirements.

    Groq retires model ids on a schedule (llama-3.1-8b-instant and
    llama-3.3-70b-versatile went away on 2026-08-16), and a snippet copied from
    a blog post will happily name one of them.
    """
    text = str(exc)
    low = text.lower()
    if "decommission" in low or "model_not_found" in low or "does not exist" in low:
        return (f"{text}\n  That model id is not available on this account. "
                f"List current ids with:\n"
                f"    curl -s -H \"Authorization: Bearer $GROQ_API_KEY\" "
                f"https://api.groq.com/openai/v1/models | "
                f"python3 -m json.tool | grep '\"id\"'")
    if "401" in text or "invalid api key" in low or "authentication" in low:
        return f"{text}\n  Check GROQ_API_KEY (https://console.groq.com/keys)."
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate Python with Groq, execute it in an E2B sandbox.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT,
                    help=f"what the code should do (default: {DEFAULT_PROMPT!r})")
    ap.add_argument("--model", default=os.environ.get("GROQ_MODEL") or DEFAULT_MODEL,
                    help=f"Groq model id (default: {DEFAULT_MODEL}; $GROQ_MODEL)")
    ap.add_argument("--template", default=os.environ.get("E2B_TEMPLATE"),
                    help="E2B template (default: the code-interpreter template)")
    ap.add_argument("--timeout", type=int, default=120,
                    help="sandbox lifetime in seconds (default: 120)")
    ap.add_argument("--show-code", action="store_true",
                    help="print the generated code before running it")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would happen, without calling either API")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="print a JSON object instead of prose")
    args = ap.parse_args(argv)

    load_env_file()

    if args.dry_run:
        plan = {
            "model": args.model,
            "template": args.template or "(default code-interpreter template)",
            "timeout": args.timeout,
            "prompt": args.prompt,
            "groq_api_key": "set" if os.environ.get("GROQ_API_KEY") else "MISSING",
            "e2b_api_key": "set" if os.environ.get("E2B_API_KEY") else "MISSING",
        }
        if args.as_json:
            print(json.dumps(plan, indent=2))
        else:
            print("dry run -- nothing was called:")
            for k, v in plan.items():
                print(f"  {k:14} {v}")
            print("\n  the code the model writes is executed in a throwaway E2B\n"
                  "  sandbox; nothing runs on this machine and nothing persists.")
        return 0

    # both keys are required for a real run; check before spending a request
    require("GROQ_API_KEY", "https://console.groq.com/keys", "needed to write the code")
    require("E2B_API_KEY", "https://e2b.dev/dashboard", "needed to run the code")

    try:
        raw = generate_code(args.prompt, args.model)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"error: the model call failed.\n  {explain_api_error(exc)}", file=sys.stderr)
        return 3

    code = extract_code(raw)
    if not code:
        print("error: the model returned no code.", file=sys.stderr)
        return 3

    if args.show_code and not args.as_json:
        print("generated code:\n")
        print("\n".join("  " + ln for ln in code.splitlines()))
        print()

    try:
        out = execute_code(code, args.template, args.timeout)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"error: the sandbox failed to run the code.\n  {exc}", file=sys.stderr)
        return 4

    out.update({"model": args.model, "prompt": args.prompt, "code": code})

    if args.as_json:
        print(json.dumps(out, indent=2))
        return 0 if out["ok"] else 4

    if out["stdout"].strip() and out["stdout"].strip() != (out["result"] or "").strip():
        print("stdout:\n" + out["stdout"].rstrip())
    if out["stderr"].strip():
        print("stderr:\n" + out["stderr"].rstrip())
    if out["ok"]:
        print(f"Result: {out['result']}")
    else:
        err = out["error"]
        print(f"the code raised {err['name']}: {err['value']}", file=sys.stderr)
        if err["traceback"]:
            print("\n".join("  " + ln for ln in err["traceback"].splitlines()),
                  file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
