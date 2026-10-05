# Groq → E2B code interpreter

Let a model on Groq write Python, then run that Python in a throwaway E2B
sandbox. Nothing executes on your machine, and the sandbox is gone when the
script exits.

```shell
pip install -r examples/requirements.txt

export GROQ_API_KEY=...   # https://console.groq.com/keys
export E2B_API_KEY=...    # https://e2b.dev/dashboard

python examples/groq_code_interpreter.py "Calculate how many r's are in the word 'strawberry.'"
```

```
Result: 3
```

## What it does, step by step

1. Sends your prompt to Groq with a system prompt asking for **code only**.
2. Unwraps the reply: models fence their code as ```` ```python ... ``` ```` even
   when told not to, so the fences are stripped in code rather than trusted.
3. Creates an E2B sandbox, runs the code there, and prints the last result, plus
   `stdout`/`stderr` if the code produced any.
4. Reports a Python error *inside* the sandbox as a failure of the run (exit
   code 4) — a raise in the sandbox is not a crash of this script.

## Flags

| Flag | Effect |
| --- | --- |
| `--dry-run` | Print the plan and exit. Needs no keys, calls nothing. |
| `--show-code` | Print the generated code before running it. |
| `--model ID` | Groq model id (default `openai/gpt-oss-120b`, or `$GROQ_MODEL`). |
| `--template NAME` | E2B template (default: the code-interpreter template). |
| `--timeout SECONDS` | Sandbox lifetime (default 120). |
| `--json` | Machine-readable output, including the code that ran. |

Exit codes: `0` success · `2` missing key or package · `3` the model call failed
(for example a retired model id) · `4` the generated code raised.

## Keys

Read from the environment, or from a `.env` file at the repo root, which is
gitignored:

```shell
cp .env.example .env      # then fill in the two values
python examples/groq_code_interpreter.py --dry-run   # shows which keys it can see
```

Never commit a key. `python3 tools/check_keys.py` reports which are configured
without printing any of them.

## Model ids move

Groq retires models on a schedule, and a snippet copied from a blog post will
often name one that no longer exists — `llama3-70b-8192` was retired, and on
**2026-08-16** Groq also retired `llama-3.1-8b-instant` and
`llama-3.3-70b-versatile`. Current replacements include `openai/gpt-oss-120b`
(quality) and `openai/gpt-oss-20b` (speed). Check what your key can use:

```shell
curl -s -H "Authorization: Bearer $GROQ_API_KEY" \
  https://api.groq.com/openai/v1/models | grep -o '"id": *"[^"]*"'
```

The script turns a decommissioned-model error into that hint instead of dumping
a traceback.

## Notes and limits

- **Neither API is reachable from a locked-down sandbox** (this repo's Arena
  sandbox only reaches GitHub and PyPI), so `--dry-run` and the offline test
  suite are the only things that run there. A real run needs a machine with
  egress to `api.groq.com` and `api.e2b.dev`, plus both keys.
- The sandbox has no access to your files: it only ever sees the code string.
- Cost is per call on both sides (Groq tokens, E2B sandbox-seconds).

## Tests

```shell
python3 examples/test_groq_code_interpreter.py     # 40 checks, all offline
```

The Groq half runs the **real `groq` SDK** against a local mock of the API, so
the request shape, auth header and reply parsing are genuinely exercised. The
E2B half is tested through an injected sandbox factory: that covers what we send
to `run_code`, how results and errors are read back, and the timeout/template
plumbing — but it cannot cover E2B's own service, which needs egress and a key.
