#!/usr/bin/env python3
"""
Tests for groq_code_interpreter.py -- all offline.

The E2B half cannot run here (api.e2b.dev is not reachable from this sandbox and
there is no key), so it is tested through the injected sandbox factory: that
proves the code we send, the results we read back and the error handling, which
is the part we wrote. The Groq half runs against a local mock of the streaming
API, using the *real* groq SDK so our use of it is exercised.

    python3 examples/test_groq_code_interpreter.py
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import groq_code_interpreter as ci  # noqa: E402

KEY = "gsk_test_key_not_real"
FENCED = "Here you go:\n\n```python\nprint(sum(range(10)))\n```\n"
seen: list[dict] = []


class MockGroq(BaseHTTPRequestHandler):
    """Just enough of /openai/v1/chat/completions to exercise the real SDK."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        return

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        seen.append({"path": self.path, "auth": self.headers.get("Authorization"),
                     "payload": json.loads(body)})
        reply = json.dumps({
            "id": "chatcmpl-1", "object": "chat.completion", "created": 1,
            "model": seen[-1]["payload"].get("model", "m"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": FENCED}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)


# --------------------------------------------------------------------- fakes
@dataclasses.dataclass
class FakeResult:
    text: str


@dataclasses.dataclass
class FakeLogs:
    stdout: str = ""
    stderr: str = ""


@dataclasses.dataclass
class FakeError:
    name: str = "ValueError"
    value: str = "boom"
    traceback: str = "Traceback (most recent call last):\n  ValueError: boom"


@dataclasses.dataclass
class FakeExecution:
    results: list
    logs: FakeLogs
    error: object = None


class FakeSandbox:
    """Stands in for e2b_code_interpreter.Sandbox; records what it was asked to run."""

    def __init__(self, execution: FakeExecution, kwargs: dict):
        self.execution = execution
        self.kwargs = kwargs
        self.ran: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def run_code(self, code, **kwargs):
        self.ran.append(code)
        return self.execution


def sandbox_factory(execution: FakeExecution, record: list):
    def make(**kwargs):
        sb = FakeSandbox(execution, kwargs)
        record.append(sb)
        return sb
    return make


def check(label: str, got, want) -> bool:
    ok = got == want
    print(f"  {'OK  ' if ok else 'FAIL'} {label}")
    if not ok:
        print(f"        got : {got!r}\n        want: {want!r}")
    return ok


def run_cli(argv) -> tuple[int, str]:
    """Run the CLI, capturing stdout only (missing-key exits go to stderr)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = ci.main(argv)
    return code, buf.getvalue()


def run_cli_err(argv) -> tuple[int, str]:
    """Run the CLI and capture the message it prints on stderr before exiting."""
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            ci.main(argv)
        return 0, err.getvalue()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return code, err.getvalue()


def main() -> int:
    results: list[bool] = []
    os.environ.pop("GROQ_API_KEY", None)
    os.environ.pop("E2B_API_KEY", None)
    os.environ.pop("GROQ_BASE_URL", None)

    # 1. extraction ---------------------------------------------------------- #
    print("code extraction (the model does not always obey 'strip backticks')")
    results.append(check("fenced with python tag", ci.extract_code(FENCED),
                         "print(sum(range(10)))"))
    results.append(check("bare fence", ci.extract_code("```\nprint(1)\n```"), "print(1)"))
    results.append(check("py tag", ci.extract_code("```py\nx=1\n```"), "x=1"))
    results.append(check("prose then fence",
                         ci.extract_code("Sure!\n```python\nprint(2)\n```\nDone."),
                         "print(2)"))
    results.append(check("nothing fenced is passed through",
                         ci.extract_code("print(3)"), "print(3)"))
    results.append(check("last of several blocks wins",
                         ci.extract_code("```python\nfirst()\n```\n```python\nsecond()\n```"),
                         "second()"))
    results.append(check("empty stays empty", ci.extract_code("   "), ""))
    results.append(check("backticks survive inside a fence as-is",
                         ci.extract_code("```python\nd = {'a': 1}\n```"), "d = {'a': 1}"))

    # 2. missing keys -------------------------------------------------------- #
    print("\nmissing keys are caught before anything is called")
    code, msg = run_cli_err(["--show-code"])
    results.append(check("exit code 2", code, 2))
    results.append(check("names GROQ_API_KEY", "GROQ_API_KEY" in msg, True))
    results.append(check("says where to get one", "console.groq.com" in msg, True))

    # 3. dry run ------------------------------------------------------------- #
    print("\n--dry-run needs no keys and touches nothing")
    code, out = run_cli(["--dry-run"])
    results.append(check("exit code 0", code, 0))
    results.append(check("reports the missing keys honestly",
                         "MISSING" in out and "dry run" in out, True))
    code, out = run_cli(["--dry-run", "--json"])
    results.append(check("json plan parses", json.loads(out)["groq_api_key"], "MISSING"))

    # 4. real SDK against the mock ------------------------------------------ #
    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockGroq)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    print(f"\nthe real groq SDK, pointed at a mock ({base})")
    os.environ["GROQ_API_KEY"] = KEY
    os.environ["GROQ_BASE_URL"] = base
    try:
        reply = ci.generate_code("how many r's in strawberry?", "openai/gpt-oss-120b")
        results.append(check("sdk call returns the model's text", reply, FENCED))
        sent = seen[-1]
        results.append(check("auth header carries the key",
                             sent["auth"], f"Bearer {KEY}"))
        results.append(check("model id passed through",
                             sent["payload"]["model"], "openai/gpt-oss-120b"))
        results.append(check("system prompt sent",
                             sent["payload"]["messages"][0]["role"], "system"))
        results.append(check("user prompt sent",
                             sent["payload"]["messages"][1]["content"],
                             "how many r's in strawberry?"))
        results.append(check("deterministic by default",
                             sent["payload"]["temperature"], 0.0))
    except Exception as exc:  # pragma: no cover
        results.append(check(f"sdk call raised {exc!r}", "raised", "no error"))
    finally:
        os.environ.pop("GROQ_API_KEY")
        os.environ.pop("GROQ_BASE_URL")

    # 5. execution via the injected sandbox --------------------------------- #
    print("\nexecution (sandbox injected: we check what we send and read back)")
    ex = FakeExecution(results=[FakeResult("45")], logs=FakeLogs(stdout="45\n"))
    made: list[FakeSandbox] = []
    out = ci.execute_code("print(sum(range(10)))", timeout=99,
                          sandbox_factory=sandbox_factory(ex, made))
    results.append(check("ran the code we handed it", made[0].ran, ["print(sum(range(10)))"]))
    results.append(check("passed the timeout", made[0].kwargs["timeout"], 99))
    results.append(check("no template key unless asked", "template" in made[0].kwargs, False))
    results.append(check("sandbox was closed", made[0].closed, True))
    results.append(check("reports ok", out["ok"], True))
    results.append(check("reads the last result", out["result"], "45"))
    results.append(check("reads stdout", out["stdout"], "45\n"))

    made.clear()
    out = ci.execute_code("x", template="my-template",
                          sandbox_factory=sandbox_factory(ex, made))
    results.append(check("template forwarded when given",
                         made[0].kwargs["template"], "my-template"))

    made.clear()
    ex_err = FakeExecution(results=[], logs=FakeLogs(stderr="ValueError: boom\n"),
                           error=FakeError())
    out = ci.execute_code("boom()", sandbox_factory=sandbox_factory(ex_err, made))
    results.append(check("error is reported, not swallowed", out["ok"], False))
    results.append(check("error name captured", out["error"]["name"], "ValueError"))
    results.append(check("stderr captured", out["stderr"], "ValueError: boom\n"))

    # 6. end to end ---------------------------------------------------------- #
    print("\nend to end (mock model + fake sandbox, via the CLI)")
    os.environ["GROQ_API_KEY"] = KEY
    os.environ["E2B_API_KEY"] = "e2b_test_key_not_real"
    os.environ["GROQ_BASE_URL"] = base
    real_execute = ci.execute_code
    made.clear()
    ex_ok = FakeExecution(results=[FakeResult("45")], logs=FakeLogs())
    # note: the helper is bound to another name on purpose -- a lambda parameter
    # called sandbox_factory would shadow this function and call None
    fake_sandboxes = sandbox_factory(ex_ok, made)
    ci.execute_code = lambda code, template=None, timeout=120, sandbox_factory=None: \
        real_execute(code, template, timeout, sandbox_factory=fake_sandboxes)
    try:
        code, out = run_cli([])
        results.append(check("exit code 0", code, 0))
        results.append(check("prints the result", out.strip().endswith("Result: 45"), True))
        results.append(check("ran the extracted code, not the fence",
                             made[0].ran, ["print(sum(range(10)))"]))
        code, out = run_cli(["--json"])
        payload = json.loads(out)
        results.append(check("json carries code+result",
                             (payload["code"], payload["result"]),
                             ("print(sum(range(10)))", "45")))
        code, out = run_cli(["--show-code"])
        results.append(check("--show-code prints the code", "generated code:" in out, True))
    finally:
        ci.execute_code = real_execute
        for k in ("GROQ_API_KEY", "E2B_API_KEY", "GROQ_BASE_URL"):
            os.environ.pop(k, None)

    # 6b. local execution (no E2B, no key) ----------------------------------- #
    print("\n--local runs the cell here, no E2B key required")
    out = ci.execute_locally('word = "strawberry"\nprint(word.count("r"))')
    results.append(check("print-only cell: ok", out["ok"], True))
    results.append(check("print-only cell: last printed line is the result",
                         out["result"], "3"))
    results.append(check("stdout captured", out["stdout"].strip(), "3"))

    out = ci.execute_locally("6 * 7")
    results.append(check("trailing expression becomes the result", out["result"], "42"))

    out = ci.execute_locally("x = 1\nprint(x)\nx + 1")
    results.append(check("statements then expression", out["result"], "2"))

    out = ci.execute_locally("raise ValueError('nope')")
    results.append(check("a raise is a failed run, not a crash", out["ok"], False))
    results.append(check("error name captured", out["error"]["name"], "ValueError"))
    results.append(check("error value captured", out["error"]["value"], "nope"))

    out = ci.execute_locally("import time; time.sleep(5)", timeout=1)
    results.append(check("timeout is reported, not hung", out["ok"], False))
    results.append(check("timeout names itself", out["error"]["name"], "Timeout"))

    # 6c. a print-only cell on the E2B path has no notebook result ----------- #
    # Regression: execution.results is empty for a print-only cell, so the CLI
    # announced "Result: None" on the exact example in the docs.
    print("\nprint-only cell via the E2B path falls back to stdout")
    ci.execute_code = lambda code, template=None, timeout=120, sandbox_factory=None: {
        "ok": True, "result": None, "results": [], "stdout": "3\n", "stderr": "",
        "error": None}
    try:
        os.environ["GROQ_API_KEY"] = KEY
        os.environ["E2B_API_KEY"] = "e2b_test_key_not_real"
        os.environ["GROQ_BASE_URL"] = base
        code, out = run_cli([])
        results.append(check("exit code 0", code, 0))
        results.append(check("prints the printed value", "Result: 3" in out, True))
        results.append(check("never says None", "None" in out, False))
        results.append(check("does not duplicate the stdout block",
                             out.count("3"), 1))
    finally:
        ci.execute_code = real_execute
        for k in ("GROQ_API_KEY", "E2B_API_KEY", "GROQ_BASE_URL"):
            os.environ.pop(k, None)

    # 6d. --local needs only the Groq key ------------------------------------ #
    print("\n--local does not demand an E2B key")
    os.environ["GROQ_API_KEY"] = KEY
    os.environ["GROQ_BASE_URL"] = base
    try:
        code, out = run_cli(["--local", "--json"])
        payload = json.loads(out)
        results.append(check("exit code 0 without E2B_API_KEY", code, 0))
        results.append(check("reports it ran here", payload["ran_on"], "local"))
        # 45 is sum(range(10)) -- the real execution caught that the stub's
        # hardcoded 55 was simply the wrong answer
        results.append(check("really ran the extracted code", payload["result"], "45"))
    finally:
        for k in ("GROQ_API_KEY", "GROQ_BASE_URL"):
            os.environ.pop(k, None)

    # 7. missing E2B key is reported before the model is called -------------- #
    print("\nthe second key is checked before spending a request")
    os.environ["GROQ_API_KEY"] = KEY
    code, msg = run_cli_err([])
    results.append(check("exit code 2", code, 2))
    results.append(check("names E2B_API_KEY", "E2B_API_KEY" in msg, True))
    os.environ.pop("GROQ_API_KEY")

    # 8. retired model ids get an actionable message ------------------------- #
    print("\nretired model ids produce advice, not a stack trace")
    msg = ci.explain_api_error(Exception(
        "The model `llama3-70b-8192` has been decommissioned and is no longer "
        "supported."))
    results.append(check("mentions the models endpoint",
                         "api.groq.com/openai/v1/models" in msg, True))
    msg = ci.explain_api_error(Exception("401 invalid api key"))
    results.append(check("points at the console", "console.groq.com" in msg, True))

    srv.shutdown()
    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
