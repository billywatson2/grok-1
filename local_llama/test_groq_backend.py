#!/usr/bin/env python3
"""
Test the Groq code paths against a local mock of Groq's API.

api.groq.com is not reachable from a locked-down sandbox, so the honest way to
test an API client is against a mock that speaks the same wire format: this
starts a throwaway HTTP server that emits OpenAI-style SSE, points both clients
at it, and checks what they sent and what they assembled.

    python3 local_llama/test_groq_backend.py

Covers:
  * GroqBackend (the playground's --backend groq) -- streaming and auth header
  * arena.openai_stream (the arena's 'groq-llama' contestant) -- chat + completions
  * a missing key produces a useful message rather than a stack trace
  * GROQ_MODEL / model_env overrides the model id
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                    # local_llama/
sys.path.insert(0, str(HERE.parent / "lm_arena"))  # arena.py

from groq_backend import GroqBackend, MissingKey  # noqa: E402

KEY = "gsk_test_key_do_not_use"
WORDS = ["Once", " upon", " a", " time", " there", " was", " a", " robot", "."]
seen: list[dict] = []


class MockGroq(BaseHTTPRequestHandler):
    """Speaks enough of the OpenAI streaming protocol to exercise both clients."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):        # keep the test output clean
        return

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        payload = json.loads(body)
        auth = self.headers.get("Authorization")
        seen.append({"path": self.path, "auth": auth, "payload": payload})

        if auth != f"Bearer {KEY}":
            err = json.dumps({"error": {"message": "invalid api key"}}).encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(piece: bytes):
            self.wfile.write(b"%x\r\n%s\r\n" % (len(piece), piece))

        chat = self.path.endswith("/chat/completions")
        for word in WORDS:
            delta = {"content": word} if chat else None
            chunk = {"choices": [{"delta": delta} if chat else {"text": word}]}
            send(f"data: {json.dumps(chunk)}\n\n".encode())
        # non-content chunks both clients must tolerate
        send(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
        send(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")


def start_mock() -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockGroq)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def check(label: str, got, want) -> bool:
    ok = got == want
    print(f"  {'OK  ' if ok else 'FAIL'} {label}")
    if not ok:
        print(f"        got : {got!r}\n        want: {want!r}")
    return ok


def main() -> int:
    srv, base = start_mock()
    want = "".join(WORDS)
    results: list[bool] = []
    print(f"mock Groq listening on {base}\n")

    # 1. playground backend -------------------------------------------------- #
    print("GroqBackend (playground --backend groq)")
    be = GroqBackend(base_url=base, api_key=KEY)
    text = "".join(be.stream("hi", 32, 0.8, 0.95, 0, 7))
    results.append(check("streams the assembled text", text, want))
    results.append(check("reports the model", be.info()["model"], "openai/gpt-oss-120b"))
    results.append(check("info hides the key", be.info()["key"], "set"))
    sent = seen[-1]
    results.append(check("posts to /chat/completions", sent["path"], "/chat/completions"))
    results.append(check("sends bearer auth", sent["auth"], f"Bearer {KEY}"))
    results.append(check("asks for a stream", sent["payload"]["stream"], True))
    results.append(check("passes seed through", sent["payload"]["seed"], 7))

    # 2. arena contestant path (chat) ---------------------------------------- #
    print("\narena openai_stream (chat)")
    import arena  # noqa: E402

    text = "".join(arena.openai_stream(base, "chat", "llama-3.3-70b-versatile",
                                      "hi", 32, 0.8, 0.95, 7, api_key=KEY))
    results.append(check("streams the assembled text", text, want))
    results.append(check("uses the registry's model",
                         seen[-1]["payload"]["model"], "llama-3.3-70b-versatile"))

    # 3. arena contestant path (legacy completions) -------------------------- #
    print("\narena openai_stream (completions)")
    text = "".join(arena.openai_stream(base, "completions", "m", "hi", 32, 0.8, 0.95,
                                      7, api_key=KEY))
    results.append(check("streams the assembled text", text, want))
    results.append(check("posts to /completions", seen[-1]["path"], "/completions"))

    # 4. bad key surfaces the provider's reason ------------------------------ #
    print("\nthe provider's own error message reaches the user")
    try:
        "".join(GroqBackend(base_url=base, api_key="wrong").stream("hi", 8, 0.8, 0.9, 0, 1))
        results.append(check("wrong key raises", "no error", "an error"))
    except RuntimeError as exc:
        results.append(check("wrong key raises with HTTP 401",
                             "401" in str(exc) and "invalid api key" in str(exc), True))

    # 5. missing key gives a hint, not a traceback --------------------------- #
    print("\nmissing key")
    os.environ.pop("GROQ_API_KEY", None)
    be = GroqBackend(base_url=base)          # no api_key argument at all
    results.append(check("info reports it is missing", be.info()["key"], "missing"))
    try:
        "".join(be.stream("hi", 8, 0.8, 0.9, 0, 1))
        results.append(check("stream raises MissingKey", "no error", "MissingKey"))
    except MissingKey as exc:
        results.append(check("MissingKey explains GROQ_API_KEY",
                             "GROQ_API_KEY" in str(exc) and "console.groq.com" in str(exc),
                             True))

    # 6. env overrides the model id (the model_env feature) ------------------ #
    print("\nGROQ_MODEL override")
    os.environ["GROQ_MODEL"] = "openai/gpt-oss-120b"
    from arena import RemoteModel  # noqa: E402

    row = json.loads((HERE.parent / "lm_arena" / "models.json").read_text())
    entry = next(e for e in row["models"] if e["name"] == "groq-llama")
    m = RemoteModel(entry["name"], entry["config"])
    results.append(check("model_env wins over the JSON default",
                         m.model, "openai/gpt-oss-120b"))
    results.append(check("base_url normalised", m.base, "https://api.groq.com/openai/v1"))
    os.environ.pop("GROQ_MODEL")

    # 7. registry entry is not-ready without a key, ready with one ----------- #
    # Regression: the UI reads info()["ready"] but the battle pairing reads the
    # ready *attribute*. When those disagreed, a keyless cloud contestant was
    # picked for every other battle and failed there. They must agree.
    print("\nregistry entry readiness (pairing must agree with the UI)")
    m = RemoteModel("groq-llama", entry["config"])
    results.append(check("greyed out without the key", m.info()["ready"], False))
    results.append(check("pairing sees it as not ready", m.ready, False))
    results.append(check("pairing sees the reason", m.error, "set $GROQ_API_KEY"))
    results.append(check("excluded from the available list",
                         [x.name for x in [m] if x.ready and not x.error], []))
    os.environ["GROQ_API_KEY"] = KEY
    results.append(check("ready once the key is set", m.info()["ready"], True))
    results.append(check("pairing agrees once the key is set", m.ready, True))
    results.append(check("no error once ready", m.error, None))
    results.append(check("model_env default is the documented one",
                         m.model, "openai/gpt-oss-120b"))
    os.environ.pop("GROQ_API_KEY")
    results.append(check("ready again reflects the missing key", m.ready, False))

    srv.shutdown()
    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
