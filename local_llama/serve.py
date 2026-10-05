#!/usr/bin/env python3
"""
Local Llama server: OpenAI-compatible API + a small web playground.

Two backends, picked automatically:

  * `llamacpp` -- starts `llama-server` from the llama.cpp build and proxies to it.
                  Fast (quantized kernels, SIMD, optional GPU). Preferred.
  * `numpy`    -- runs the checkpoint in-process with np_llama.py. No build step,
                  no extra binaries; slower but always available.

Endpoints (all standard, so existing tools work):
  GET  /                     web playground
  GET  /health               backend status
  GET  /v1/models            OpenAI-compatible model list
  POST /v1/completions       text completion (what this base model is good at)
  POST /v1/chat/completions  chat-completions shape; messages are flattened into
                             a single prompt, since stories15M is a base model

Usage:
    python serve.py                                  # auto backend, port 8000
    python serve.py --backend numpy                  # force the NumPy runner
    python serve.py --port 8000 --ctx 2048
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from np_llama import Llama2cModel, load_tokenizer  # noqa: E402

MODEL_NAME = "stories15M"
DEFAULT_BIN = HERE / "models" / "stories15M.bin"
DEFAULT_GGUF = HERE / "models" / "stories15M-f16.gguf"
DEFAULT_TOKENIZER = HERE / "models" / "tokenizer.model"
DEFAULT_TOKENIZER_BIN = HERE / "models" / "tokenizer.bin"
LLAMA_SERVER_BIN = HERE / "llama.cpp" / "build" / "bin" / "llama-server"
WEB_DIR = HERE / "web"

# Built phone app, offered for download so a phone can fetch it directly (over
# mobile data, say) instead of needing a computer to copy it across. Fixed names
# only -- nothing here is derived from the request path.
PHONE_DIR = HERE.parent / "phone"
DOWNLOADS = {
    "/LlamaPhone.html": (PHONE_DIR / "LlamaPhone.html", "text/html; charset=utf-8"),
    "/LlamaPhoneLite.html": (PHONE_DIR / "LlamaPhoneLite.html", "text/html; charset=utf-8"),
    "/LlamaPhone.llm": (PHONE_DIR / "LlamaPhone.llm", "application/octet-stream"),
}


def load_env_file(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from a .env file into os.environ.

    Existing environment variables always win, so `VAR=x python serve.py` still
    overrides the file. Never logs values. Missing file is not an error -- the
    file is a convenience for keeping keys out of your shell history.
    """
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


# --------------------------------------------------------------------------- #
# numpy backend
# --------------------------------------------------------------------------- #
class NumpyBackend:
    name = "numpy"

    def __init__(self, model_path: Path, tokenizer_model: Path, tokenizer_bin: Path):
        self.model = Llama2cModel(model_path)
        self.tokenizer = load_tokenizer(tokenizer_model, tokenizer_bin)
        self.lock = threading.Lock()  # one forward pass at a time

    def info(self) -> dict:
        m = self.model
        return {
            "backend": "numpy",
            "model": MODEL_NAME,
            "checkpoint": DEFAULT_BIN.name,
            "params": f"{m.n_layers}L/{m.dim}d/{m.n_heads}h",
            "context": 2048,
            "note": "pure NumPy reference runner (no llama.cpp build needed)",
        }

    def stream(self, prompt: str, max_tokens: int, temperature: float, top_p: float,
               top_k: int, seed: int):
        """Yield text deltas for a prompt."""
        from np_llama import sample

        with self.lock:
            tokens = self.tokenizer.encode(prompt, bos=True)
            max_seq = min(4096, len(tokens) + max_tokens + 8)
            cos_all, sin_all = self.model.rope_tables(max_seq)
            state = self.model.new_state(max_seq)
            rng = np.random.default_rng(seed)

            logits = None
            for tok in tokens:
                logits = self.model.forward(tok, state, cos_all, sin_all)
                if state["pos"] >= max_seq - 1:
                    break

            out_ids: list[int] = []
            emitted = ""
            for _ in range(max_tokens):
                nxt = sample(logits, temperature, top_p, top_k, rng)
                if nxt == self.tokenizer.eos_id:
                    break
                out_ids.append(nxt)
                text = self.tokenizer.decode(out_ids)
                if text != emitted:
                    yield text[len(emitted):]
                    emitted = text
                logits = self.model.forward(nxt, state, cos_all, sin_all)
                if state["pos"] >= max_seq - 1:
                    break
            self.last_stats = {"tokens": len(out_ids)}


# --------------------------------------------------------------------------- #
# llama.cpp backend
# --------------------------------------------------------------------------- #
class LlamaCppBackend:
    name = "llamacpp"

    def __init__(self, gguf_path: Path, port: int, ctx: int, threads: int):
        if not LLAMA_SERVER_BIN.exists():
            raise FileNotFoundError(
                f"{LLAMA_SERVER_BIN} not found -- run ./build.sh first")
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        cmd = [str(LLAMA_SERVER_BIN), "-m", str(gguf_path), "--port", str(port),
               "--host", "127.0.0.1", "-c", str(ctx), "-t", str(threads)]
        flags = _llama_server_flags()
        if "--no-webui" in flags:  # we serve our own playground
            cmd.append("--no-webui")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._wait_ready()

    def _wait_ready(self, timeout: float = 120.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited during startup")
            try:
                with urllib.request.urlopen(f"{self.base}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(0.4)
        raise RuntimeError("timed out waiting for llama-server")

    def info(self) -> dict:
        try:
            with urllib.request.urlopen(f"{self.base}/props", timeout=5) as r:
                props = json.load(r)
        except Exception:
            props = {}
        return {
            "backend": "llama.cpp",
            "model": props.get("model_path", str(DEFAULT_GGUF)).split("/")[-1],
            "context": props.get("n_ctx", 2048),
            "note": f"llama-server on {self.base}",
        }

    def stream(self, prompt: str, max_tokens: int, temperature: float, top_p: float,
               top_k: int, seed: int):
        body = json.dumps({
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "stream": True,
            "cache_prompt": False,
        }).encode()
        req = urllib.request.Request(f"{self.base}/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                piece = chunk.get("content")
                if piece:
                    yield piece
                if chunk.get("stop"):
                    return

    def stop(self) -> None:
        self.proc.terminate()


def _llama_server_flags() -> set[str]:
    """Cheap probe for which flags this llama-server build supports."""
    try:
        out = subprocess.run([str(LLAMA_SERVER_BIN), "--help"], capture_output=True,
                             text=True, timeout=30).stdout
    except Exception:
        return set()
    return set(out.split())


def pick_backend(args) -> object:
    if args.backend == "groq":
        from groq_backend import GroqBackend, MissingKey

        backend = GroqBackend(model=getattr(args, "groq_model", None))
        try:
            backend.require_key()   # refuse to boot rather than fail per request
        except MissingKey as exc:
            raise SystemExit(f"[serve] {exc}") from exc
        return backend
    if args.backend in ("auto", "llamacpp") and LLAMA_SERVER_BIN.exists() and args.gguf.exists():
        try:
            return LlamaCppBackend(args.gguf, args.llama_port, args.ctx, args.threads)
        except Exception as exc:  # fall back rather than fail to boot
            print(f"[serve] llama.cpp backend unavailable: {exc}", file=sys.stderr)
            if args.backend == "llamacpp":
                raise
    if not args.model.exists():
        raise SystemExit(f"missing checkpoint {args.model} -- run ./download_model.sh")
    return NumpyBackend(args.model, args.tokenizer, args.tokenizer_bin)


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #
class _ClientGone(Exception):
    """Raised when a streaming client hangs up; stops generation promptly."""


class Handler(BaseHTTPRequestHandler):
    backend = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter logs
        if os.environ.get("LLAMA_VERBOSE"):
            super().log_message(fmt, *args)

    # ---- helpers ---------------------------------------------------------- #
    def _write_body(self, body: bytes) -> None:
        """HEAD must send the same headers as GET but no body."""
        if getattr(self, "_head_only", False):
            return
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self._write_body(body)

    def _file(self, path: Path, content_type: str) -> None:
        if not path.exists() or not path.is_file():
            self._json({"error": f"{path.name} not found"}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if path.suffix in (".html", ""):
            # never cache a page: a stale copy looks exactly like a broken app
            self.send_header("Cache-Control", "no-store, must-revalidate")
        elif path.suffix in (".png", ".webmanifest", ".js", ".css"):
            self.send_header("Cache-Control", "public, max-age=300")
        self.end_headers()
        self._write_body(body)

    def _static(self, relative: str) -> None:
        """Serve a file from web/ without ever escaping it (path traversal)."""
        root = WEB_DIR.resolve()
        try:
            target = (root / relative.lstrip("/")).resolve()
        except (OSError, ValueError):
            self._json({"error": "bad path"}, 400)
            return
        if not target.is_relative_to(root):
            self._json({"error": "bad path"}, 400)
            return
        mime, _ = mimetypes.guess_type(target.name)
        self._file(target, mime or "application/octet-stream")

    def _download(self, path: Path, content_type: str) -> None:
        """Like _file, but offered as a download so a phone browser saves it."""
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        self._write_body(body)

    def _sse_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _sse(self, payload: dict | str) -> None:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        try:
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise _ClientGone from exc

    # ---- routes ----------------------------------------------------------- #
    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()

    def do_HEAD(self):  # noqa: N802
        """Preview proxies probe with HEAD; BaseHTTPRequestHandler would 501."""
        self._head_only = True
        try:
            self.do_GET()
        finally:
            self._head_only = False

    def do_GET(self):  # noqa: N802
        route = self.path.split("?")[0]
        if route in ("/", "/index.html"):
            self._file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        elif route == "/phone":
            # landing page for getting the apps onto a phone; contains QR codes
            self._file(PHONE_DIR / "iphone.html", "text/html; charset=utf-8")
        elif route == "/manifest.webmanifest":
            self._file(WEB_DIR / "manifest.webmanifest", "application/manifest+json")
        elif route == "/sw.js":
            self._file(WEB_DIR / "sw.js", "text/javascript; charset=utf-8")
        elif route.startswith("/static/"):
            self._static(route[len("/static/"):])
        elif route in DOWNLOADS:
            path, content_type = DOWNLOADS[route]
            if not path.exists():
                self._json({"error": f"{path.name} not built yet — run "
                                     f"tools/build_phone_app.py"}, 404)
                return
            self._download(path, content_type)
        elif route == "/health":
            info = dict(self.backend.info())
            info["status"] = "ok"
            self._json(info)
        elif route == "/v1/models":
            self._json({"object": "list", "data": [{
                "id": MODEL_NAME, "object": "model", "owned_by": "local",
                "created": int(time.time()),
            }]})
        else:
            self._json({"error": {"message": f"unknown route {route}"}}, 404)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _flatten_chat(messages: list[dict], system: str | None) -> str:
        """This is a base TinyStories model, not a chat model: flatten to text."""
        parts = []
        if system:
            parts.append(system.strip())
        for msg in messages or []:
            content = msg.get("content") or ""
            if isinstance(content, list):  # OpenAI content-parts shape
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            role = msg.get("role", "user")
            parts.append(f"{role}: {content.strip()}" if role == "system" else content.strip())
        return "\n".join(p for p in parts if p)

    def do_POST(self):  # noqa: N802
        route = self.path.split("?")[0]
        body = self._read_body()

        if route == "/v1/chat/completions":
            prompt = self._flatten_chat(body.get("messages", []), body.get("system"))
            stream = bool(body.get("stream"))
            reply = self._complete(prompt, body, stream, chat=True)
            if not stream:
                self._json(reply)
        elif route == "/v1/completions":
            prompt = body.get("prompt") or ""
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            stream = bool(body.get("stream"))
            reply = self._complete(prompt, body, stream, chat=False)
            if not stream:
                self._json(reply)
        else:
            self._json({"error": {"message": f"unknown route {route}"}}, 404)

    def _complete(self, prompt: str, body: dict, stream: bool, chat: bool):
        max_tokens = int(body.get("max_tokens") or body.get("n_predict") or 200)
        temperature = float(body.get("temperature", 0.8))
        top_p = float(body.get("top_p", 0.9))
        top_k = int(body.get("top_k", 40))
        seed = int(body.get("seed", -1))
        if seed < 0:
            seed = int(time.time() * 1000) % (2**31)

        started = time.time()
        text = ""
        sse_open = False
        try:
            for delta in self.backend.stream(prompt, max_tokens, temperature, top_p,
                                             top_k, seed):
                text += delta
                if stream:
                    if not sse_open:
                        self._sse_start()
                        sse_open = True
                    if chat:
                        self._sse({"object": "chat.completion.chunk", "model": MODEL_NAME,
                                   "choices": [{"index": 0, "delta": {"content": delta}}]})
                    else:
                        self._sse({"object": "text_completion", "model": MODEL_NAME,
                                   "choices": [{"index": 0, "text": delta}]})
        except _ClientGone:
            return None  # browser navigated away / aborted; drop the connection

        elapsed = max(time.time() - started, 1e-6)
        approx_tokens = max(1, len(text) // 4)
        if stream:
            self._sse("[DONE]")
            return None
        if chat:
            return {
                "id": f"chatcmpl-{int(started)}", "object": "chat.completion",
                "created": int(started), "model": MODEL_NAME,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": approx_tokens,
                          "total_tokens": (len(prompt) + len(text)) // 4},
                "timings": {"tokens_per_second": round(approx_tokens / elapsed, 1)},
            }
        return {
            "id": f"cmpl-{int(started)}", "object": "text_completion",
            "created": int(started), "model": MODEL_NAME,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": approx_tokens,
                      "total_tokens": (len(prompt) + len(text)) // 4},
            "timings": {"tokens_per_second": round(approx_tokens / elapsed, 1)},
        }


def make_server(host: str, port: int, handler):
    """An IPv4 server (which is what the preview detector looks for) plus an
    IPv6 loopback listener on the same port.

    Why both: the stdlib default binds IPv4 only, so a proxy that resolves
    "localhost" to ::1 gets connection-refused while every IPv4 test passes --
    it looks exactly like a dead server. But binding the *wildcard* only as IPv6
    (`::`) hides the port from tooling that scans for 0.0.0.0. So: keep the IPv4
    bind that gets detected, and add [::1]:port for IPv6-localhost clients.

    The accept backlog is raised from the stdlib default of 5 to 128: a browser
    opens about six connections per host, and 5 slots is easy to exhaust.
    """
    ThreadingHTTPServer.request_queue_size = 128
    ThreadingHTTPServer.daemon_threads = True

    if host in ("0.0.0.0", "", "::"):
        primary = ThreadingHTTPServer(("0.0.0.0", port), handler)
        threading.Thread(target=_serve_ipv6_loopback, args=(port, handler),
                         daemon=True).start()
        return primary
    return ThreadingHTTPServer((host, port), handler)


def _serve_ipv6_loopback(port: int, handler) -> None:
    """Second listener for ::1 only; silently skipped where IPv6 is unavailable."""
    class V6Loopback(ThreadingHTTPServer):
        address_family = socket.AF_INET6
        request_queue_size = 128
        daemon_threads = True

        def server_bind(self):
            # V6ONLY keeps this socket off IPv4, so it cannot clash with the
            # 0.0.0.0 listener on the same port.
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            super().server_bind()

    try:
        V6Loopback(("::1", port), handler).serve_forever()
    except OSError:
        pass  # no IPv6 here, or the port is taken: the IPv4 listener still serves


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--backend", choices=("auto", "llamacpp", "numpy", "groq"),
                    default="auto",
                    help="groq sends prompts to api.groq.com (needs GROQ_API_KEY); "
                         "auto never picks it")
    ap.add_argument("--env-file", default=None,
                    help="load KEY=VALUE pairs from this file (default: .env at "
                         "the repo root, if present)")
    ap.add_argument("--groq-model", default=None,
                    help="model id for --backend groq (default $GROQ_MODEL or "
                         "openai/gpt-oss-120b)")
    ap.add_argument("--model", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--tokenizer-bin", type=Path, default=DEFAULT_TOKENIZER_BIN)
    ap.add_argument("--llama-port", type=int, default=8081)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()

    load_env_file(Path(args.env_file) if args.env_file else None)
    backend = pick_backend(args)
    Handler.backend = backend
    info = backend.info()
    print(f"[serve] backend: {info['backend']}  ({info.get('note', '')})")
    print(f"[serve] listening on http://{args.host}:{args.port}/")

    server = make_server(args.host, args.port, Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(backend, "stop"):
            backend.stop()


if __name__ == "__main__":
    main()
