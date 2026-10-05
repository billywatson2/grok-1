#!/usr/bin/env python3
"""
LM Arena -- a local, self-hosted model battle arena.

Two models answer the same prompt side by side. Their identities stay hidden
until you vote. Votes feed an Elo leaderboard. Runs entirely on your machine:
local llama.cpp models are started and managed for you, and you can register any
OpenAI-compatible endpoint as an extra contestant.

    python3 arena.py                 # http://localhost:8100
    python3 arena.py --port 8100 --host 0.0.0.0

Layout
    arena.py                this server (HTTP + SQLite + Elo)
    models.json             which models compete (seeded into the DB on first run)
    arena.db                votes, battles, ratings (created on first run)
    static/                 battle UI, leaderboard, styles

API
    GET  /                          battle UI
    GET  /leaderboard               leaderboard UI
    GET  /api/models                contestants + ratings
    POST /api/models                add an OpenAI-compatible endpoint
    POST /api/battle                run a blind battle, streams SSE
    POST /api/vote                  vote on a battle, returns the reveal + rating change
    GET  /api/leaderboard           ranked contestants
    GET  /api/battles               recent battle history
    GET  /health                    server + model status
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import queue
import random
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
DEFAULT_DB = HERE / "arena.db"
DEFAULT_REGISTRY = HERE / "models.json"
LLAMA_SERVER = HERE.parent / "local_llama" / "llama.cpp" / "build" / "bin" / "llama-server"

# rating constants
ELO_START = 1000.0
ELO_K = 32.0
ELO_SCALE = 400.0


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #
class Store:
    """SQLite persistence: models, battles (with both responses) and ratings."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,             -- 'local' | 'openai'
                    config TEXT NOT NULL,           -- JSON blob
                    elo REAL NOT NULL DEFAULT 1000,
                    wins INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,
                    ties INTEGER NOT NULL DEFAULT 0,
                    battles INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS battles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    prompt TEXT NOT NULL,
                    category TEXT,
                    model_a INTEGER NOT NULL,
                    model_b INTEGER NOT NULL,
                    text_a TEXT NOT NULL DEFAULT '',
                    text_b TEXT NOT NULL DEFAULT '',
                    winner TEXT,                    -- 'a' | 'b' | 'tie' | 'both_bad'
                    elo_a_before REAL, elo_b_before REAL,
                    elo_a_after REAL,  elo_b_after REAL,
                    voted_at REAL
                );
            """)

    # -- models ------------------------------------------------------------- #
    def add_model(self, name: str, kind: str, config: dict) -> int:
        with self.lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO models (name, kind, config, created_at) "
                "VALUES (?, ?, ?, ?)", (name, kind, json.dumps(config), time.time()))
            if cur.lastrowid:
                return int(cur.lastrowid)
            row = conn.execute("SELECT id FROM models WHERE name = ?", (name,)).fetchone()
            return int(row["id"])

    def models(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM models ORDER BY elo DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["config"] = json.loads(d["config"])
            out.append(d)
        return out

    def model(self, model_id: int) -> dict | None:
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["config"] = json.loads(d["config"])
        return d

    def delete_model(self, model_id: int) -> None:
        with self.lock, self._connect() as conn:
            conn.execute("DELETE FROM models WHERE id = ?", (model_id,))

    # -- battles ------------------------------------------------------------ #
    def create_battle(self, prompt: str, category: str | None, a: int, b: int,
                      elo_a: float, elo_b: float) -> int:
        with self.lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO battles (created_at, prompt, category, model_a, model_b,"
                " elo_a_before, elo_b_before) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (time.time(), prompt, category, a, b, elo_a, elo_b))
            return int(cur.lastrowid)

    def save_responses(self, battle_id: int, text_a: str, text_b: str) -> None:
        with self.lock, self._connect() as conn:
            conn.execute("UPDATE battles SET text_a = ?, text_b = ? WHERE id = ?",
                         (text_a, text_b, battle_id))

    def battle(self, battle_id: int) -> dict | None:
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM battles WHERE id = ?", (battle_id,)).fetchone()
        return dict(r) if r else None

    def recent_battles(self, limit: int = 25) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT b.*, ma.name AS name_a, mb.name AS name_b FROM battles b "
                "JOIN models ma ON ma.id = b.model_a JOIN models mb ON mb.id = b.model_b "
                "ORDER BY b.id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def vote(self, battle_id: int, winner: str,
             elo_a_after: float, elo_b_after: float) -> None:
        """Record the vote and apply the rating bookkeeping."""
        with self.lock, self._connect() as conn:
            conn.execute(
                "UPDATE battles SET winner = ?, voted_at = ?, elo_a_after = ?,"
                " elo_b_after = ? WHERE id = ?",
                (winner, time.time(), elo_a_after, elo_b_after, battle_id))
            b = conn.execute("SELECT * FROM battles WHERE id = ?", (battle_id,)).fetchone()
            if b is None:
                return
            if winner == "both_bad":
                # Recorded in the battle history, but deliberately excluded from
                # both the ratings and the W/L/T record. This keeps
                # battles == wins + losses + ties exactly.
                return
            a, bb = b["model_a"], b["model_b"]
            if winner == "a":
                outcomes = ((a, "wins"), (bb, "losses"))
            elif winner == "b":
                outcomes = ((a, "losses"), (bb, "wins"))
            else:
                outcomes = ((a, "ties"), (bb, "ties"))
            for model_id, column in outcomes:
                conn.execute(f"UPDATE models SET {column} = {column} + 1 WHERE id = ?",
                             (model_id,))
            for model_id, elo in ((a, elo_a_after), (bb, elo_b_after)):
                conn.execute("UPDATE models SET elo = ?, battles = battles + 1"
                             " WHERE id = ?", (elo, model_id))

    def rated_battles(self) -> list[tuple[int, int, float]]:
        """(winner_id, loser_id, score) for every vote that counts toward ratings."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT model_a, model_b, winner FROM battles "
                "WHERE winner IN ('a','b','tie') ORDER BY id").fetchall()
        out = []
        for r in rows:
            if r["winner"] == "a":
                out.append((r["model_a"], r["model_b"], 1.0))
            elif r["winner"] == "b":
                out.append((r["model_b"], r["model_a"], 1.0))
            else:
                out.append((r["model_a"], r["model_b"], 0.5))
        return out

    def update_elo(self, model_id: int, elo: float) -> None:
        with self.lock, self._connect() as conn:
            conn.execute("UPDATE models SET elo = ? WHERE id = ?", (elo, model_id))


# --------------------------------------------------------------------------- #
# rating math
# --------------------------------------------------------------------------- #
def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / ELO_SCALE))


def elo_update(rating_a: float, rating_b: float, score_a: float,
               k: float = ELO_K) -> tuple[float, float]:
    """Both ratings after a game where A scored `score_a` (1 win, 0.5 tie, 0 loss)."""
    ea = expected_score(rating_a, rating_b)
    delta = k * (score_a - ea)
    return rating_a + delta, rating_b - delta


def bradley_terry(battles: list[tuple[int, int, float]], ids: list[int],
                  iterations: int = 300) -> dict[int, float]:
    """Order-independent Bradley-Terry fit, returned on the Elo 400-point scale.

    Wins and half-wins are the counts; the MM update is the classic
    `R_i <- W_i / sum_j n_ij / (R_i + R_j)`.
    """
    wins = {i: 0.0 for i in ids}
    games: dict[tuple[int, int], float] = {}
    for winner, loser, score in battles:
        wins[winner] += score
        wins[loser] += 1.0 - score
        games[(winner, loser)] = games.get((winner, loser), 0.0) + 1.0
        games[(loser, winner)] = games.get((loser, winner), 0.0) + 1.0

    strength = {i: 1.0 for i in ids}
    for _ in range(iterations):
        updated = {}
        for i in ids:
            denom = 0.0
            for (a, b), n in games.items():
                if a == i:
                    denom += n / (strength[i] + strength[b])
            updated[i] = (wins[i] / denom) if denom > 0 else strength[i]
        norm = math.exp(sum(math.log(max(v, 1e-9)) for v in updated.values())
                        / max(len(updated), 1))
        strength = {i: max(v, 1e-9) / norm for i, v in updated.items()}

    base = 1000.0
    if not strength:
        return {}
    logs = {i: math.log(max(v, 1e-12)) for i, v in strength.items()}
    mean = sum(logs.values()) / len(logs)
    return {i: base + ELO_SCALE * (logs[i] - mean) / math.log(10) for i in ids}


def bootstrap_ci(battles: list[tuple[int, int, float]], ids: list[int],
                 samples: int = 200, seed: int = 7) -> dict[int, tuple[float, float]]:
    """95% CI per model, by refitting Bradley-Terry on resampled battle lists."""
    if not battles:
        return {i: (ELO_START, ELO_START) for i in ids}
    rng = random.Random(seed)
    draws: dict[int, list[float]] = {i: [] for i in ids}
    for _ in range(samples):
        resampled = [battles[rng.randrange(len(battles))] for _ in battles]
        fit = bradley_terry(resampled, ids, iterations=120)
        for i in ids:
            draws[i].append(fit.get(i, ELO_START))
    out = {}
    for i in ids:
        values = sorted(draws[i])
        lo = values[int(0.025 * (len(values) - 1))]
        hi = values[int(0.975 * (len(values) - 1))]
        out[i] = (lo, hi)
    return out


# --------------------------------------------------------------------------- #
# model backends
# --------------------------------------------------------------------------- #
class LocalModel:
    """A llama.cpp model: the arena starts and stops its llama-server."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.gguf = (HERE / config["gguf"]).resolve() if not Path(config["gguf"]).is_absolute() \
            else Path(config["gguf"])
        self.port = int(config.get("port", 0))
        self.ctx = int(config.get("ctx", 512))
        self.threads = int(config.get("threads", 2))
        self.base = f"http://127.0.0.1:{self.port}"          # native llama-server
        self.api_base = f"{self.base}/v1"                    # OpenAI-compatible
        self.api = "completions"
        self.process: subprocess.Popen | None = None
        self.ready = False
        self.error: str | None = None
        self.lock = threading.Lock()

    def start(self) -> None:
        with self.lock:
            if self.process and self.process.poll() is None:
                self.ready = True
                return
            if not self.gguf.exists():
                self.error = f"missing checkpoint: {self.gguf}"
                return
            if not LLAMA_SERVER.exists():
                self.error = f"missing {LLAMA_SERVER} (build local_llama first)"
                return
            log = open(HERE / f"llama-server-{self.port}.log", "ab")
            self.process = subprocess.Popen(
                [str(LLAMA_SERVER), "-m", str(self.gguf), "--port", str(self.port),
                 "--host", "127.0.0.1", "-c", str(self.ctx), "-t", str(self.threads)],
                stdout=log, stderr=log)
            deadline = time.time() + 120
            while time.time() < deadline:
                if self.process.poll() is not None:
                    self.error = "llama-server exited during startup"
                    return
                try:
                    with urllib.request.urlopen(f"{self.base}/health", timeout=2) as r:
                        if r.status == 200:
                            self.ready = True
                            self.error = None
                            return
                except Exception:
                    time.sleep(0.3)
            self.error = "timed out waiting for llama-server"

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()

    def info(self) -> dict:
        return {"name": self.name, "kind": "local", "ready": self.ready,
                "error": self.error, "endpoint": self.base,
                "model": self.gguf.name}

    def stream(self, prompt: str, max_tokens: int, temperature: float, top_p: float,
               seed: int):
        if not self.ready:
            self.start()
        if not self.ready:
            raise RuntimeError(self.error or "model not ready")
        return openai_stream(self.api_base, "completions", "local", prompt,
                             max_tokens, temperature, top_p, seed, api_key=None)


class RemoteModel:
    """Any OpenAI-compatible endpoint (Ollama, LM Studio, vLLM, OpenRouter, ...)."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.base = config["base_url"].rstrip("/")
        if not self.base.endswith("/v1"):
            self.base += "/v1"
        self.model = config.get("model", name)
        self.api = config.get("api", "chat")           # 'chat' | 'completions'
        self.api_key_env = config.get("api_key_env")
        self.ready = True
        self.error = None

    def start(self) -> None:  # nothing to start
        return

    def stop(self) -> None:
        return

    def info(self) -> dict:
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        return {"name": self.name, "kind": "openai", "ready": bool(key or not self.api_key_env),
                "error": None if (key or not self.api_key_env)
                else f"set ${self.api_key_env}", "endpoint": self.base, "model": self.model}

    def stream(self, prompt: str, max_tokens: int, temperature: float, top_p: float,
               seed: int):
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        return openai_stream(self.base, self.api, self.model, prompt, max_tokens,
                             temperature, top_p, seed, api_key=key)


def openai_stream(base: str, api: str, model: str, prompt: str, max_tokens: int,
                  temperature: float, top_p: float, seed: int, api_key: str | None):
    """Yield text deltas from an OpenAI-compatible streaming endpoint.

    `base` must already include the API prefix, e.g. http://host:port/v1.
    """
    if api == "chat":
        url = f"{base}/chat/completions"
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": temperature,
                   "top_p": top_p, "seed": seed, "stream": True}
    else:
        url = f"{base}/completions"
        payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
                   "temperature": temperature, "top_p": top_p, "seed": seed,
                   "stream": True}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                return
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            text = choice.get("text")
            if text is None:
                text = delta.get("content") or ""
            if text:
                yield text


def build_model(row: dict):
    cfg = row["config"]
    if row["kind"] == "local":
        model = LocalModel(row["name"], cfg)
        model.id = row["id"]
        return model
    model = RemoteModel(row["name"], cfg)
    model.id = row["id"]
    return model


# --------------------------------------------------------------------------- #
# arena
# --------------------------------------------------------------------------- #
class Arena:
    def __init__(self, store: Store, registry: Path):
        self.store = store
        self.models: dict[int, object] = {}
        self._ci_cache: tuple[float, dict] = (0.0, {})
        self._load_registry(registry)
        self._boot_thread = threading.Thread(target=self._boot, daemon=True)
        self._boot_thread.start()

    def _load_registry(self, registry: Path) -> None:
        if not registry.exists():
            return
        spec = json.loads(registry.read_text())
        for entry in spec.get("models", []):
            self.store.add_model(entry["name"], entry["kind"], entry.get("config", {}))
        # refresh in-memory handles from the DB (so ids are right)
        self.reload()

    def reload(self) -> None:
        self.models = {}
        for row in self.store.models():
            self.models[row["id"]] = build_model(row)

    def _boot(self) -> None:
        for model in list(self.models.values()):
            if isinstance(model, LocalModel):
                try:
                    model.start()
                except Exception as exc:  # keep the arena usable regardless
                    model.error = str(exc)

    def available(self) -> list:
        return [m for m in self.models.values()
                if getattr(m, "ready", False) and not getattr(m, "error", None)]

    def shutdown(self) -> None:
        for model in self.models.values():
            try:
                model.stop()
            except Exception:
                pass

    def ci(self) -> dict:
        """Bootstrap CIs, cached for 30s (they only change when a vote lands)."""
        stamp, cached = self._ci_cache
        if time.time() - stamp < 30 and cached:
            return cached
        ids = [m["id"] for m in self.store.models()]
        fit = bootstrap_ci(self.store.rated_battles(), ids, samples=200)
        self._ci_cache = (time.time(), fit)
        return fit


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    arena: Arena = None
    access_token: str | None = None      # set with --token; None means open access
    protocol_version = "HTTP/1.1"

    # -- access control ----------------------------------------------------- #
    def _token_from_request(self) -> tuple[str | None, bool]:
        """Return (token, came_from_query) for this request."""
        auth = self.headers.get("Authorization") or ""
        if auth.startswith("Bearer "):
            return auth[7:].strip(), False
        for chunk in (self.headers.get("Cookie") or "").split(";"):
            name, _, value = chunk.strip().partition("=")
            if name == "arena_token":
                return value, False
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        for pair in query.split("&"):
            name, _, value = pair.partition("=")
            if name == "token":
                from urllib.parse import unquote_plus
                return unquote_plus(value), True
        return None, False

    def _authorized(self) -> bool:
        """True when the request may proceed; sends the 401 itself when not."""
        if not self.access_token:
            return True
        token, from_query = self._token_from_request()
        if token == self.access_token:
            if from_query:
                # remember it so the page's own asset/API calls work; HttpOnly so
                # scripts cannot read it back out
                self._pending_cookie = ("arena_token=" + token
                                        + "; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000")
            return True
        self._json({"error": {"message": "unauthorized: append ?token=... once, "
                                         "or send Authorization: Bearer <token>"}}, 401)
        return False

    def end_headers(self):  # noqa: N802
        cookie = getattr(self, "_pending_cookie", None)
        if cookie:
            self.send_header("Set-Cookie", cookie)
            self._pending_cookie = None
        super().end_headers()

    def log_message(self, fmt, *args):
        if os.environ.get("ARENA_VERBOSE"):
            super().log_message(fmt, *args)

    # -- helpers ------------------------------------------------------------ #
    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type: str) -> None:
        if not path.exists() or not path.is_file():
            self._json({"error": f"{path.name} not found"}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if path.suffix in (".png", ".webmanifest", ".css", ".js"):
            self.send_header("Cache-Control", "public, max-age=300")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, relative: str) -> None:
        """Serve a file from static/ without ever escaping it (path traversal)."""
        root = STATIC.resolve()
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

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def _sse_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # no Content-Length is possible for a stream, so the connection close is
        # what tells the client the response is over -- without this the browser
        # keeps the fetch() open forever and the UI stays stuck in "running".
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.close_connection = True

    def _sse(self, payload) -> None:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        self.wfile.write(f"data: {data}\n\n".encode())
        self.wfile.flush()

    # -- routes ------------------------------------------------------------- #
    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS, DELETE")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        if not self._authorized():
            return
        route = self.path.split("?")[0]
        if route in ("/", "/index.html"):
            self._file(STATIC / "index.html", "text/html; charset=utf-8")
        elif route == "/leaderboard":
            self._file(STATIC / "leaderboard.html", "text/html; charset=utf-8")
        elif route == "/manifest.webmanifest":
            self._file(STATIC / "manifest.webmanifest", "application/manifest+json")
        elif route == "/sw.js":
            # served from the root so its scope covers "/" -- required for the
            # navigations to be handled offline
            self._file(STATIC / "sw.js", "text/javascript; charset=utf-8")
        elif route.startswith("/static/"):
            self._static(route[len("/static/"):])
        elif route == "/health":
            self._json(self._health())
        elif route == "/api/models":
            self._json(self._models())
        elif route == "/api/leaderboard":
            self._json(self._leaderboard())
        elif route == "/api/battles":
            self._json({"battles": self.arena.store.recent_battles(25)})
        else:
            self._json({"error": {"message": f"unknown route {route}"}}, 404)

    def do_DELETE(self):  # noqa: N802
        if not self._authorized():
            return
        route = self.path.split("?")[0]
        if route.startswith("/api/models/"):
            try:
                model_id = int(route.rsplit("/", 1)[-1])
            except ValueError:
                self._json({"error": {"message": "bad id"}}, 400)
                return
            model = self.arena.models.get(model_id)
            if model:
                model.stop()
            self.arena.store.delete_model(model_id)
            self.arena.reload()
            self._json({"ok": True})
        else:
            self._json({"error": {"message": "unknown route"}}, 404)

    def do_POST(self):  # noqa: N802
        if not self._authorized():
            return
        route = self.path.split("?")[0]
        body = self._body()
        if route == "/api/battle":
            self._battle(body)
        elif route == "/api/vote":
            self._vote(body)
        elif route == "/api/models":
            self._add_model(body)
        else:
            self._json({"error": {"message": f"unknown route {route}"}}, 404)

    # -- handlers ----------------------------------------------------------- #
    def _health(self) -> dict:
        return {
            "status": "ok",
            "models": [m.info() for m in self.arena.models.values()],
            "votes": len(self.arena.store.rated_battles()),
        }

    def _models(self) -> dict:
        ci = self.arena.ci()
        out = []
        for row in self.arena.store.models():
            model = self.arena.models.get(row["id"])
            entry = {
                "id": row["id"], "name": row["name"], "kind": row["kind"],
                "elo": round(row["elo"], 1),
                "ci": [round(v, 1) for v in ci.get(row["id"], (ELO_START, ELO_START))],
                "wins": row["wins"], "losses": row["losses"], "ties": row["ties"],
                "battles": row["battles"],
                "available": bool(getattr(model, "ready", False)) if model else False,
                "detail": getattr(model, "info", lambda: {})() if model else {},
            }
            if row["kind"] == "local":
                entry["context"] = row["config"].get("ctx")
            out.append(entry)
        return {"models": out}

    def _leaderboard(self) -> dict:
        data = self._models()["models"]
        data.sort(key=lambda m: -m["elo"])
        for rank, entry in enumerate(data, start=1):
            entry["rank"] = rank
            decided = entry["wins"] + entry["losses"]
            entry["win_rate"] = round(entry["wins"] / decided, 3) if decided else None
        return {"leaderboard": data}

    def _add_model(self, body: dict) -> None:
        name = (body.get("name") or "").strip()
        if not name:
            self._json({"error": {"message": "name required"}}, 400)
            return
        if body.get("kind") == "local":
            config = {"gguf": body["gguf"], "port": int(body.get("port", 8200)),
                      "ctx": int(body.get("ctx", 512)),
                      "threads": int(body.get("threads", 2))}
        else:
            if not body.get("base_url"):
                self._json({"error": {"message": "base_url required"}}, 400)
                return
            config = {"base_url": body["base_url"], "model": body.get("model", name),
                      "api": body.get("api", "chat"),
                      "api_key_env": body.get("api_key_env")}
        model_id = self.arena.store.add_model(name, body.get("kind", "openai"), config)
        self.arena.reload()
        model = self.arena.models.get(model_id)
        if isinstance(model, LocalModel):
            threading.Thread(target=model.start, daemon=True).start()
        self._json({"id": model_id, "name": name})

    def _battle(self, body: dict) -> None:
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            self._json({"error": {"message": "prompt required"}}, 400)
            return
        max_tokens = int(body.get("max_tokens", 120))
        temperature = float(body.get("temperature", 0.8))
        top_p = float(body.get("top_p", 0.95))

        pool = self.arena.available()
        requested = [body.get("model_a"), body.get("model_b")]
        chosen = []
        for want in requested:
            if want:
                match = next((m for m in self.arena.models.values() if m.name == want), None)
                if match and match not in chosen:
                    chosen.append(match)
        remaining = [m for m in pool if m not in chosen]
        random.shuffle(remaining)
        while len(chosen) < 2 and remaining:
            chosen.append(remaining.pop())
        if len(chosen) < 2:
            self._json({"error": {"message": "need two available models: "
                                             f"{[m.info() for m in self.arena.models.values()]}"}},
                       409)
            return

        model_a, model_b = chosen[0], chosen[1]
        row_a = self.arena.store.model(model_a.id)
        row_b = self.arena.store.model(model_b.id)
        battle_id = self.arena.store.create_battle(
            prompt, body.get("category"), model_a.id, model_b.id,
            row_a["elo"], row_b["elo"])

        started = time.time()
        results: queue.Queue = queue.Queue()

        def pump(side: str, model) -> None:
            text = ""
            try:
                for delta in model.stream(prompt, max_tokens, temperature, top_p,
                                          seed=random.randint(0, 2**31 - 1)):
                    text += delta
                    results.put({"type": "chunk", "side": side, "text": delta})
            except Exception as exc:
                results.put({"type": "error", "side": side, "message": str(exc)})
            finally:
                results.put({"type": "done", "side": side, "length": len(text)})
                results.put(("final", side, text))

        threads = [threading.Thread(target=pump, args=("a", model_a), daemon=True),
                   threading.Thread(target=pump, args=("b", model_b), daemon=True)]
        for t in threads:
            t.start()

        self._sse_start()
        self._sse({"type": "start", "battle_id": battle_id,
                   "prompt": prompt, "max_tokens": max_tokens})
        texts = {"a": "", "b": ""}
        pending = 2
        try:
            while pending > 0:
                item = results.get(timeout=900)
                if isinstance(item, tuple):
                    _, side, text = item
                    texts[side] = text
                    pending -= 1
                    continue
                if item["type"] == "chunk":
                    texts[item["side"]] += item["text"]
                self._sse(item)
            self.arena.store.save_responses(battle_id, texts["a"], texts["b"])
            self._sse({"type": "end", "battle_id": battle_id,
                       "seconds": round(time.time() - started, 1)})
            self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away; threads finish on their own and are daemonized

    def _vote(self, body: dict) -> None:
        battle_id = int(body.get("battle_id") or 0)
        winner = body.get("winner")
        if winner not in ("a", "b", "tie", "both_bad"):
            self._json({"error": {"message": "winner must be a|b|tie|both_bad"}}, 400)
            return
        battle = self.arena.store.battle(battle_id)
        if not battle:
            self._json({"error": {"message": "unknown battle"}}, 404)
            return
        if battle["winner"]:
            self._json({"error": {"message": "already voted"}}, 409)
            return

        elo_a, elo_b = battle["elo_a_before"], battle["elo_b_before"]
        if winner == "both_bad":
            new_a, new_b = elo_a, elo_b
        else:
            score_a = 1.0 if winner == "a" else 0.0 if winner == "b" else 0.5
            new_a, new_b = elo_update(elo_a, elo_b, score_a)

        self.arena.store.vote(battle_id, winner, new_a, new_b)
        row_a = self.arena.store.model(battle["model_a"])
        row_b = self.arena.store.model(battle["model_b"])
        self.arena._ci_cache = (0.0, {})  # invalidate

        def side(row, before, after):
            return {"id": row["id"], "name": row["name"], "elo_before": round(before, 1),
                    "elo_after": round(after, 1), "delta": round(after - before, 1),
                    "wins": row["wins"], "losses": row["losses"], "ties": row["ties"]}

        self._json({"ok": True, "battle_id": battle_id, "winner": winner,
                    "a": side(row_a, elo_a, new_a), "b": side(row_b, elo_b, new_b),
                    "rated": winner != "both_bad"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8100)))
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--reset", action="store_true", help="delete the ratings database first")
    ap.add_argument("--token", default=os.environ.get("ARENA_TOKEN"),
                    help="require this token for every route (also $ARENA_TOKEN). "
                         "Use it whenever the arena is reachable from other devices.")
    args = ap.parse_args()

    if args.reset and args.db.exists():
        args.db.unlink()
        print(f"[arena] reset {args.db}")

    store = Store(args.db)
    arena = Arena(store, args.registry)
    Handler.arena = arena
    Handler.access_token = args.token or None
    print(f"[arena] {len(arena.models)} contestants registered, "
          f"{len(arena.available())} ready")
    print(f"[arena] listening on http://{args.host}:{args.port}/")
    if args.token:
        print("[arena] access token required (open /?token=... once, or send "
              "Authorization: Bearer <token>)")
    elif args.host not in ("127.0.0.1", "localhost"):
        print("[arena] NOTE: bound to a non-local address with no token -- anyone who "
              "can reach this port can use your models and cast votes")

    # SIGTERM (Ctrl-C's cousin, and what process managers send) should stop the
    # llama-server children too, not leave them orphaned on their ports.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[arena] shutting down")
    finally:
        server.server_close()
        arena.shutdown()


if __name__ == "__main__":
    main()
