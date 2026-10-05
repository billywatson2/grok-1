# LM Arena

A self-hosted model battle arena — the LMArena/“chatbot arena” idea, running
entirely on your machine.

Two models answer the same prompt side by side. **Their names stay hidden until
you vote**, so the rating reflects preference rather than brand. Votes feed an
Elo leaderboard with bootstrap confidence intervals. Local GGUF models are
started and managed for you; any OpenAI-compatible endpoint can join the pool.

```
        prompt ──┬──────────────────────────────┐
                 ▼                              ▼
          Model A (hidden)              Model B (hidden)
                 │  llama.cpp            │  llama.cpp / any OpenAI API
                 └────────► you vote ◄───┘
                              │
                              ▼
                 Elo  +  bootstrap 95% CI  →  leaderboard
```

Everything below is pure Python standard library — no FastAPI, no npm, nothing to
install beyond what `../local_llama` already needs.

## Quickstart

```bash
# 1. models (61 MB + 33 MB, checksum-verified)
cd ../local_llama && ./download_model.sh && ./build.sh     # ~10-25 min on 2 cores

# 2. the arena
cd ../lm_arena
python3 arena.py                # → http://localhost:8100
```

**On a phone instead of a computer: see [`../phone/README.md`](../phone/README.md).**
The arena is a PWA — it installs to the home screen and, with the models running
on the phone itself, needs no WiFi at all.

Open `http://localhost:8100`, type a prompt, hit **Start battle**, watch both
models stream, then vote. The leaderboard is at `/leaderboard`.

Useful flags:

```bash
python3 arena.py --port 8100 --host 0.0.0.0    # defaults
python3 arena.py --reset                       # wipe ratings and start over
python3 arena.py --registry my_models.json     # a different contestant list
python3 arena.py --token LONG_SECRET           # require a token (see below)
python3 arena.py --host 127.0.0.1              # this device only, no network
```

`--token` (or `$ARENA_TOKEN`) locks every route behind a shared secret. Open the
page once with `?token=...`; it is then kept as an HttpOnly cookie. Use it
whenever the arena is reachable from another device — otherwise anyone who can
reach the port can spend your CPU, vote, and delete contestants. Binding to
`127.0.0.1` needs no token: nothing outside the device can connect.

The UI is installable (PWA): *Add to Home Screen* gives it its own icon and
window, and a service worker keeps the shell cached so a model-server restart
does not leave you staring at a blank page. Generation and voting are never
cached.

## The contestants

Seeded from `models.json` on first run:

| Model | What it is | Context |
|---|---|---|
| `stories15M` | Llama-2 architecture, 15M params, 32k SentencePiece vocab, TinyStories | 512 |
| `stories15M-tok4096` | same recipe, retrained with a 4096-token BPE vocab | 256 |
| `groq-llama` | cloud contestant: `openai/gpt-oss-120b` on Groq, if `$GROQ_API_KEY` is set | provider |

`groq-llama` is registered from `models.json` but is **not ready without a key**:
`/health` reports it with `"error": "set $GROQ_API_KEY"`, and the battle pairing
skips it, so the arena keeps working exactly as before. Set the key where the
arena runs and it joins the rotation:

```bash
export GROQ_API_KEY=gsk_...       # https://console.groq.com/keys
export GROQ_MODEL=openai/gpt-oss-20b   # optional; $GROQ_MODEL overrides the JSON
python arena.py
```

Groq retires model ids on a schedule — `llama-3.1-8b-instant` and
`llama-3.3-70b-versatile` both went away on 2026-08-16 — so if a battle reports
the model as missing, list what your key can use:

```bash
curl -s -H "Authorization: Bearer $GROQ_API_KEY" \
  https://api.groq.com/openai/v1/models | grep -o '"id": *"[^"]*"'
```

Keys live in the environment (or in a gitignored `.env` at the repo root; see
`.env.example`). `python3 tools/check_keys.py` reports what is configured
without printing any value.

Two real checkpoints with different tokenizers, so the same prompt produces
genuinely different stories — a fair fight for a preference vote. Both are tiny
(31 MB and 33 MB), run on CPU at a few hundred tokens/sec, and are converted/
served by the `local_llama` tooling next door.

## Adding contestants

### Any OpenAI-compatible endpoint

Ollama, LM Studio, vLLM, llama.cpp's own server, OpenRouter, Together, OpenAI…
anything that speaks `/v1/chat/completions` or `/v1/completions`:

```bash
curl -X POST http://localhost:8100/api/models -H 'Content-Type: application/json' -d '{
  "name": "llama3.2-1b",
  "kind": "openai",
  "base_url": "http://localhost:11434/v1",
  "model": "llama3.2:1b",
  "api": "chat"
}'
```

For a hosted endpoint, point `api_key_env` at the *name* of an environment
variable; the arena reads the key from your environment at request time and never
stores it. `model_env` does the same for the model id, so one registry entry
covers every model a provider offers:

```bash
export OPENROUTER_API_KEY=sk-or-...
curl -X POST http://localhost:8100/api/models -H 'Content-Type: application/json' -d '{
  "name": "qwen-72b", "kind": "openai",
  "base_url": "https://openrouter.ai/api/v1", "model": "qwen/qwen-72b-chat",
  "api": "chat", "api_key_env": "OPENROUTER_API_KEY"
}'
```

### Another local GGUF

```bash
curl -X POST http://localhost:8100/api/models -H 'Content-Type: application/json' -d '{
  "name": "llama3.2-1b-q4", "kind": "local",
  "gguf": "/abs/path/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  "port": 8203, "ctx": 2048, "threads": 2
}'
```

Each local model gets its own `llama-server` on its own port, started on demand
and logged to `llama-server-<port>.log`. Remove a contestant with
`DELETE /api/models/<id>`. Remote models unavailable (no key, host down) are
skipped when a battle is drawn, and shown as `offline` in the UI.

## Rating method (and its limits)

* Ratings start at **1000** and move by Elo with **K = 32**:
  `R += K · (score − expected)`, where a win is 1, a tie ½, a loss 0.
* Ties move both models by the same small amount in opposite directions, based
  on the pre-battle expectation.
* **“Both bad” votes are recorded in the battle history but excluded from ratings
  and from win/loss records.** `battles == wins + losses + ties` always holds.
* The **± range** is a 95% bootstrap interval: Bradley-Terry is refit on 200
  resampled vote histories and the 2.5/97.5 percentiles are reported, converted
  to the same 400-point scale. With few votes it stays wide — that is information,
  not a bug.
* Elo is order-dependent and the CI assumes votes are independent; a serious
  arena would use the BT MLE (already implemented for the CIs, in
  `bradley_terry()`) as the headline number. Ratings here are **only meaningful
  within this pool** — they are not comparable to any public leaderboard.

## API

| Route | Purpose |
|---|---|
| `GET /` , `GET /leaderboard` | battle UI and leaderboard UI |
| `GET /api/models` | contestants with Elo, CI, record, availability |
| `POST /api/models` | register a contestant (see above) |
| `DELETE /api/models/<id>` | remove a contestant |
| `POST /api/battle` | `{prompt, max_tokens?, temperature?, top_p?, model_a?, model_b?, category?}` → SSE stream |
| `POST /api/vote` | `{battle_id, winner: "a"\|"b"\|"tie"\|"both_bad"}` → reveal + Elo change |
| `GET /api/leaderboard` | ranked standings |
| `GET /api/battles` | recent battles with both responses |
| `GET /health` | server + per-model status |

The battle stream emits `start` → `chunk` (per side) → `done` (per side) → `end`,
with model identities *not* in the payload until you vote.

## Files

| File | What it does |
|---|---|
| `arena.py` | server: HTTP, SQLite storage, model lifecycle, Elo + bootstrap CI |
| `models.json` | contestants seeded into the DB on first run |
| `static/index.html` | battle UI (blind A/B, streaming, voting, reveal) |
| `static/leaderboard.html` | standings, recent battles, method notes |
| `static/style.css` | shared styling (no CDN, works offline) |
| `arena.db` | your votes and ratings (git-ignored, `--reset` to clear) |

## Notes

* Votes are **yours**; the arena ships with an empty database.
* One battle = one prompt, two generations, one vote. There is no undo API — use
  `--reset` to start a fresh season.
* The UI is deliberately dependency-free (no CDN fonts or scripts) so it works on
  an air-gapped machine.
