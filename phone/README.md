# Running it on your phone (no computer, no WiFi)

Two ways to do this, depending on how much you want to install:

| | **Single-file app** (easiest) | **Termux apparatus** |
|---|---|---|
| Setup | copy one file to the phone | install Termux + run a script |
| Runs | in your normal browser, in airplane mode | a local server on the phone |
| Model | stories15M, int8 (15 MB, baked in) | stories15M + any GGUF you add |
| Extras | none | arena, voting, leaderboard, OpenAI API |
| Start at | [Single-file app](#the-single-file-app-no-install) | [Android — exact steps](#android--exact-steps) |

## The single-file app (no install)

`phone/LlamaPhone.html` is the whole thing — a Llama model, its tokenizer and the
inference engine in one 21 MB HTML file. Copy it to the phone (USB, email, cloud,
messaging yourself), tap it, and it opens in Chrome. Turn on airplane mode and it
keeps working: there is no server, no install, and it never opens a socket.

```bash
# build it from a checkout (needs the checkpoint: local_llama/download_model.sh)
python3 tools/build_phone_app.py
#   -> phone/LlamaPhone.html   21.1 MB, self-contained
#   -> phone/LlamaPhone.llm    15.4 MB, the raw quantised container
```

You can also build it **on the phone** in Termux after `termux_setup.sh`, since
that already fetches the checkpoint:

```bash
cd ~/grok-1 && python3 tools/build_phone_app.py
```

Then open `file:///sdcard/Download/LlamaPhone.html` (or wherever you copied it).

### What is actually in it

* **Model**: stories15M — the same Llama-2 architecture checkpoint the rest of
  this repo uses — quantised to int8 with one scale per row: 15.4 MB instead of
  60.8 MB. Measured against the full-precision model over 80 teacher-forced
  steps: mean |Δlogit| 0.066, max 0.37, **top-1 agreement 100%**. Turning the
  dial to 4-bit halves the file again (7.8 MB) but drops agreement to 74% *and*
  runs ~3× slower in JavaScript, so int8 is the default for good reason.
* **Engine**: `phone/web/llama-engine.js` — plain JavaScript, no WebAssembly, no
  build toolchain. It mirrors `local_llama/np_llama.py` operation for operation.
* **Proof it is correct**: `node phone/test_engine.js phone/LlamaPhone.llm
  "Once upon a time" 24` prints greedy token ids that match the Python reference
  **exactly** (both engines produce `29892,727,471,263,2217,...`). If you change
  the engine, re-run that before trusting it.

### Speed

Roughly 50 tok/s per core in Node on this machine. A phone browser is slower —
expect a handful of tokens per second to a few tens, which is fine for stories
and is the honest cost of having no server at all.

---

# The Termux apparatus

The arena and the playground run **on the phone itself**. The models are 60 MB
and 33 MB, so a phone that is a few years old handles them easily; you do not
need a laptop, and after setup nothing touches the network — the server listens
on `127.0.0.1`, which is only reachable from the phone.

```
   ┌──────────────────────── your phone ────────────────────────┐
   │  Termux            localhost:8100            Chrome        │
   │  ┌──────────┐      ┌───────────────┐        ┌───────────┐  │
   │  │ arena.py │◄────►│  llama-server │◄──────►│  Arena    │  │
   │  │ + SQLite │      │  (or numpy)   │ local  │  (PWA)    │  │
   │  └──────────┘      └───────────────┘  only  └───────────┘  │
   └────────────────────────────────────────────────────────────┘
                    no WiFi · no data · no accounts
```

## Android — exact steps

1. **Install Termux from F-Droid**, not the Play Store (the Play Store build is
   abandoned and its package mirrors are broken):
   <https://f-droid.org/packages/com.termux/>

2. **Get the code onto the phone.** Any one of these works:

   ```bash
   # a) git, if the repo is pushed (see note at the bottom)
   pkg install -y git
   git clone https://github.com/<you>/grok-1.git -b arena/01a10a28-grok-1
   cd grok-1

   # b) copy it across once, from the computer
   #    (from the computer:  adb push grok-1 /sdcard/Download/   — or any file
   #     manager / USB transfer), then in Termux:
   termux-setup-storage          # grant access to shared storage once
   cp -r /sdcard/Download/grok-1 ~/grok-1 && cd ~/grok-1
   ```

3. **Run the setup** (from the repository root):

   ```bash
   bash phone/termux_setup.sh
   ```

   That installs python + numpy, downloads the models with checksum
   verification, and stops there — the NumPy backend needs no compiler and works
   immediately. Add `--build` if you want llama.cpp too (5–20 minutes on a phone,
   roughly 5× faster afterwards), and `--model-1b` for a much better 1B model
   (~800 MB, needs a Hugging Face connection).

4. **Start the arena:**

   ```bash
   cd ~/grok-1/lm_arena
   termux-wake-lock                    # keeps Android from suspending the server
   python3 arena.py --port 8100 --host 127.0.0.1
   ```

5. **Open `http://localhost:8100` in Chrome**, then ⋮ → **Add to Home Screen**.
   It installs as an app (own icon, no browser chrome) and works offline; votes
   are stored in `lm_arena/arena.db` on the phone.

The playground is the same shape:

```bash
cd ~/grok-1/local_llama && python3 serve.py --port 8000
# → http://localhost:8000
```

## Which backend on a phone?

| | NumPy backend (`--backend numpy`) | llama.cpp backend |
|---|---|---|
| Setup | instant — numpy only | `--build` (5–20 min) |
| Speed, 15M model | a few tokens/sec slower, still usable | noticeably snappier |
| RAM | ~250 MB | ~50 MB |

Start with NumPy; build later if you like it. `arena.py` picks llama.cpp
automatically when the binaries exist and falls back to NumPy when they do not —
no flags to remember.

## Keeping it alive in the background

* `termux-wake-lock` before starting the server, `termux-wake-unlock` when done.
* Android may still kill Termux; in **Settings → Apps → Termux → Battery**, set
  it to *Unrestricted*.
* To keep a session alive while you close the Termux window, run the server under
  `tmux` (`pkg install tmux`, then `tmux new -s arena`).
* The service worker caches the app shell, so if the server restarts you can
  still open the app and you will see live data again as soon as it is up.

## iPhone / iPad

Termux is Android-only, so the "everything on the phone" path is different:

* For chat with on-device models, apps like **PocketPal AI** or **LLM Farm** run
  GGUF models locally and overlap with what the playground does.
* There is no supported way to run a Python server in the background on iOS
  without a Mac and Xcode, so the arena UI is the piece that has to live
  elsewhere — see the tunnel option below, then add the site to your Home Screen
  (Safari → Share → *Add to Home Screen*); the PWA works on iOS too.

## If you *do* want it off the phone (tunnels)

For using it from another device — your phone while the model runs on a desktop,
or a laptop while it runs on the phone — something has to carry the traffic.
Whatever you choose, **use a token**:

```bash
python3 arena.py --port 8100 --host 0.0.0.0 --token "pick-something-long"
# open once with ?token=pick-something-long -- it is then remembered as a cookie
```

Without a token, anyone who can reach the port can spend your CPU, cast votes and
delete contestants through the API. Options, cheapest first:

* **Phone as the server:** run the arena on the phone with `--host 0.0.0.0`,
  then open `http://<phone-ip>:8100` from another device *on the same WiFi*.
  The phone does the work; the other device is just a screen.
* **Tailscale** (`pkg install tailscale` in Termux, or the Android app): gives
  every device a private address with real encryption, no port forwarding.
  This is the safest way to reach it from outside your home.
* **Cloudflare Tunnel / ngrok:** `cloudflared tunnel --url http://localhost:8100`
  publishes it to a random public HTTPS URL. Convenient, but public — the token is
  not optional here.

## Which path should you pick?

* Want to read stories on a plane or in a tunnel, with nothing installed? →
  **single-file app**.
* Want the A/B arena, voting, an Elo leaderboard, bigger GGUF models, or an
  OpenAI-compatible endpoint for other apps? → **Termux**.

Both are fully offline and neither needs a computer while you use them.

## Notes and caveats

* Setup needs internet **once** to download models and packages. After that the
  arena is fully local: put the phone in airplane mode and it still works.
* The 15M models write children's stories in simple English. They are a real
  Llama forward pass, but they are not assistants — for chat quality on a phone,
  the 1B model in step 3 is the honest upgrade, and `models.json` is where you
  add it as a contestant.
* Ratings live in `lm_arena/arena.db`. Back it up by copying that one file.
* PWA install and service workers need `localhost` or HTTPS. From a LAN IP over
  plain `http://`, the app still works — you just lose offline caching and the
  install prompt.
