# Running it on your phone (no computer, no WiFi)

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
