#!/usr/bin/env python3
"""
Build phone/iphone.html: a scan-this page for getting the apps onto an iPhone.

    python3 tools/make_qr_page.py                 # uses $E2B_SANDBOX_ID
    python3 tools/make_qr_page.py --host my.host  # or an explicit host

Generates QR codes (SVG, inlined -- no Pillow, no CDN) for:
  * the playground (port 8000)
  * the arena (port 8100)
  * the 0.6 MB phone-app download

Served by both servers at /phone.
"""

from __future__ import annotations

import argparse
import html
import io
import os
from pathlib import Path

import qrcode
from qrcode.image.svg import SvgPathImage

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "phone" / "iphone.html"


def qr_svg(data: str, box: int = 8) -> str:
    qr = qrcode.QRCode(border=2, box_size=box, error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(image_factory=SvgPathImage)
    buf = io.BytesIO()
    img.save(buf)
    svg = buf.getvalue().decode("utf-8")
    # strip the XML prolog; inline SVG must not carry one
    body = svg[svg.index("<svg"):]
    return body.replace('width="', 'class="qr" width="').replace('height="', 'height="', 1)


def card(title: str, url: str, note: str, box: int = 8) -> str:
    return f"""
    <div class="card">
      <div class="qrbox">{qr_svg(url, box)}</div>
      <div class="meta">
        <h3>{html.escape(title)}</h3>
        <p>{note}</p>
        <code>{html.escape(url)}</code>
      </div>
    </div>"""


def build(base_host: str, ports: tuple[int, int], out: Path) -> None:
    playground = f"https://{ports[0]}-{base_host}.e2b.app/"
    arena = f"https://{ports[1]}-{base_host}.e2b.app/"
    lite = f"https://{ports[0]}-{base_host}.e2b.app/LlamaPhoneLite.html"

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
<title>Put this on your iPhone</title>
<meta name="theme-color" content="#0f1115" />
<meta name="color-scheme" content="dark" />
<style>
  :root {{ --bg:#0f1115; --panel:#171a21; --line:#2a2f3a; --text:#e6e8ee;
           --dim:#9aa2b1; --accent:#7cc4ff; --accent-2:#8be9a8; }}
  * {{ box-sizing:border-box; }}
  html {{ -webkit-text-size-adjust:100%; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
          font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
          padding:env(safe-area-inset-top) env(safe-area-inset-right)
                  env(safe-area-inset-bottom) env(safe-area-inset-left); }}
  .wrap {{ max-width:640px; margin:0 auto; padding:20px 16px 60px; }}
  h1 {{ font-size:21px; margin:0 0 6px; }}
  h2 {{ font-size:15px; margin:26px 0 10px; color:var(--accent-2);
        text-transform:uppercase; letter-spacing:.7px; }}
  p {{ color:var(--dim); font-size:14.5px; margin:6px 0; }}
  .card {{ display:flex; gap:14px; align-items:center; background:var(--panel);
           border:1px solid var(--line); border-radius:12px; padding:14px;
           margin-bottom:12px; }}
  .qrbox {{ flex:0 0 auto; background:#fff; border-radius:8px; padding:8px; }}
  .qr {{ display:block; width:132px; height:132px; }}
  .meta h3 {{ margin:0 0 4px; font-size:16px; color:var(--text); }}
  .meta p {{ margin:0 0 6px; font-size:13px; }}
  code {{ background:#1e222b; border:1px solid var(--line); border-radius:5px;
          padding:2px 6px; font-size:11.5px; word-break:break-all; color:var(--dim); }}
  ol {{ padding-left:22px; color:var(--dim); font-size:14.5px; }}
  ol li {{ margin-bottom:8px; }}
  b {{ color:var(--text); }}
  .warn {{ border-left:3px solid #ffb4a2; padding:10px 14px; background:#1b1a1c;
           border-radius:8px; margin:14px 0; }}
  .warn p {{ color:#ffd9d0; }}
  @media (max-width:430px) {{ .card {{ flex-direction:column; align-items:flex-start; }}
                              .qr {{ width:168px; height:168px; }} }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Put this on your iPhone 13</h1>
  <p>Scan a code with the Camera app. It works over cellular — no WiFi, and no
     computer involved.</p>

  <h2>1 · Open it</h2>
  {card("Playground", playground,
        "Type a prompt, get a story. Fast (llama.cpp on the host).", 8)}
  {card("Battle arena", arena,
        "Two models answer the same prompt; you vote; the Elo board updates.", 8)}

  <h2>2 · Make it an app icon</h2>
  <ol>
    <li>Open one of the links above <b>in Safari</b> (not Chrome — only Safari can
        install to the home screen on iPhone).</li>
    <li>Tap the <b>Share</b> button (square with an arrow).</li>
    <li>Scroll down, tap <b>Add to Home Screen</b>, then <b>Add</b>.</li>
  </ol>
  <p>You get an icon and a full-screen window with no Safari chrome — that is what
     the manifest and apple-touch-icon in these apps are for.</p>

  <h2>3 · About offline on iPhone — the honest version</h2>
  <div class="warn">
    <p><b>A downloaded HTML file will not run as an app on iOS.</b> Safari saves it
    to Files, and tapping it there gives you a preview that does not execute the
    page's JavaScript. So the single-file app is an Android/desktop path; on iPhone
    it will just look like a dead page.</p>
  </div>
  <p>Three ways to get real on-device inference on an iPhone:</p>
  <ol>
    <li><b>Keep using Safari</b> with the links above. Your phone needs a
        connection to reach the host, but the phone itself does no heavy lifting —
        great on cellular, useless in airplane mode.</li>
    <li><b>PocketPal AI</b> (free, App Store) — runs GGUF models locally on the
        iPhone. Download a model once on WiFi, then it works in airplane mode.
        Your A15 with 4 GB RAM handles a 1B model at Q4 comfortably. This is the
        best "genuinely offline on iPhone" answer.</li>
    <li><b>Pyto</b> (paid) — a Python IDE for iOS with numpy. It can run a local
        HTTP server, so you can browse to it at <code>localhost</code> in Safari.
        That is the closest thing to running this repo's server on the phone. I
        have not tested this from here, so treat it as a lead rather than a
        guarantee.</li>
  </ol>

  <h2>Optional · the small download</h2>
  {card("LlamaPhoneLite.html", lite,
        "0.6 MB page for Android/desktop browsers. On iOS it saves to Files but "
        "will not execute — see above.", 6)}
</div>
</body>
</html>
"""
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("E2B_SANDBOX_ID", ""),
                    help="sandbox host id (defaults to $E2B_SANDBOX_ID)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if not args.host:
        raise SystemExit("need --host or $E2B_SANDBOX_ID")
    build(args.host, (8000, 8100), args.out)


if __name__ == "__main__":
    main()
