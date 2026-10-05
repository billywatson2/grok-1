#!/usr/bin/env python3
"""
Generate the PWA icons for both apps (pure numpy + zlib, no Pillow needed).

    python3 tools/make_pwa_icons.py

Writes icon-{192,512}.png, icon-maskable-512.png and apple-touch-icon-180.png
into lm_arena/static/ and local_llama/web/.

The mark is a small podium: two contestants (blue A, pink B) standing on a base,
which is what the arena is. Everything is drawn at 4x and box-downsampled for
antialiasing.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TARGETS = [ROOT / "lm_arena" / "static", ROOT / "local_llama" / "web"]

BG = (0x0F, 0x11, 0x15, 0xFF)
BAR_A = (0x7C, 0xC4, 0xFF, 0xFF)
BAR_B = (0xF0, 0xA3, 0xC8, 0xFF)
BASE = (0x8B, 0xE9, 0xA8, 0xFF)

SS = 4  # supersampling factor


def rounded_rect(dim: int, x0: float, y0: float, x1: float, y1: float,
                 radius: float) -> np.ndarray:
    """Coverage mask (0..1) of a rounded rectangle, antialiased."""
    size = dim * SS
    xs = (np.arange(size) + 0.5) / SS
    ys = (np.arange(size) + 0.5) / SS
    X, Y = np.meshgrid(xs, ys)

    inside_box = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    dx = np.maximum(np.maximum(x0 + radius - X, X - (x1 - radius)), 0.0)
    dy = np.maximum(np.maximum(y0 + radius - Y, Y - (y1 - radius)), 0.0)
    inside = inside_box & (np.sqrt(dx * dx + dy * dy) <= radius)
    return inside.reshape(dim, SS, dim, SS).mean(axis=(1, 3))


def draw_icon(dim: int, scale: float = 1.0, transparent_bg: bool = False) -> np.ndarray:
    """RGBA icon; `scale` shrinks the artwork toward the centre (maskable safe zone)."""
    # masks arrive already antialiased (downsampled from SS), so the canvas is
    # at final resolution and `alpha` broadcasts as (dim, dim, 1)
    canvas = np.zeros((dim, dim, 4), dtype=np.float64)
    canvas[..., 0] = BG[0]
    canvas[..., 1] = BG[1]
    canvas[..., 2] = BG[2]
    canvas[..., 3] = 0.0 if transparent_bg else 255.0

    def place(mask: np.ndarray, color: tuple[int, int, int, int]) -> None:
        # mask is (dim, dim) so it broadcasts directly against a channel
        for c in range(3):
            canvas[..., c] = canvas[..., c] * (1 - mask) + color[c] * mask
        canvas[..., 3] = np.maximum(canvas[..., 3], mask * 255.0)

    u = dim / 512.0            # design was drawn on a 512 grid
    def R(x0, y0, x1, y1, r):
        # scale about the centre, then convert to pixels
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        x0, x1 = cx + (x0 - cx) * scale, cx + (x1 - cx) * scale
        y0, y1 = cy + (y0 - cy) * scale, cy + (y1 - cy) * scale
        return rounded_rect(dim, x0 * u, y0 * u, x1 * u, y1 * u, r * scale * u)

    if not transparent_bg:
        place(rounded_rect(dim, 0, 0, float(dim) - 0.01, float(dim) - 0.01, dim * 0.22), BG)

    place(R(112, 140, 224, 368, 26), BAR_A)   # contestant A
    place(R(288, 168, 400, 368, 26), BAR_B)   # contestant B (slightly shorter = "ranking")
    place(R(96, 392, 416, 440, 22), BASE)     # the podium base

    return canvas.clip(0, 255).astype(np.uint8)


def write_png(path: Path, rgba: np.ndarray) -> None:
    height, width, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)
    print(f"  {path.relative_to(ROOT)}  ({len(png) / 1024:.1f} KB, {width}x{height})")


def main() -> None:
    for target in TARGETS:
        target.mkdir(parents=True, exist_ok=True)
        print(f"{target.relative_to(ROOT)}/")
        write_png(target / "icon-192.png", draw_icon(192))
        write_png(target / "icon-512.png", draw_icon(512))
        # maskable: full-bleed background, artwork inside the 80% safe zone
        write_png(target / "icon-maskable-512.png", draw_icon(512, scale=0.66))
        write_png(target / "apple-touch-icon-180.png",
                  draw_icon(180, scale=0.92, transparent_bg=False))


if __name__ == "__main__":
    main()
