"""Draw the README hero, assets/hero-dark.svg and assets/hero-light.svg, from assets/hero.txt.

The hero is character art: NOISE VAULT in block letters, and a noisy trace that runs flat
after a pin. Every character becomes geometry on one fixed grid, so the image is the same in
every browser whatever fonts it has. Run from the repository root:

    python scripts/build_hero.py           # rewrite both SVGs
    python scripts/build_hero.py --check   # exit 1 if either SVG is out of date
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
GRID = ASSETS / "hero.txt"

CELL_W, CELL_H = 10, 20
BLOCK_GAP = 3  # blank pixels under each row of blocks, so the letters keep their text rows
STROKE = 3.6
DOT_R = 6.5
PAD = 8

Point = tuple[float, float]

# Each stroke character as a segment in a unit cell, (0, 0) at the top left, left end first.
SEGMENTS: dict[str, tuple[Point, Point]] = {
    "/": ((0, 1), (1, 0)),
    "\\": ((0, 0), (1, 1)),
    "_": ((0, 1), (1, 1)),
    "─": ((0, 0.5), (1, 0.5)),
    "│": ((0.5, 0), (0.5, 1)),
    "●": ((0.5, 0.5), (0.5, 1)),  # the pin's own cell carries the top of its stem
}


@dataclass(frozen=True)
class Palette:
    mark_from: str  # wordmark gradient, left end
    mark_to: str  # wordmark gradient, right end
    noise: str  # the trace before the pin
    pin: str  # the pin dot
    held: str  # the stem, the flat line and the halo around the pin


PALETTES = {
    "dark": Palette("#3fd0c9", "#6c9eff", "#ffb547", "#f4f6fa", "#6c9eff"),
    "light": Palette("#0b8a84", "#2f5fe0", "#c0630a", "#0d1117", "#2f5fe0"),
}


@dataclass(frozen=True)
class Hero:
    width: int
    height: int
    blocks: list[tuple[int, int, int]]  # (row, first column, run length)
    mark_span: tuple[int, int]  # pixel x of the wordmark's left and right edges
    noise: list[list[Point]]  # polylines left of the pin
    held: list[list[Point]]  # polylines from the pin rightwards
    pin: Point


def parse(rows: list[str]) -> Hero:
    pins = [(c, r) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "●"]
    if len(pins) != 1:
        raise ValueError(f"{GRID.name} needs exactly one ● pin, found {len(pins)}")
    pin_col, pin_row = pins[0]
    blocks, noise, held = [], [], []
    for r, row in enumerate(rows):
        c = 0
        while c < len(row):
            ch = row[c]
            if ch == "█":
                end = c
                while end < len(row) and row[end] == "█":
                    end += 1
                blocks.append((r, c, end - c))
                c = end
                continue
            if ch in SEGMENTS:
                (ax, ay), (bx, by) = SEGMENTS[ch]
                seg = (
                    (CELL_W * (c + ax), CELL_H * (r + ay)),
                    (CELL_W * (c + bx), CELL_H * (r + by)),
                )
                (noise if c < pin_col else held).append(seg)
            elif ch != " ":
                raise ValueError(f"{GRID.name} row {r + 1}: no drawing for {ch!r}")
            c += 1
    left = min(c for _, c, _ in blocks)
    right = max(c + n for _, c, n in blocks)
    return Hero(
        width=CELL_W * max(len(row) for row in rows),
        height=CELL_H * len(rows),
        blocks=blocks,
        mark_span=(CELL_W * left, CELL_W * right),
        noise=chain(noise),
        held=chain(held),
        pin=(CELL_W * (pin_col + 0.5), CELL_H * (pin_row + 0.5)),
    )


def chain(segments: list[tuple[Point, Point]]) -> list[list[Point]]:
    """Join segments left to right into polylines, and merge straight runs into one leg.

    Neighbouring strokes meet at a shared cell edge, but │ stands in the middle of its cell. So a
    start within half a cell of the current end, on the same line, continues the polyline.
    """
    lines: list[list[Point]] = []
    for a, b in sorted(segments):
        line = lines[-1] if lines else None
        if line and a[1] == line[-1][1] and 0 <= a[0] - line[-1][0] <= CELL_W / 2:
            if len(line) > 1 and straight(line[-2], line[-1], b):
                line[-1] = b
            else:
                line.append(b)
        else:
            lines.append([a, b])
    return lines


def straight(a: Point, b: Point, c: Point) -> bool:
    (ux, uy), (vx, vy) = (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])
    return ux * vy == uy * vx and ux * vx + uy * vy > 0


def num(x: float) -> str:
    return f"{x:g}"


def path(lines: list[list[Point]]) -> str:
    return "".join(
        "M" + "L".join(f"{num(x + PAD)} {num(y + PAD)}" for x, y in line) for line in lines
    )


def svg(hero: Hero, palette: Palette) -> str:
    x0, x1 = hero.mark_span
    rects = "".join(
        f'<rect x="{c * CELL_W + PAD}" y="{r * CELL_H + PAD}" width="{n * CELL_W}" '
        f'height="{CELL_H - BLOCK_GAP}"/>'
        for r, c, n in hero.blocks
    )
    px, py = (num(v + PAD) for v in hero.pin)
    stroke = (
        f'fill="none" stroke-width="{num(STROKE)}" stroke-linecap="round" stroke-linejoin="round"'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {hero.width + 2 * PAD} '
        f'{hero.height + 2 * PAD}" role="img" aria-labelledby="t">'
        '<title id="t">NoiseVault</title>'
        f'<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="{x0 + PAD}" y1="0" '
        f'x2="{x1 + PAD}" y2="0"><stop offset="0" stop-color="{palette.mark_from}"/>'
        f'<stop offset="1" stop-color="{palette.mark_to}"/></linearGradient></defs>'
        f'<g fill="url(#g)" shape-rendering="crispEdges">{rects}</g>'
        f'<path d="{path(hero.noise)}" stroke="{palette.noise}" {stroke}/>'
        f'<path d="{path(hero.held)}" stroke="{palette.held}" {stroke}/>'
        f'<circle cx="{px}" cy="{py}" r="{num(DOT_R * 2)}" fill="{palette.held}" opacity="0.3"/>'
        f'<circle cx="{px}" cy="{py}" r="{num(DOT_R)}" fill="{palette.pin}"/>'
        "</svg>\n"
    )


def outputs() -> dict[Path, str]:
    hero = parse(GRID.read_text(encoding="utf-8").splitlines())
    return {ASSETS / f"hero-{name}.svg": svg(hero, p) for name, p in PALETTES.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if an SVG is out of date")
    args = parser.parse_args()
    files = outputs()
    if args.check:
        stale = [p for p, text in files.items() if not p.exists() or p.read_text("utf-8") != text]
        for p in stale:
            print(f"{p.relative_to(ROOT)} is out of date. Run python scripts/build_hero.py")
        if not stale:
            print("up to date")
        return 1 if stale else 0
    for p, text in files.items():
        p.write_text(text, encoding="utf-8")
        print(f"wrote {p.relative_to(ROOT)} ({len(text.encode())} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
