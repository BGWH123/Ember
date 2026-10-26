#!/usr/bin/env python3
"""Re-render every tutorial HTML from its Markdown with the CURRENT academic
template, preserving each file's existing hero metadata.

Why this exists
---------------
The per-file ``--title / --subtitle / --eyebrow / --author / --lang`` that
produced each tutorial HTML were passed as CLI flags at render time and survive
ONLY inside the rendered HTML — there is no manifest and the tutorial Markdown
carries no frontmatter. So a faithful "template bump" re-render (propagate a
change in tools/templates/academic.html to all tutorials WITHOUT altering their
content) must recover that metadata from the existing HTML. This harness does
that, re-renders via tools/render_html.py, and writes a manifest so future
re-renders are deterministic.

It deliberately PRESERVES whatever is there today, warts and all (e.g. the EN
editions' auto-generated titles) — fixing those is separate content work, not a
dormant shell bump.

Usage:
    python3 tools/rerender_tutorials.py            # re-render all, write manifest
    python3 tools/rerender_tutorials.py --dry-run  # show what would run, no writes
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RENDER = ROOT / "tools" / "render_html.py"
TUT_DIR = ROOT / "docs" / "tutorials"
MANIFEST = ROOT / "tools" / "tutorials_render_manifest.json"

_PATTERNS = {
    "title": re.compile(r"<title>(.*?)</title>", re.DOTALL),
    "eyebrow": re.compile(r'<div class="eyebrow">(.*?)</div>', re.DOTALL),
    "subtitle": re.compile(r'<p class="subtitle">(.*?)</p>', re.DOTALL),
    "author": re.compile(r'<p class="byline">By <strong>(.*?)</strong></p>', re.DOTALL),
    "lang": re.compile(r'<html lang="([^"]*)"'),
}


def extract_meta(html_text: str) -> dict:
    """Pull hero metadata out of a rendered HTML. Values are HTML-UNESCAPED so
    that re-feeding them through render_html.py (which escapes again) round-trips
    cleanly instead of double-escaping."""
    meta: dict = {}
    for key, rx in _PATTERNS.items():
        m = rx.search(html_text)
        if m:
            val = html_lib.unescape(m.group(1).strip())
            if val:
                meta[key] = val
    return meta


def build_cmd(md_path: Path, html_path: Path, meta: dict) -> list[str]:
    cmd = [
        sys.executable, str(RENDER), str(md_path),
        "--template", "academic",
        "--out", str(html_path),
        "--lang", meta.get("lang", "zh-CN"),
    ]
    if meta.get("title"):
        cmd += ["--title", meta["title"]]
    if meta.get("eyebrow"):
        cmd += ["--eyebrow", meta["eyebrow"]]
    if meta.get("subtitle"):
        cmd += ["--subtitle", meta["subtitle"]]
    if meta.get("author"):
        cmd += ["--author", meta["author"]]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description="Faithfully re-render all tutorial HTMLs (template bump).")
    ap.add_argument("--dry-run", action="store_true", help="print planned renders without running")
    args = ap.parse_args()

    htmls = sorted(TUT_DIR.glob("*.html"))
    if not htmls:
        print(f"no tutorial HTMLs under {TUT_DIR}", file=sys.stderr)
        return 1

    manifest: dict = {}
    ok = skipped = failed = 0
    for html_path in htmls:
        md_path = html_path.with_name(html_path.name[: -len(".html")] + ".md")
        rel_html = html_path.relative_to(ROOT).as_posix()
        if not md_path.is_file():
            print(f"  SKIP {rel_html} — no sibling .md ({md_path.name})")
            skipped += 1
            continue
        meta = extract_meta(html_path.read_text(encoding="utf-8", errors="replace"))
        cmd = build_cmd(md_path, html_path, meta)
        manifest[rel_html] = {"md": md_path.relative_to(ROOT).as_posix(), **meta}

        if args.dry_run:
            shown = " ".join(f'"{c}"' if " " in c else c for c in cmd[2:])
            print(f"  would render {rel_html}\n      {shown}")
            ok += 1
            continue

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            ok += 1
            tag = meta.get("title", "?")
            print(f"  OK   {rel_html}  [{meta.get('lang','zh-CN')}] {tag[:42]}")
        else:
            failed += 1
            print(f"  FAIL {rel_html}\n      {res.stderr.strip()[:300]}")

    if not args.dry_run:
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nmanifest -> {MANIFEST.relative_to(ROOT).as_posix()} ({len(manifest)} entries)")
    print(f"\nsummary: {ok} rendered · {skipped} skipped · {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
