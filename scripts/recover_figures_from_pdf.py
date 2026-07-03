"""Recover lost figure PNGs from the committed (pre-restructure) main.pdf.

Aligns the OLD-source \\includegraphics sequence to the PDF's embedded raster
stream with dynamic programming.

Match rules per (ref i, embedded j):
  - ref exists on disk       -> match only on exact pixel-size equality (+2)
  - ref missing / vector pdf -> wildcard match (+1)
Gap on ref side (ref emitted no separate raster, e.g. vector .pdf or a
deduplicated repeat): -1.  Gap on embedded side: -3 (should not occur).

Prints the alignment; writes NOTHING unless --write is passed. Never
overwrites an existing file.
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import sys

import fitz
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "paper", "main_preRestructure_backup.pdf")


def git_show(path: str) -> str:
    out = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"git show failed for {path}")
    return out.stdout.decode("utf-8", errors="replace")


def strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def old_document_source() -> str:
    def expand(tex: str) -> str:
        return re.sub(
            r"\\input\{([^}]+)\}",
            lambda m: expand(strip_comments(git_show(
                f"paper/{m.group(1)}{'' if m.group(1).endswith('.tex') else '.tex'}"))),
            tex,
        )
    return expand(strip_comments(git_show("paper/main.tex")))


def main() -> int:
    write = "--write" in sys.argv

    refs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                      old_document_source())
    disk_size = {}
    for r in refs:
        p = os.path.normpath(os.path.join(ROOT, r.replace("../", "")))
        if os.path.exists(p) and p.lower().endswith(".png"):
            with Image.open(p) as im:
                disk_size[r] = im.size

    doc = fitz.open(PDF)
    emb = []
    for page in doc:
        seen = set()
        for info in page.get_image_info(xrefs=True):
            x = info["xref"]
            if x == 0 or x in seen:
                continue
            seen.add(x)
            emb.append((page.number + 1, x, info["width"], info["height"]))

    n, m = len(refs), len(emb)
    print(f"refs={n} embedded={m} on-disk-known-size={len(disk_size)}")

    NEG = -10**9
    score = [[NEG] * (m + 1) for _ in range(n + 1)]
    move = [[None] * (m + 1) for _ in range(n + 1)]
    score[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            s = score[i][j]
            if s == NEG:
                continue
            if i < n and j < m:
                r = refs[i]
                pg, x, w, h = emb[j]
                if r in disk_size:
                    ms = 2 if disk_size[r] == (w, h) else NEG
                else:
                    ms = 1
                if ms != NEG and s + ms > score[i + 1][j + 1]:
                    score[i + 1][j + 1] = s + ms
                    move[i + 1][j + 1] = "M"
            if i < n and s - 1 > score[i + 1][j]:
                score[i + 1][j] = s - 1
                move[i + 1][j] = "R"
            if j < m and s - 3 > score[i][j + 1]:
                score[i][j + 1] = s - 3
                move[i][j + 1] = "E"

    if score[n][m] == NEG:
        print("NO VALID ALIGNMENT")
        return 1

    pairs, i, j = [], n, m
    while i or j:
        mv = move[i][j]
        if mv == "M":
            pairs.append((i - 1, j - 1)); i, j = i - 1, j - 1
        elif mv == "R":
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    pairs.reverse()

    n_val = n_wild = 0
    to_write = []
    for pi, pj in pairs:
        if pi is None:
            pg, x, w, h = emb[pj]
            print(f"  [EMB-ONLY]           p{pg:3d} {w}x{h}")
            continue
        r = refs[pi]
        if pj is None:
            print(f"  [NO-RASTER] {r}")
            continue
        pg, x, w, h = emb[pj]
        if r in disk_size:
            n_val += 1
        else:
            n_wild += 1
            tag = "RECOVER" if r.lower().endswith(".png") else "vector "
            print(f"  [{tag}] p{pg:3d} {w}x{h}  {r}")
            if tag == "RECOVER":
                to_write.append((r, pj))
    print(f"alignment: size-validated={n_val}  wildcards={n_wild}  "
          f"score={score[n][m]}")

    if not write:
        print("(dry run; pass --write to save recovered figures)")
        return 0

    for r, pj in to_write:
        pg, x, w, h = emb[pj]
        img = doc.extract_image(x)
        base = Image.open(io.BytesIO(img["image"])).convert("RGB")
        if img.get("smask"):
            mimg = doc.extract_image(img["smask"])
            alpha = Image.open(io.BytesIO(mimg["image"])).convert("L")
            base = base.convert("RGBA"); base.putalpha(alpha)
        p = os.path.normpath(os.path.join(ROOT, r.replace("../", "")))
        if os.path.exists(p):
            print(f"  SKIP (exists): {r}")
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        base.save(p, "PNG")
        print(f"  wrote {os.path.relpath(p, ROOT)}  ({w}x{h}, p{pg})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
