#!/usr/bin/env python3
"""
Scan the repository for common de-anonymization risks.

Checks
- Non-ASCII CJK characters
- Email-like patterns
- URLs (http/https)
- IP addresses
- Common absolute path prefixes
- Optional custom keywords

Usage:
  python scripts/sanitize_scan.py --root . --fail_on_findings

Tip:
- Run this *before* pushing to any remote repository.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple


RE_CJK = re.compile(r"[\u4e00-\u9fff]")
RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RE_URL = re.compile(r"https?://")
RE_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
RE_WIN_ABS = re.compile(r"\b[A-Za-z]:\\")
ABS_PREFIXES = ("/home/", "/Users/", "/mnt/", "/data/", "/workspace/", "/srv/", "/var/")


def iter_text_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if ".git" in p.parts:
            continue
        # Do not scan the scanner itself (avoids false positives).
        if p.name == "sanitize_scan.py":
            continue
        # Skip binary-like extensions
        if p.suffix.lower() in {".pt", ".pth", ".npz", ".png", ".jpg", ".jpeg", ".pdf", ".zip", ".pptx", ".docx"}:
            continue
        yield p


def scan_file(path: Path, patterns: List[Tuple[str, re.Pattern]]) -> List[Tuple[str, int, str]]:
    findings = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return findings
    for i, line in enumerate(text, start=1):
        for name, pat in patterns:
            if pat.search(line):
                findings.append((name, i, line.strip()))
    # Absolute prefix check (string-based)
    for i, line in enumerate(text, start=1):
        for pref in ABS_PREFIXES:
            if pref in line:
                findings.append(("ABS_PATH", i, line.strip()))
    return findings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--keywords", type=str, nargs="*", default=[])
    ap.add_argument("--fail_on_findings", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    patterns: List[Tuple[str, re.Pattern]] = [
        ("CJK", RE_CJK),
        ("EMAIL", RE_EMAIL),
        ("URL", RE_URL),
        ("IP", RE_IP),
        ("WIN_ABS_PATH", RE_WIN_ABS),
    ]
    for kw in args.keywords:
        if kw.strip():
            patterns.append((f"KW:{kw}", re.compile(re.escape(kw), flags=re.IGNORECASE)))

    any_findings = False
    for p in iter_text_files(root):
        f = scan_file(p, patterns)
        if not f:
            continue
        any_findings = True
        print(f"\n[findings] {p.relative_to(root)}")
        for name, lineno, line in f[:30]:
            print(f"  - {name} line {lineno}: {line}")
        if len(f) > 30:
            print(f"  ... {len(f) - 30} more lines omitted")

    if not any_findings:
        print("No findings detected.")
        return

    if args.fail_on_findings:
        raise SystemExit("Scan finished with findings.")


if __name__ == "__main__":
    main()
