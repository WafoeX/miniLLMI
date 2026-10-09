#!/usr/bin/env python3
"""Check repository-local Markdown links used by the Stage 19 documentation audit."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
SKIP_PREFIXES = ("#", "http://", "https://", "mailto:", "data:")


def local_target(raw: str) -> str | None:
    target = raw.strip().split(maxsplit=1)[0].strip("<>")
    if not target or target.startswith(SKIP_PREFIXES):
        return None
    return unquote(target.split("#", 1)[0])


def main() -> int:
    paths = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    failures: list[str] = []
    checked = 0
    for document in paths:
        text = document.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            target = local_target(match.group(1))
            if target is None:
                continue
            checked += 1
            candidate = (document.parent / target).resolve()
            try:
                candidate.relative_to(ROOT)
            except ValueError:
                failures.append(f"{document.relative_to(ROOT)} escapes repository: {target}")
                continue
            if not candidate.exists():
                failures.append(f"{document.relative_to(ROOT)} missing target: {target}")
    if failures:
        print("documentation link check failed:", file=sys.stderr)
        print("\n".join(f"- {failure}" for failure in failures), file=sys.stderr)
        return 1
    print(f"documentation link check passed: {len(paths)} files, {checked} local links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
