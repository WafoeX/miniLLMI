#!/usr/bin/env python3
"""Record the build's code tree, excluding generated results and baseline report."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True).stdout


def is_code(path):
    return not (path.startswith("results/") or path == "docs/baseline.md")


def inspect(root):
    root = Path(root).resolve()
    try:
        commit = git(root, "rev-parse", "HEAD").decode().strip()
    except subprocess.CalledProcessError:
        commit = "UNCOMMITTED"
    paths = sorted({p.decode("utf-8") for p in
                    git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0")
                    if p and is_code(p.decode("utf-8"))})
    digest = hashlib.sha256()
    for name in paths:
        p = root / name
        digest.update(name.encode() + b"\0")
        if p.is_symlink():
            digest.update(b"symlink\0" + os.readlink(p).encode())
        elif p.is_file():
            digest.update(p.read_bytes())
        else:
            digest.update(b"DELETED")
        # Include executable bits (shell scripts must remain executable).
        digest.update(b"\0exec=" + str(p.stat().st_mode & 0o111 if p.exists() else 0).encode() + b"\0")
    dirty_paths = set()
    for args in [("diff", "--name-only", "-z", "HEAD"),
                 ("ls-files", "--others", "--exclude-standard", "-z")]:
        try:
            dirty_paths.update(p.decode() for p in git(root, *args).split(b"\0") if p)
        except subprocess.CalledProcessError:
            dirty_paths.update(paths)
    return {"commit": commit, "source_digest": digest.hexdigest(),
            "source_dirty": commit == "UNCOMMITTED" or any(is_code(p) for p in dirty_paths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--header", type=Path)
    args = parser.parse_args()
    data = inspect(args.root)
    if args.header:
        text = ("#pragma once\nnamespace stage0 {\n"
                f"inline constexpr const char* kCommit = {json.dumps(data['commit'])};\n"
                f"inline constexpr const char* kSourceDigest = {json.dumps(data['source_digest'])};\n"
                f"inline constexpr bool kSourceDirty = {'true' if data['source_dirty'] else 'false'};\n"
                "}\n")
        args.header.parent.mkdir(parents=True, exist_ok=True)
        if not args.header.exists() or args.header.read_text() != text:
            args.header.write_text(text)
    else:
        print(json.dumps(data, sort_keys=True))


if __name__ == "__main__":
    main()
