# /// script
# requires-python = ">=3.10"
# dependencies = ["Markdown==3.8.2", "pymdown-extensions==10.16.1"]
# ///
"""Regression checks for Milestone 3 resource URLs and repeatable builds.

Run: uv run --no-project scripts/check_milestone_three.py
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import build_milestone_three as builder


def manifest(directory: Path) -> dict[str, bytes]:
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in directory.rglob("*") if p.is_file()}


def check() -> None:
    with TemporaryDirectory(prefix="milestone-three-check-") as temporary:
        output = Path(temporary) / "site"
        builder.build_milestone_three(output)
        first = manifest(output)
        assert len(json.loads((output / "search.json").read_text(encoding="utf-8"))) == 11
        for url in (*builder.URLS, *builder.LEGACY_GUIDES, "walkthrough-notebook.html"):
            page = (output / url).read_text(encoding="utf-8")
            assert 'href="../index.html"' in page
            assert 'href="../../index.html"' in page
        for url in builder.LEGACY_GUIDES:
            assert "previously shared URL belongs to the earlier" in (output / url).read_text(encoding="utf-8")

        # An incremental build must remove stale resources and preserve all URLs.
        builder.build_milestone_three(output)
        assert manifest(output) == first, "Clean and repeated builds differ"

    print("PASS: clean/repeated builds, search index, legacy routes and navigation links")


if __name__ == "__main__":
    check()
