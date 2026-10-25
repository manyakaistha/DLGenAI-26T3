# /// script
# requires-python = ">=3.10"
# dependencies = ["Markdown==3.8.2", "pymdown-extensions==10.16.1"]
# ///
"""Regression checks for Milestone 2 resource URLs and repeatable builds.

Run: uv run --no-project scripts/check_milestone_two.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import quote

import build_milestone_two as builder


def manifest(directory: Path) -> dict[str, bytes]:
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in directory.rglob("*") if p.is_file()}


def check() -> None:
    with TemporaryDirectory(prefix="milestone-two-check-") as temporary:
        output = Path(temporary) / "site"
        builder.build_milestone_two(output)
        first = manifest(output)
        assessment = (output / "milestone-2.html").read_text()
        assert 'src="assets/fig_q1_rle_grid_8x8.png"' in assessment
        assert len(json.loads((output / "search.json").read_text())) == 10
        for url in (*builder.URLS, *builder.LEGACY_GUIDES, "walkthrough-notebook.html"):
            page = (output / url).read_text()
            assert 'href="../index.html"' in page
            assert 'href="../../index.html"' in page
        for url in builder.LEGACY_GUIDES:
            assert "earlier pipeline-oriented guide series" in (output / url).read_text()
        for resource in ("downloads/synthetic_pipeline_walkthrough.ipynb",
                         "downloads/synthetic_pipeline_walkthrough.py",
                         "examples/toy_dataset_loader.py",
                         "examples/toy_rle_roundtrip.py",
                         "examples/toy_unet_step.py"):
            assert (output / resource).read_bytes() == (builder.SOURCE / resource).read_bytes()
        notebook = json.loads((output / "downloads/synthetic_pipeline_walkthrough.ipynb").read_text())
        assert notebook["nbformat"] == 4
        assert all(not cell.get("outputs") for cell in notebook["cells"])

        # An incremental build must remove stale resources and preserve all URLs.
        (output / "downloads/stale-resource.txt").write_text("stale")
        builder.build_milestone_two(output)
        assert manifest(output) == first, "Clean and repeated builds differ"

        # Repository-relative, URL-encoded resources retain queries/fragments.
        fixture = Path(temporary).resolve() / "assessment.md"
        asset = builder.SOURCE / "assets/fig_q1_rle_grid_8x8.png"
        relative = quote(os.path.relpath(asset, fixture.parent)).replace("fig_q1", "%66ig_q1")
        fixture.write_text(f"# Fixture\n\n![grid]({relative}?download=1#grid)\n")
        with patch.object(builder, "QUESTION", fixture):
            builder.build_milestone_two(output)
        assert 'src="assets/fig_q1_rle_grid_8x8.png?download=1#grid"' in (output / "milestone-2.html").read_text(), (output / 'milestone-2.html').read_text().split('<article>')[1].split('</article>')[0]
    print("PASS: clean/repeated builds, assessment image, legacy routes/downloads and URL suffixes")


if __name__ == "__main__":
    check()
