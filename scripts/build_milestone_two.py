# /// script
# requires-python = ">=3.10"
# dependencies = ["Markdown==3.8.2", "pymdown-extensions==10.16.1"]
# ///
"""Build Milestone 2 student study guides, questions, and walkthrough resources."""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import markdown
from build_student_pages import (
    CREDIT_ART,
    CSS,
    JS,
    Inspect,
    normalize_markdown,
    version_theme_assets,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/milestone_2_student_guide"
QUESTION = ROOT / "MILESTONE_2.md"

URLS = [
    "index.html",
    "milestone-2.html",
    "guide_01_dataset_contract_and_strip_tensors.html",
    "guide_02_batching_dataloaders_and_isolation.html",
    "guide_03_industrial_augmentations_and_alignment.html",
    "guide_04_multilabel_stratification_and_leakage.html",
    "guide_05_wandb_experiment_tracking.html",
    "guide_06_baseline_unet_and_training_loop.html",
    "guide_07_postprocessing_logits_to_masks.html",
    "guide_08_rle_submission_and_verification.html",
    "walkthrough-notebook.html",
]

LABELS = [
    ("START HERE", "Milestone 2 Overview"),
    ("ASSESSMENT · 8 QUESTIONS", "Milestone 2 questions"),
    ("GUIDE 01", "Guide 01 · The Dataset Contract"),
    ("GUIDE 02", "Guide 02 · Batching & DataLoaders"),
    ("GUIDE 03", "Guide 03 · Industrial Augmentations"),
    ("GUIDE 04", "Guide 04 · Validation Stratification"),
    ("GUIDE 05", "Guide 05 · W&B Experiment Tracking"),
    ("GUIDE 06", "Guide 06 · Baseline U-Net & Training"),
    ("GUIDE 07", "Guide 07 · Post-Processing & Thresholds"),
    ("GUIDE 08", "Guide 08 · RLE & First Submission"),
    ("NOTEBOOK & CODE", "Walkthrough Notebook & Code Examples"),
]


def render_sidebar_nav(current_index: int) -> str:
    nav = '<input class="search" aria-label="Search Milestone 2 guides" placeholder="Search Milestone 2…" type="search"><div class="results" aria-live="polite"></div><nav class="nav" aria-label="Milestone 2 study guide">'
    for j, (url, (tag, title)) in enumerate(zip(URLS, LABELS)):
        active = ' aria-current="page"' if current_index == j else ""
        nav += f'<a href="{url}"{active}><span class="num">{tag}</span>{html.escape(title)}</a>'
    nav += '</nav><nav class="nav section-exits" aria-label="Leave Milestone 2"><a href="../index.html">← All milestone guides</a><a href="../../index.html">← App homepage</a></nav>'
    return nav


def render_pagination(current_index: int) -> str:
    pager = ""
    if current_index > 0:
        prev_url = URLS[current_index - 1]
        prev_title = LABELS[current_index - 1][1]
        pager += f'<a href="{prev_url}">← {html.escape(prev_title)}</a>'
    else:
        pager += "<span></span>"

    if current_index + 1 < len(URLS):
        next_url = URLS[current_index + 1]
        next_title = LABELS[current_index + 1][1]
        pager += f'<a href="{next_url}">{html.escape(next_title)} →</a>'
    else:
        pager += "<span></span>"
    return pager


def build_questions_page(output: Path) -> str:
    md = markdown.Markdown(
        tab_length=2,
        extensions=[
            "tables",
            "toc",
            "sane_lists",
            "md_in_html",
            "pymdownx.superfences",
            "pymdownx.arithmatex",
        ],
        extension_configs={
            "pymdownx.arithmatex": {"generic": True},
            "toc": {"toc_depth": "2-2"},
        },
    )
    raw = normalize_markdown(QUESTION.read_text(encoding="utf-8"))
    raw = re.sub(r"<details(?=\s|>)", '<details markdown="1"', raw)
    body = md.convert(raw)

    # The assessment lives at the repository root; the published page lives
    # beside the guides. Rebase only resource URLs, preserving its content.
    source_prefix = "docs/milestone_2_student_guide/"

    def rebase_resource(match):
        parts = urlsplit(html.unescape(match[2]))
        if not parts.scheme and not parts.netloc and parts.path.startswith(source_prefix):
            parts = parts._replace(path=parts.path[len(source_prefix):])
            return f'{match[1]}="{html.escape(parts.geturl(), quote=True)}"'
        return match[0]

    body = re.sub(r'(href|src)="([^"]+)"', rebase_resource, body)

    nav = render_sidebar_nav(1)
    pager = render_pagination(1)

    page = f"""<!doctype html>
<html lang="en" data-theme="paper"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Milestone 2 Questions · Industrial Surface Anomaly Segmentation</title>
<meta name="description" content="Milestone 2 assessment questions: RLE standards, decoding, class distribution, co-occurrence, spatial mutual exclusivity, PyTorch Dataset, and U-Net training dynamics.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,700;1,700&display=swap" rel="stylesheet">
<script>try {{ const t=localStorage.getItem('guide-garden-theme'); if(['paper','midnight','moss','butter','blush','mono'].includes(t)) document.documentElement.dataset.theme=t; }} catch {{}}</script>
<link rel="stylesheet" href="site.css">
<link rel="stylesheet" href="styles/generative/theme.css">
<link rel="stylesheet" href="styles/generative/navigation.css">
<script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}},options:{{ignoreHtmlClass:'tex2jax_ignore',processHtmlClass:'arithmatex'}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"></script>
<script defer src="site.js"></script><script defer src="styles/generative/theme.js"></script></head><body data-overview="false" data-ai-help="off">
<div class="reading-progress" aria-hidden="true"></div>
<a class="skip" href="#main">Skip to content</a><div class="layout">
<aside class="sidebar"><span class="eyebrow">DLGenAI · 26T3 · Milestone 2</span><a class="brand" href="index.html">Milestone 2<br>Study Guides</a>
<p>From tensors to submissions. Understand. Build. Validate.</p>
<details class="palette-panel" open><summary>Choose your palette</summary><div class="theme-options">
<button class="theme-option" data-palette="paper" aria-pressed="true" style="--swatch:#f3eee4;--dot:#e7462b"><span aria-hidden="true"></span>Paper</button>
<button class="theme-option" data-palette="midnight" aria-pressed="false" style="--swatch:#101b38;--dot:#ff7568"><span aria-hidden="true"></span>Midnight</button>
<button class="theme-option" data-palette="moss" aria-pressed="false" style="--swatch:#133a2a;--dot:#ffb7c5"><span aria-hidden="true"></span>Moss</button>
<button class="theme-option" data-palette="butter" aria-pressed="false" style="--swatch:#ffe9a8;--dot:#d72d43"><span aria-hidden="true"></span>Butter</button>
<button class="theme-option" data-palette="blush" aria-pressed="false" style="--swatch:#fad4d8;--dot:#bd2348"><span aria-hidden="true"></span>Blush</button>
<button class="theme-option" data-palette="mono" aria-pressed="false" style="--swatch:#141414;--dot:#f2f2f2"><span aria-hidden="true"></span>Mono</button>
</div><button class="motion-toggle" aria-pressed="true">Pause header motion</button></details>
<div class="desktop-nav">{nav}</div><details class="mobile-menu"><summary>Browse Milestone 2</summary>{nav}</details>
<footer><a href="https://github.com/manyakaistha/DLGenAI-26T3">View study materials on GitHub ↗</a></footer></aside>
<main id="main"><div class="content"><div class="topline"><span>Milestone 2 · Assessment questions</span><a href="walkthrough-notebook.html">View walkthrough code ↗</a></div>
<div class="garden-hero"><canvas aria-hidden="true"></canvas><div class="hero-copy"><span class="hero-kicker">Milestone 2 Assessment</span><p class="hero-title">From tensors<br>to submissions.</p><p class="hero-caption">Understand. Build. Validate.</p></div></div>
<span class="section-label">Assessment · 8 questions</span><details class="toc"><summary>On this page</summary>{md.toc}</details>
<article>{body}</article><nav class="pagination" aria-label="Previous and next page">{pager}</nav>
<footer class="endnote"><span class="author-credit">{CREDIT_ART}<span>Created by manaykaistha.</span>{CREDIT_ART}</span><br>Understand the concepts. Practice on synthetic examples. Build and audit your own pipeline.</footer></div></main></div></body></html>"""
    (output / "milestone-2.html").write_text(page, encoding="utf-8")
    return body


def build_milestone_two(output: Path):
    """Build all Milestone 2 pages into output directory."""
    output.mkdir(parents=True, exist_ok=True)

    # Copy assets, downloads, examples
    if (SOURCE / "assets").is_dir():
        shutil.copytree(SOURCE / "assets", output / "assets", dirs_exist_ok=True)
    if (SOURCE / "downloads").is_dir():
        shutil.copytree(SOURCE / "downloads", output / "downloads", dirs_exist_ok=True)
    if (SOURCE / "examples").is_dir():
        shutil.copytree(SOURCE / "examples", output / "examples", dirs_exist_ok=True)
    shutil.copytree(ROOT / "styles/generative", output / "styles/generative", dirs_exist_ok=True)

    (output / "site.css").write_text(CSS, encoding="utf-8")
    (output / "site.js").write_text(JS, encoding="utf-8")
    (output / ".nojekyll").touch()

    # Build milestone-2.html from MILESTONE_2.md
    questions_body = build_questions_page(output)

    search = []
    inspect_q = Inspect()
    inspect_q.feed(questions_body)
    search.append({
        "title": "Milestone 2 questions",
        "url": "milestone-2.html",
        "text": " ".join(inspect_q.text),
    })

    # Now update and copy the other 10 HTML pages from SOURCE
    source_pages = [
        ("index.html", 0),
        ("guide_01_dataset_contract_and_strip_tensors.html", 2),
        ("guide_02_batching_dataloaders_and_isolation.html", 3),
        ("guide_03_industrial_augmentations_and_alignment.html", 4),
        ("guide_04_multilabel_stratification_and_leakage.html", 5),
        ("guide_05_wandb_experiment_tracking.html", 6),
        ("guide_06_baseline_unet_and_training_loop.html", 7),
        ("guide_07_postprocessing_logits_to_masks.html", 8),
        ("guide_08_rle_submission_and_verification.html", 9),
        ("walkthrough-notebook.html", 10),
    ]

    for filename, idx in source_pages:
        src_file = SOURCE / filename
        content = src_file.read_text(encoding="utf-8")

        # These pages were copied from an unlisted review snapshot. Only the
        # public output joins the discoverable milestone directory; keep the
        # source and guide2701 review snapshot untouched.
        content = content.replace(
            '<meta name="robots" content="noindex,nofollow,noarchive,nosnippet">',
            '',
        )

        # Ensure navigation.css is included in head
        if "styles/generative/navigation.css" not in content:
            content = content.replace(
                '<link rel="stylesheet" href="styles/generative/theme.css">',
                '<link rel="stylesheet" href="styles/generative/theme.css">\n<link rel="stylesheet" href="styles/generative/navigation.css">',
            )

        # Replace desktop nav
        nav_html = render_sidebar_nav(idx)
        content = re.sub(
            r'<div class="desktop-nav">.*?</div>\s*<details class="mobile-menu">',
            f'<div class="desktop-nav">{nav_html}</div><details class="mobile-menu">',
            content,
            flags=re.S,
        )

        # Replace mobile menu
        content = re.sub(
            r'<details class="mobile-menu">.*?</details>\s*<footer>',
            f'<details class="mobile-menu"><summary>Browse Milestone 2</summary>{nav_html}</details><footer>',
            content,
            flags=re.S,
        )

        # Replace topline link to point to milestone-2.html
        content = re.sub(
            r'<div class="topline"><span>(.*?)</span><a href="[^"]*">(.*?)</a></div>',
            r'<div class="topline"><span>\1</span><a href="milestone-2.html">View questions ↗</a></div>',
            content,
        )

        # Replace pagination
        pager_html = render_pagination(idx)
        content = re.sub(
            r'<nav class="pagination"[^>]*>.*?</nav>',
            f'<nav class="pagination" aria-label="Previous and next page">{pager_html}</nav>',
            content,
            flags=re.S,
        )

        # Update brand link text
        content = re.sub(
            r'<a class="brand" href="index.html">.*?</a>',
            r'<a class="brand" href="index.html">Milestone 2<br>Study Guides</a>',
            content,
            flags=re.S,
        )

        (output / filename).write_text(content, encoding="utf-8")

        # Collect text for search index
        parsed = Inspect()
        # Sidebar labels occur on every page and would make topic searches
        # return every guide. Index the lesson itself, like Milestone 1.
        article = re.search(r'<article>(.*?)</article>', content, flags=re.S)
        if article is None:
            raise ValueError(f"Missing lesson article in {src_file}")
        parsed.feed(article[1])
        search.append({
            "title": LABELS[idx][1],
            "url": filename,
            "text": " ".join(parsed.text),
        })

    (output / "search.json").write_text(json.dumps(search, ensure_ascii=False), encoding="utf-8")
    print(f"Built {len(URLS)} Milestone 2 pages → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/milestone_2_preview")
    build_milestone_two(parser.parse_args().output.resolve())
