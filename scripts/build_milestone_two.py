# /// script
# requires-python = ">=3.10"
# dependencies = ["Markdown==3.8.2", "pymdown-extensions==10.16.1"]
# ///
"""Build Milestone 2 question-wise student study guides, questions, and walkthrough resources."""
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
    "guide_01_rle_column_major_encoding.html",
    "guide_02_rle_decoding_and_pixel_mass.html",
    "guide_03_global_class_distribution_and_prevalence.html",
    "guide_04_defect_cooccurrence_matrix.html",
    "guide_05_cooccurrence_vs_correlation_base_rate.html",
    "guide_06_pixel_level_spatial_overlap_and_mutual_exclusivity.html",
    "guide_07_pytorch_dataset_and_architectural_paradigms.html",
    "guide_08_training_dynamics_and_convergence.html",
]

LABELS = [
    ("START HERE", "Milestone 2 Overview"),
    ("ASSESSMENT · 8 QUESTIONS", "Milestone 2 questions"),
    ("GUIDE 01", "Q1 · RLE Column-Major Encoding"),
    ("GUIDE 02", "Q2 · RLE Decoding & Pixel Mass"),
    ("GUIDE 03", "Q3 · Global Class Distribution"),
    ("GUIDE 04", "Q4 · Defect Co-Occurrence Matrix"),
    ("GUIDE 05", "Q5 · Co-Occurrence vs. Correlation"),
    ("GUIDE 06", "Q6 · Spatial Overlap & Exclusivity"),
    ("GUIDE 07", "Q7 · PyTorch Dataset & Paradigms"),
    ("GUIDE 08", "Q8 · Training Dynamics & Convergence"),
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


def build_milestone_two(output: Path):
    """Build all Milestone 2 question-wise pages from Markdown sources into output directory."""
    output.mkdir(parents=True, exist_ok=True)

    sources = [
        SOURCE / "README.md",
        QUESTION,
        SOURCE / "guide_01_rle_column_major_encoding.md",
        SOURCE / "guide_02_rle_decoding_and_pixel_mass.md",
        SOURCE / "guide_03_global_class_distribution_and_prevalence.md",
        SOURCE / "guide_04_defect_cooccurrence_matrix.md",
        SOURCE / "guide_05_cooccurrence_vs_correlation_base_rate.md",
        SOURCE / "guide_06_pixel_level_spatial_overlap_and_mutual_exclusivity.md",
        SOURCE / "guide_07_pytorch_dataset_and_architectural_paradigms.md",
        SOURCE / "guide_08_training_dynamics_and_convergence.md",
    ]

    titles = [item[1] for item in LABELS]
    mapping = {p.resolve(): url for p, url in zip(sources, URLS)}

    # Clean up stale files
    stale_files = ["walkthrough-notebook.html"]
    for stale_file in stale_files:
        stale_path = output / stale_file
        if stale_path.is_file():
            stale_path.unlink()
    for stale_dir in ["downloads", "examples"]:
        stale_path = output / stale_dir
        if stale_path.is_dir():
            shutil.rmtree(stale_path)

    # Copy assets
    if (SOURCE / "assets").is_dir():
        if (output / "assets").is_dir():
            shutil.rmtree(output / "assets")
        shutil.copytree(SOURCE / "assets", output / "assets", dirs_exist_ok=True)
    if (output / "styles").is_dir():
        shutil.rmtree(output / "styles")
    shutil.copytree(ROOT / "styles/generative", output / "styles/generative", dirs_exist_ok=True)

    (output / "site.css").write_text(CSS, encoding="utf-8")
    (output / "site.js").write_text(JS, encoding="utf-8")
    (output / ".nojekyll").touch()

    search = []

    for i, source in enumerate(sources):
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
        raw_text = source.read_text(encoding="utf-8")
        markdown_source = normalize_markdown(raw_text)
        markdown_source = re.sub(r"<details(?=\s|>)", '<details markdown="1"', markdown_source)
        body = md.convert(markdown_source)

        if re.search(r'<(?:span|div) class="arithmatex">\\[\[(]\s*<(?:span|div) class="arithmatex">', body):
            raise ValueError(f"Nested math delimiters in {source}")

        def link(match):
            attr, raw = match.groups()
            parts = urlsplit(html.unescape(raw))
            if parts.scheme or parts.netloc or not parts.path:
                return match.group(0)
            resolved = (source.parent / unquote(parts.path)).resolve()
            if resolved in mapping:
                target = mapping[resolved]
            elif parts.path.startswith("assets/") or parts.path.startswith("downloads/"):
                target = parts.path
            elif (output / parts.path).exists():
                target = parts.path
            else:
                target = parts.path
            if parts.fragment:
                target += "#" + parts.fragment
            return f'{attr}="{html.escape(target, quote=True)}"'

        body = re.sub(r'(href|src)="([^"]+)"', link, body)

        nav = render_sidebar_nav(i)
        pager = render_pagination(i)

        section_tag = LABELS[i][0]
        if i == 0:
            top_crumb = "Milestone 2 · Student learning resources"
            hero_kicker = "Milestone 2 Curriculum"
            hero_title = "From tensors<br>to submissions."
            hero_caption = "Understand. Build. Validate."
            section_label = "Study guide overview"
        elif i == 1:
            top_crumb = "Milestone 2 · Assessment questions"
            hero_kicker = "Milestone 2 Assessment"
            hero_title = "From tensors<br>to submissions."
            hero_caption = "Understand. Build. Validate."
            section_label = "Assessment · 8 questions"
        else:
            top_crumb = "Milestone 2 · Student learning resources"
            hero_kicker = f"Question {i-1} Guided Study"
            hero_title = "From tensors<br>to submissions."
            hero_caption = "Observe. Understand. Experiment."
            section_label = f"Concept guide {i-1:02d} / 08"

        page = f"""<!doctype html>
<html lang="en" data-theme="paper"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(titles[i])} · Industrial Surface Anomaly Segmentation</title>
<meta name="description" content="Milestone 2 student study guides: question-wise conceptual guides, PyTorch datasets, co-occurrence, spatial overlap, and baseline U-Net training dynamics.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,700;1,700&display=swap" rel="stylesheet">
<script>try {{ const t=localStorage.getItem('guide-garden-theme'); if(['paper','midnight','moss','butter','blush','mono'].includes(t)) document.documentElement.dataset.theme=t; }} catch {{}}</script>
<link rel="stylesheet" href="site.css">
<link rel="stylesheet" href="styles/generative/theme.css">
<link rel="stylesheet" href="styles/generative/navigation.css">
<script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}},options:{{ignoreHtmlClass:'tex2jax_ignore',processHtmlClass:'arithmatex'}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"></script>
<script defer src="site.js"></script><script defer src="styles/generative/theme.js"></script></head><body data-overview="{str(i == 0).lower()}" data-ai-help="{'off' if source == QUESTION else 'study'}">
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
<main id="main"><div class="content"><div class="topline"><span>{top_crumb}</span><a href="milestone-2.html">View questions ↗</a></div>
<div class="garden-hero"><canvas aria-hidden="true"></canvas><div class="hero-copy"><span class="hero-kicker">{hero_kicker}</span><p class="hero-title">{hero_title}</p><p class="hero-caption">{hero_caption}</p></div></div>
<span class="section-label">{section_label}</span><details class="toc"><summary>On this page</summary>{md.toc}</details>
<article>{body}</article><nav class="pagination" aria-label="Previous and next page">{pager}</nav>
<footer class="endnote"><span class="author-credit">{CREDIT_ART}<span>Created by manaykaistha.</span>{CREDIT_ART}</span><br>Study the concepts. Practice on toy examples. Compute your own assessment results.</footer></div></main></div></body></html>"""

        (output / URLS[i]).write_text(page, encoding="utf-8")
        parsed = Inspect()
        parsed.feed(body)
        search.append({
            "title": titles[i],
            "url": URLS[i],
            "text": " ".join(parsed.text),
        })

    (output / "search.json").write_text(json.dumps(search, ensure_ascii=False), encoding="utf-8")
    print(f"Built {len(URLS)} question-wise Milestone 2 pages → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/milestone_2_preview")
    build_milestone_two(parser.parse_args().output.resolve())
