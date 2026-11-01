# /// script
# requires-python = ">=3.10"
# dependencies = ["Markdown==3.8.2", "pymdown-extensions==10.16.1"]
# ///
"""Build Milestone 3 question-wise student study guides, questions, and walkthrough resources."""
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
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/milestone_3_student_guide"
QUESTION = ROOT / "MILESTONE_3.md"

# Let wrapped hero copy define height; the shared fixed-height hero clips this
# milestone's longer title at narrow widths. Keep the change local to M3 pages.
MILESTONE_CSS = """
.garden-hero{height:auto;min-height:185px}
body[data-overview="true"] .garden-hero{height:auto;min-height:285px}
@media(max-width:850px){
  .garden-hero{min-height:160px}
  .hero-copy{max-width:74%}
  body[data-overview="true"] .garden-hero{min-height:215px}
}
"""

URLS = [
    "index.html",
    "milestone-3.html",
    "guide_01_feature_map_geometry_and_downsampling.html",
    "guide_02_pretrained_backbone_and_parameter_audit.html",
    "guide_03_grayscale_adaptation_and_channel_physics.html",
    "guide_04_atrous_convolutions_and_multiscale_aspp.html",
    "guide_05_soft_dice_loss_and_laplace_smoothing.html",
    "guide_06_output_heads_and_target_semantics.html",
    "guide_07_compound_loss_and_clean_image_dynamics.html",
    "guide_08_scratch_vs_pretrained_convergence.html",
    "guide_09_backbone_architecture_tradeoffs.html",
]

LABELS = [
    ("START HERE", "Milestone 3 Overview"),
    ("ASSESSMENT · 9 QUESTIONS", "Milestone 3 questions"),
    ("GUIDE 01", "Q1 · Feature Map Geometry & Downsampling"),
    ("GUIDE 02", "Q2 · Pretrained Backbones & Parameter Audit"),
    ("GUIDE 03", "Q3 · Grayscale Adaptation & Channel Physics"),
    ("GUIDE 04", "Q4 · Atrous Convolutions & ASPP"),
    ("GUIDE 05", "Q5 · Soft Dice Loss & Laplace Smoothing"),
    ("GUIDE 06", "Q6 · Output Heads & Target Semantics"),
    ("GUIDE 07", "Q7 · Compound Loss & Clean Image Dynamics"),
    ("GUIDE 08", "Q8 · Scratch vs. Pretrained Convergence"),
    ("GUIDE 09", "Q9 · Backbone Architecture Trade-offs"),
]

LEGACY_GUIDES = {
    "guide_01_encoder_decoder_architectures.html": ("Encoder-decoder architectures and spatial mechanics", URLS[2]),
    "guide_02_pretrained_backbones_and_transfer_learning.html": ("Pretrained backbones and feature hierarchies", URLS[3]),
    "guide_03_grayscale_input_adaptation.html": ("Grayscale input adaptation strategies", URLS[4]),
    "guide_04_output_heads_and_target_semantics.html": ("Output heads and target semantics", URLS[7]),
    "guide_05_bce_dice_and_compound_losses.html": ("BCE, Dice, and compound loss behavior", URLS[6]),
    "guide_06_experimental_design_and_model_evaluation.html": ("Experimental design and model evaluation", URLS[9]),
}


def render_sidebar_nav(current_index: int) -> str:
    nav = '<input class="search" aria-label="Search Milestone 3 guides" placeholder="Search Milestone 3…" type="search"><div class="results" aria-live="polite"></div><nav class="nav" aria-label="Milestone 3 study guide">'
    for j, (url, (tag, title)) in enumerate(zip(URLS, LABELS)):
        active = ' aria-current="page"' if current_index == j else ""
        nav += f'<a href="{url}"{active}><span class="num">{tag}</span>{html.escape(title)}</a>'
    nav += '</nav><nav class="nav section-exits" aria-label="Leave Milestone 3"><a href="../index.html">← All milestone guides</a><a href="../../index.html">← App homepage</a></nav>'
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


def build_legacy_pages(output: Path) -> None:
    """Keep previously published lesson and download URLs accessible."""
    legacy_notice = """<p><strong>Legacy reference material.</strong> These synthetic transfer learning walkthroughs
    demonstrate feature extraction and adaptation on synthetic tensors. Milestone 3 is now organized
    directly by assessment question.</p>"""
    resources = """<ul>
    <li><a href="downloads/synthetic_transfer_learning_walkthrough.ipynb" download>Synthetic transfer learning · Jupyter notebook</a></li>
    <li><a href="downloads/synthetic_transfer_learning_walkthrough.py" download>Synthetic transfer learning · Python script</a></li>
    <li><a href="examples/toy_feature_extractor.py" download>Toy feature extractor script</a></li>
    <li><a href="examples/toy_grayscale_adaptation.py" download>Toy grayscale adaptation script</a></li>
    <li><a href="examples/toy_head_comparison.py" download>Toy head comparison script</a></li>
    <li><a href="examples/toy_loss_behavior.py" download>Toy loss behavior script</a></li>
    </ul>"""
    pages = {}
    for url, (topic, target) in LEGACY_GUIDES.items():
        title = f"Previously shared guide: {topic}"
        target_title = LABELS[URLS.index(target)][1]
        body = f"""<h1>{html.escape(topic)}</h1>
        <p>This previously shared URL belongs to the earlier topic-oriented guide series.
        Milestone 3 is now organized directly by assessment question.</p>
        <p>Start with <a href="{target}">{html.escape(target_title)}</a> to find the
        relevant current material, or browse the <a href="index.html">current curriculum map</a>.</p>
        <p>The <a href="walkthrough-notebook.html">synthetic walkthrough and scripts</a>
        remain available as reference downloads.</p>"""
        pages[url] = (title, body)
    pages["walkthrough-notebook.html"] = ("Synthetic transfer learning walkthrough", f"""
        <h1>Synthetic transfer learning walkthrough</h1>{legacy_notice}{resources}
        <p>For the current assessment, follow <a href="{URLS[2]}">Guide 1: Feature map geometry</a>,
        <a href="{URLS[3]}">Guide 2: Pretrained backbones</a>, and <a href="{URLS[9]}">Guide 8: Training convergence</a>.</p>""")
    nav = render_sidebar_nav(-1)
    for url, (title, body) in pages.items():
        page = f"""<!doctype html>
<html lang="en" data-theme="paper"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · Milestone 3</title>
<script>try {{ const t=localStorage.getItem('guide-garden-theme'); if(['paper','midnight','moss','butter','blush','mono'].includes(t)) document.documentElement.dataset.theme=t; }} catch {{}}</script>
<link rel="stylesheet" href="site.css"><link rel="stylesheet" href="styles/generative/theme.css">
<link rel="stylesheet" href="styles/generative/navigation.css">
<style>{MILESTONE_CSS}</style>
<script defer src="site.js"></script><script defer src="styles/generative/theme.js"></script>
</head><body data-overview="false" data-ai-help="off">
<a class="skip" href="#main">Skip to content</a><div class="layout">
<aside class="sidebar"><span class="eyebrow">DLGenAI · 26T3 · Milestone 3</span>
<a class="brand" href="index.html">Milestone 3<br>Study Guides</a>
<p>Previously shared resources</p><div class="desktop-nav">{nav}</div>
<details class="mobile-menu"><summary>Browse Milestone 3</summary>{nav}</details></aside>
<main id="main"><div class="content"><span class="section-label">Legacy link · Current navigation</span>
<article>{body}</article><nav class="pagination" aria-label="Return to current guides">
<a href="index.html">← Current Milestone 3 guides</a></nav></div></main></div></body></html>"""
        (output / url).write_text(page, encoding="utf-8")


def build_milestone_three(output: Path, *, milestone_one_url: str = "../milestone-1/milestone-1.html"):
    """Build all Milestone 3 question-wise pages from Markdown sources into output directory."""
    output.mkdir(parents=True, exist_ok=True)

    sources = [
        SOURCE / "README.md",
        QUESTION,
        SOURCE / "guide_01_feature_map_geometry_and_downsampling.md",
        SOURCE / "guide_02_pretrained_backbone_and_parameter_audit.md",
        SOURCE / "guide_03_grayscale_adaptation_and_channel_physics.md",
        SOURCE / "guide_04_atrous_convolutions_and_multiscale_aspp.md",
        SOURCE / "guide_05_soft_dice_loss_and_laplace_smoothing.md",
        SOURCE / "guide_06_output_heads_and_target_semantics.md",
        SOURCE / "guide_07_compound_loss_and_clean_image_dynamics.md",
        SOURCE / "guide_08_scratch_vs_pretrained_convergence.md",
        SOURCE / "guide_09_backbone_architecture_tradeoffs.md",
    ]

    titles = [item[1] for item in LABELS]
    mapping = {p.resolve(): url for p, url in zip(sources, URLS)}
    mapping[(ROOT / "MILESTONE_1.md").resolve()] = milestone_one_url

    for directory in ("assets", "downloads", "examples"):
        destination = output / directory
        if destination.is_dir():
            shutil.rmtree(destination)
        if (SOURCE / directory).is_dir():
            shutil.copytree(SOURCE / directory, destination)
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
            elif resolved.is_relative_to(SOURCE.resolve()) and resolved.relative_to(SOURCE.resolve()).parts[0] in {"assets", "downloads", "examples"}:
                target = resolved.relative_to(SOURCE.resolve()).as_posix()
            else:
                target = parts.path
            if parts.query:
                target += "?" + parts.query
            if parts.fragment:
                target += "#" + parts.fragment
            return f'{attr}="{html.escape(target, quote=True)}"'

        body = re.sub(r'(href|src)="([^"]+)"', link, body)

        nav = render_sidebar_nav(i)
        pager = render_pagination(i)

        if i == 0:
            top_crumb = "Milestone 3 · Student learning resources"
            hero_kicker = "Milestone 3 Curriculum"
            hero_title = "Transfer learning.<br>Stronger models."
            hero_caption = "Understand. Adapt. Evaluate."
            section_label = "Study guide overview"
        elif i == 1:
            top_crumb = "Milestone 3 · Assessment questions"
            hero_kicker = "Milestone 3 Assessment"
            hero_title = "Transfer learning.<br>Stronger models."
            hero_caption = "Understand. Adapt. Evaluate."
            section_label = "Assessment · 9 questions"
        else:
            top_crumb = "Milestone 3 · Student learning resources"
            hero_kicker = f"Question {i-1} Guided Study"
            hero_title = "Transfer learning.<br>Stronger models."
            hero_caption = "Observe. Understand. Experiment."
            section_label = f"Concept guide {i-1:02d} / 09"

        page = f"""<!doctype html>
<html lang="en" data-theme="paper"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(titles[i])} · Industrial Surface Anomaly Segmentation</title>
<meta name="description" content="Milestone 3 student study guides: transfer learning, pretrained vision backbones, grayscale adaptation, ASPP, and compound loss.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,700;1,700&display=swap" rel="stylesheet">
<script>try {{ const t=localStorage.getItem('guide-garden-theme'); if(['paper','midnight','moss','butter','blush','mono'].includes(t)) document.documentElement.dataset.theme=t; }} catch {{}}</script>
<link rel="stylesheet" href="site.css">
<link rel="stylesheet" href="styles/generative/theme.css">
<link rel="stylesheet" href="styles/generative/navigation.css">
<style>{MILESTONE_CSS}</style>
<script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}},options:{{ignoreHtmlClass:'tex2jax_ignore',processHtmlClass:'arithmatex'}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"></script>
<script defer src="site.js"></script><script defer src="styles/generative/theme.js"></script></head><body data-overview="{str(i == 0).lower()}" data-ai-help="{'off' if source == QUESTION else 'study'}">
<div class="reading-progress" aria-hidden="true"></div>
<a class="skip" href="#main">Skip to content</a><div class="layout">
<aside class="sidebar"><span class="eyebrow">DLGenAI · 26T3 · Milestone 3</span><a class="brand" href="index.html">Milestone 3<br>Study Guides</a>
<p>Transfer learning, pretrained backbones, output heads, and compound loss objectives.</p>
<details class="palette-panel" open><summary>Choose your palette</summary><div class="theme-options">
<button class="theme-option" data-palette="paper" aria-pressed="true" style="--swatch:#f3eee4;--dot:#e7462b"><span aria-hidden="true"></span>Paper</button>
<button class="theme-option" data-palette="midnight" aria-pressed="false" style="--swatch:#101b38;--dot:#ff7568"><span aria-hidden="true"></span>Midnight</button>
<button class="theme-option" data-palette="moss" aria-pressed="false" style="--swatch:#133a2a;--dot:#ffb7c5"><span aria-hidden="true"></span>Moss</button>
<button class="theme-option" data-palette="butter" aria-pressed="false" style="--swatch:#ffe9a8;--dot:#d72d43"><span aria-hidden="true"></span>Butter</button>
<button class="theme-option" data-palette="blush" aria-pressed="false" style="--swatch:#fad4d8;--dot:#bd2348"><span aria-hidden="true"></span>Blush</button>
<button class="theme-option" data-palette="mono" aria-pressed="false" style="--swatch:#141414;--dot:#f2f2f2"><span aria-hidden="true"></span>Mono</button>
</div><button class="motion-toggle" aria-pressed="true">Pause header motion</button></details>
<div class="desktop-nav">{nav}</div><details class="mobile-menu"><summary>Browse Milestone 3</summary>{nav}</details>
<footer><a href="https://github.com/manyakaistha/DLGenAI-26T3">View study materials on GitHub ↗</a></footer></aside>
<main id="main"><div class="content"><div class="topline"><span>{top_crumb}</span><a href="milestone-3.html">View questions ↗</a></div>
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

    build_legacy_pages(output)

    (output / "search.json").write_text(json.dumps(search, ensure_ascii=False), encoding="utf-8")
    print(f"Built {len(URLS)} question-wise Milestone 3 pages → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "guide2701/milestone-3")
    output = parser.parse_args().output.resolve()
    milestone_one_url = ("../../milestones/milestone-1/milestone-1.html"
                         if output.parent == ROOT / "guide2701"
                         else "../milestone-1/milestone-1.html")
    build_milestone_three(output, milestone_one_url=milestone_one_url)
