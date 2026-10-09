# /// script
# requires-python = ">=3.10"
# dependencies = ["Markdown==3.8.2", "pymdown-extensions==10.16.1"]
# ///
"""Build the public Milestone 1 guide; run with uv run --no-project."""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import markdown

ROOT = Path(__file__).resolve().parents[1]
GUIDES = ROOT / "docs/milestone_1_student_guide_v2"
QUESTION = ROOT / "MILESTONE_1.md"
LABELS = [
    "Data sparsity & cardinality", "Spatial exclusivity & output heads",
    "RLE decoding & boundary checks", "Aspect ratios & aliasing",
    "Photometric contrast vs. texture", "Dice & empty masks",
    "Competition metric & baselines", "Morphology & recall",
    "Stratification & threshold sweeps",
]
CSS = """
:root{--ink:#172d3a;--muted:#536773;--paper:#fffefa;--line:#dde4e5;--accent:#006e6b;--wash:#e9f3f1;--code:#edf1f3}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:30px}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.75 system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}a{color:var(--accent);text-underline-offset:3px}a:hover{text-decoration-thickness:2px}a:focus-visible,button:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid #c57924;outline-offset:4px}.skip{position:absolute;top:-100px}.skip:focus{top:10px;left:20px;z-index:5;background:white;padding:10px}.layout{display:grid;grid-template-columns:290px minmax(0,1fr);min-height:100vh}.sidebar{position:sticky;top:0;height:100vh;overflow:auto;padding:34px 25px;background:#f2f6f5;border-right:1px solid var(--line)}.brand{display:block;font-weight:800;font-size:21px;line-height:1.3;text-decoration:none;color:var(--ink);margin-bottom:9px}.eyebrow{font-size:11px;font-weight:800;letter-spacing:.13em;text-transform:uppercase;color:var(--accent)}.sidebar p{font-size:13px;color:var(--muted);line-height:1.6;margin:14px 0 24px}.nav a{display:block;text-decoration:none;font-size:14px;line-height:1.45;padding:11px 12px;border-radius:7px;margin:4px 0}.nav a[aria-current]{background:var(--accent);color:white}.nav .num{font:11px ui-monospace,monospace;opacity:.7;display:block;margin-bottom:3px}.sidebar footer{border-top:1px solid var(--line);margin-top:26px;padding-top:20px;font-size:12px}main{min-width:0;padding:55px clamp(24px,5vw,90px) 40px}.content{max-width:920px;margin:auto}.topline{display:flex;justify-content:space-between;gap:20px;border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:34px;font-size:12px}.topline a{font-weight:650}.section-label{color:var(--accent);font-size:12px;text-transform:uppercase;letter-spacing:.13em;font-weight:750}h1,h2,h3,h4{line-height:1.3;letter-spacing:-.025em}h1{font-size:clamp(30px,4vw,45px);margin:14px 0 25px;max-width:850px}h2{font-size:27px;margin:48px 0 20px;padding-top:10px}h3{font-size:21px;margin-top:32px}h4{font-size:18px}p{margin:16px 0}li{margin:7px 0}hr{border:0;border-top:1px solid var(--line);margin:36px 0}img{display:block;max-width:100%;height:auto;border-radius:8px;margin:24px auto}code{background:var(--code);font-size:.86em;padding:3px 6px;border-radius:4px;overflow-wrap:anywhere}pre{background:#152d3a;color:#e7f1f4;padding:20px;border-radius:9px;overflow:auto;line-height:1.6;font-size:14px}pre code{background:none;color:inherit;padding:0;overflow-wrap:normal}blockquote{background:var(--wash);border-left:3px solid var(--accent);margin:24px 0;padding:10px 22px}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.65}th,td{padding:13px 15px;border:1px solid var(--line);text-align:left;vertical-align:top}th{background:var(--wash)}.table-scroll{overflow:auto;margin:24px 0}details{border:1px solid var(--line);border-radius:8px;margin:25px 0;padding:14px 20px;background:#f7faf8}summary{cursor:pointer;color:var(--accent);font-weight:650}details[open] summary{margin-bottom:22px}.toc{background:#f2f6f5;border-radius:10px;margin:30px 0;padding:16px 22px}.toc ul{list-style:none;padding-left:0;font-size:14px}.toc li{margin:7px 0}.arithmatex{overflow-x:auto;overflow-y:hidden;max-width:100%}span.arithmatex{display:inline-block;vertical-align:middle}mjx-container{max-width:100%}.pagination{display:flex;justify-content:space-between;gap:24px;margin-top:50px;padding-top:25px;border-top:1px solid var(--line);font-size:14px}.pagination a{max-width:48%}.endnote{font-size:12px;color:var(--muted);margin-top:34px}.search{width:100%;padding:10px 12px;border:1px solid #b8c9c8;border-radius:6px;background:white;color:var(--ink);margin-bottom:12px;font:inherit;font-size:13px}.results a{display:block;font-size:13px;margin-bottom:12px}.results p{font-size:12px}.mobile-menu{display:none}.no-results{font-size:12px}.hero-note{background:var(--wash);padding:18px 23px;border-radius:9px;font-size:14px;margin:25px 0}
@media(max-width:850px){.layout{display:block}.sidebar{position:relative;height:auto;padding:20px 24px;border-right:0;border-bottom:1px solid var(--line)}.sidebar p,.sidebar footer{display:none}.brand{font-size:19px}.mobile-menu{display:block;margin-top:15px}.mobile-menu summary{font-size:14px}.desktop-nav{display:none}main{padding:28px 22px}.topline{margin-bottom:25px}.pagination{font-size:13px}h2{font-size:24px}body{font-size:16px}.toc{padding:12px 17px}}
@media print{.sidebar,.topline,.toc,.pagination,.endnote{display:none}.layout{display:block}main{padding:0}body{font-size:11pt}details{display:block}pre{white-space:pre-wrap;background:#eee;color:black}a{color:inherit}.content{max-width:none}}
"""
JS = """
const inputs = document.querySelectorAll('.search');
let index;
inputs.forEach(input => input.addEventListener('input', async () => {
  const results = input.nextElementSibling;
  const query = input.value.trim().toLowerCase();
  results.replaceChildren();
  if (query.length < 2) return;
  try {
    index ||= await (await fetch('search.json')).json();
    const matches = index.filter(p => (p.title+' '+p.text).toLowerCase().includes(query));
    for (const p of matches) {
      const a = document.createElement('a'); a.href=p.url; a.textContent=p.title; results.append(a);
    }
    if (!matches.length) results.textContent='No matching pages.';
  } catch { results.textContent='Search unavailable. Use the guide navigation below.'; }
}));
document.querySelectorAll('article table').forEach(table => {
 const wrapper=document.createElement('div'); wrapper.className='table-scroll';
 wrapper.tabIndex=0; wrapper.setAttribute('role','region'); wrapper.setAttribute('aria-label','Scrollable table');
 table.before(wrapper); wrapper.append(table);
});
"""

class Inspect(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids, self.text = [], set(), []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for key in ("href", "src"):
            if key in attrs:
                self.links.append(attrs[key])
    def handle_data(self, data):
        self.text.append(data)


def normalize_markdown(source: str) -> str:
    """Adapt the guide's GitHub-style spacing to Python-Markdown block rules."""
    lines = []
    fence_char, fence_length = "", 0
    in_display = False
    for line in source.splitlines():
        fence = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)
            if not fence_char:
                fence_char, fence_length = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                fence_char, fence_length = "", 0
            lines.append(line)
            continue
        if fence_char:
            lines.append(line)
            continue
        # Display math must form its own block. Otherwise $$ is consumed by
        # two inline-math matches, leaving nested wrappers and visible \( \).
        if in_display:
            lines.append(line)
            if line.rstrip().endswith("$$"):
                in_display = False
                lines.append("")
            continue
        display = re.match(r"^(\s*)\$\$", line)
        if display:
            if lines and lines[-1].strip():
                lines.append("")
            # Lists use two-space nesting; a leftover third/fifth space keeps
            # the math block from matching after Markdown removes list indent.
            indent = " " * (len(display.group(1)) // 2 * 2)
            lines.append(indent + line.lstrip())
            in_display = not (line.strip().endswith("$$") and len(line.strip()) > 2)
            if not in_display:
                lines.append("")
            continue
        # GitHub permits lists to interrupt paragraphs; Python-Markdown needs
        # a blank line. Keep adjacent items together and preserve indentation.
        item = re.match(r"^(\s*)(?:[-+*]|\d+[.)])\s+", line)
        if item and lines and lines[-1].strip():
            previous = lines[-1]
            previous_item = re.match(r"^\s*(?:[-+*]|\d+[.)])\s+", previous)
            indent = len(item.group(1))
            previous_indent = len(previous) - len(previous.lstrip())
            if not previous_item and (indent <= previous_indent or previous.rstrip().endswith(":")):
                lines.append("")
        lines.append(line)
    return "\n".join(lines)


def build(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    sources = [GUIDES / "README.md", QUESTION, *sorted(GUIDES.glob("guide_*.md"))]
    if len(sources) != 11:
        raise ValueError("Expected the guide index, revised assignment and nine modules")
    urls = ["index.html", "milestone-1.html", *[p.stem + ".html" for p in sources[2:]]]
    titles = ["Study guide overview", "Milestone 1 questions", *LABELS]
    mapping = {p.resolve(): url for p, url in zip(sources, urls)}
    shutil.copytree(GUIDES / "assets", output / "assets", dirs_exist_ok=True)
    (output / "site.css").write_text(CSS)
    (output / "site.js").write_text(JS)
    (output / ".nojekyll").touch()
    search = []
    for i, source in enumerate(sources):
        md = markdown.Markdown(
            tab_length=2,
            extensions=["tables", "toc", "sane_lists", "md_in_html", "pymdownx.superfences", "pymdownx.arithmatex"],
            extension_configs={"pymdownx.arithmatex": {"generic": True}, "toc": {"toc_depth": "2-2"}},
        )
        markdown_source = normalize_markdown(source.read_text())
        # Python-Markdown treats raw HTML blocks as opaque; opt <details> into
        # Markdown parsing so worked solutions render emphasis, lists and math.
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
            elif source.parent == GUIDES and parts.path.startswith("assets/"):
                target = parts.path
            else:
                raise ValueError(f"Unmapped local link in {source}: {raw}")
            if parts.fragment:
                target += "#" + parts.fragment
            return f'{attr}="{html.escape(target, quote=True)}"'
        body = re.sub(r'(href|src)="([^"]+)"', link, body)
        nav = '<input class="search" aria-label="Search guides" placeholder="Search the full guide…" type="search"><div class="results" aria-live="polite"></div><nav class="nav" aria-label="Study guide">'
        for j, (url, title) in enumerate(zip(urls, titles)):
            active = ' aria-current="page"' if i == j else ""
            label = "START HERE" if j == 0 else "ASSESSMENT · 12 QUESTIONS" if j == 1 else f"GUIDE {j-1:02d}"
            nav += f'<a href="{url}"{active}><span class="num">{label}</span>{html.escape(title)}</a>'
        nav += "</nav>"
        pager = ""
        if i > 0:
            pager += f'<a href="{urls[i-1]}">← {html.escape(titles[i-1])}</a>'
        else:
            pager += "<span></span>"
        if i + 1 < len(urls):
            pager += f'<a href="{urls[i+1]}">{html.escape(titles[i+1])} →</a>'
        label = "Study guide overview" if i == 0 else "Revised assessment" if i == 1 else f"Concept guide {i-1:02d} / 09"
        page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(titles[i])} · Industrial Surface Anomaly Segmentation</title>
<meta name="description" content="Milestone 1 student study guides: industrial surface anomaly segmentation, data understanding, evaluation and baseline modeling.">
<link rel="stylesheet" href="site.css">
<script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}},options:{{ignoreHtmlClass:'tex2jax_ignore',processHtmlClass:'arithmatex'}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"></script>
<script defer src="site.js"></script></head><body>
<a class="skip" href="#main">Skip to content</a><div class="layout">
<aside class="sidebar"><span class="eyebrow">DLGenAI · 26T3</span><a class="brand" href="index.html">Industrial Surface<br>Anomaly Segmentation</a>
<p>A guided path from raw annotations to your first segmentation baseline.</p>
<div class="desktop-nav">{nav}</div><details class="mobile-menu"><summary>Browse & search the guides</summary>{nav}</details>
<footer><a href="https://github.com/manyakaistha/DLGenAI-26T3">View study materials on GitHub ↗</a></footer></aside>
<main id="main"><div class="content"><div class="topline"><span>Milestone 1 · Student learning resources</span><a href="milestone-1.html">View questions ↗</a></div>
<span class="section-label">{label}</span><details class="toc"><summary>On this page</summary>{md.toc}</details>
<article>{body}</article><nav class="pagination" aria-label="Previous and next page">{pager}</nav>
<footer class="endnote">Created by manaykaistha.<br>Study the concepts. Practice on toy examples. Compute your own assessment results.</footer></div></main></div></body></html>'''
        (output / urls[i]).write_text(page)
        parsed = Inspect()
        parsed.feed(body)
        search.append({"title": titles[i], "url": urls[i], "text": " ".join(parsed.text)})
    (output / "search.json").write_text(json.dumps(search, ensure_ascii=False))
    parsed_pages = {}
    for file in output.glob("*.html"):
        parsed = Inspect()
        parsed.feed(file.read_text())
        parsed_pages[file.name] = parsed
    for name, page in parsed_pages.items():
        for link in page.links:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc:
                continue
            target = parts.path or name
            if not (output / target).is_file():
                raise ValueError(f"Broken link from {name}: {link}")
            if parts.fragment and target in parsed_pages and unquote(parts.fragment) not in parsed_pages[target].ids:
                raise ValueError(f"Broken anchor from {name}: {link}")
    print(f"Built and checked {len(parsed_pages)} pages, {len(list((output / 'assets').glob('*')))} images → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/student_pages_preview")
    build(parser.parse_args().output.resolve())
