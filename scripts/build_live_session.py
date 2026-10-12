# /// script
# requires-python = ">=3.10"
# dependencies = ["Markdown==3.8.2", "pymdown-extensions==10.16.1"]
# ///
"""Build separate teaching notes and notebook; no milestone source is modified."""
import argparse, html, json, re, shutil
from pathlib import Path
from urllib.parse import urlsplit, unquote
import markdown
from build_student_pages import CSS, JS, CREDIT_ART, Inspect, normalize_markdown
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/live_session_materials'
MODULES=['README','wandb','01_data','02_heads','03_rle','04_geometry','05_texture']
TITLES=['Session overview','W&B notebook lab','1 · Count the right things','2 · Decide what a pixel means','3 · Turn masks into numbers','4 · Preserve geometry','5 · See beyond brightness']
URLS=['index.html','wandb.html','01_data.html','02_heads.html','03_rle.html','04_geometry.html','05_texture.html']

def notebook():
    cells=[]
    for chunk in re.split(r'^# %%', (SOURCE/'wandb_tutorial.py').read_text(),flags=re.M)[1:]:
        first,*lines=chunk.splitlines()
        kind='markdown' if '[markdown]' in first else 'code'
        if kind=='markdown': lines=[re.sub(r'^# ?', '', s) for s in lines]
        src='\n'.join(lines).strip()+'\n'
        cell=dict(cell_type=kind,metadata={},source=src.splitlines(keepends=True))
        if kind=='code': cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    # A separate installation cell lets Colab/Jupyter users actually run setup.
    cells.insert(2,dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=['%pip install torch==2.8.0 numpy==2.2.6 matplotlib==3.10.9 wandb==0.23.1\n']))
    doc=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.11')),nbformat=4,nbformat_minor=5)
    for i,c in enumerate(cells): c['id']=f'session-{i:02d}'
    (SOURCE/'wandb_tutorial.ipynb').write_text(json.dumps(doc,indent=1))

def build(out):
    notebook(); out.mkdir(parents=True,exist_ok=True)
    # Remove a previously generated instructor page from reused output directories.
    (out / "instructor.html").unlink(missing_ok=True)
    shutil.copytree(SOURCE/'figures',out/'figures',dirs_exist_ok=True)
    shutil.copytree(ROOT/'styles/generative',out/'styles/generative',dirs_exist_ok=True)
    for name in ['wandb_tutorial.ipynb','wandb_tutorial.py']: shutil.copy2(SOURCE/name,out/name)
    (out/'site.css').write_text(CSS+'\n'+(ROOT/'styles/generative/session.css').read_text())
    (out/'site.js').write_text(JS)
    (out/'.nojekyll').touch()
    mapping={f'{m}.md':u for m,u in zip(MODULES,URLS)}
    pages={};search=[]
    for i,(module,title,url) in enumerate(zip(MODULES,TITLES,URLS)):
        md=markdown.Markdown(tab_length=2,extensions=['tables','toc','sane_lists','md_in_html','pymdownx.superfences'])
        source=normalize_markdown((SOURCE/f'{module}.md').read_text())
        source=re.sub(r'<details(?=\s|>)','<details markdown="1"',source)
        body=md.convert(source)
        for name,target in mapping.items(): body=body.replace(f'href="{name}"',f'href="{target}"')
        nav='<nav class="nav" aria-label="Session pages">'+''.join(f'<a href="{u}"'+(' aria-current="page"' if i==j else '')+f'>{html.escape(t)}</a>' for j,(t,u) in enumerate(zip(TITLES,URLS)))+'</nav>'
        palette=''.join(f'<button class="theme-option" data-palette="{p}" aria-pressed="{str(p=="paper").lower()}">{p.title()}</button>' for p in ['paper','midnight','moss','butter','blush','mono'])
        pager=(f'<a href="{URLS[i-1]}">← {TITLES[i-1]}</a>' if i else '<span></span>')+(f'<a href="{URLS[i+1]}">{TITLES[i+1]} →</a>' if i+1<len(URLS) else '')
        page=f'''<!doctype html><html lang="en" data-theme="paper"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)} · Pixels, Masks & Experiments</title>
<meta name="description" content="Interactive live lab: experiment tracking, label sparsity, output heads, RLE, sensor geometry and texture.">
<link rel="stylesheet" href="site.css"><link rel="stylesheet" href="styles/generative/theme.css">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,700;1,700&display=swap" rel="stylesheet">
<script>try{{let t=localStorage.getItem('guide-garden-theme');if(['paper','midnight','moss','butter','blush','mono'].includes(t))document.documentElement.dataset.theme=t}}catch{{}}</script>
<script defer src="site.js"></script><script defer src="styles/generative/theme.js"></script><script defer src="styles/generative/live_session_interactions.js"></script></head>
<body data-overview="{str(i==0).lower()}"><div class="reading-progress" aria-hidden="true"></div><a class="skip" href="#main">Skip to content</a><div class="layout"><aside class="sidebar"><span class="eyebrow">DLGenAI · live lab</span><a class="brand" href="index.html">Pixels, Masks<br>& Experiments</a><p>Explore segmentation data, masks and experiment tracking.</p>
<details class="palette-panel"><summary>Choose your palette</summary><div class="theme-options">{palette}</div><button class="motion-toggle" aria-pressed="true">Pause header motion</button></details>
<div class="desktop-nav"><input class="search" type="search" aria-label="Search session" placeholder="Search session…"><div class="results" aria-live="polite"></div>{nav}</div><details class="mobile-menu"><summary>Browse session</summary>{nav}</details><footer><a href="https://manyakaistha.github.io/DLGenAI-26T3/">Milestone study guides ↗</a></footer></aside>
<main id="main"><div class="content"><div class="topline"><span>Industrial surface segmentation</span><a href="wandb_tutorial.ipynb" download>Download notebook ↓</a></div>
<div class="garden-hero"><canvas aria-hidden="true"></canvas><div class="hero-copy"><span class="hero-kicker">Observe. Represent. Measure.</span><p class="hero-title">From pixels<br>to experiments.</p><p class="hero-caption">A live lab in industrial segmentation.</p></div></div>
<span class="section-label">{html.escape(title)}</span><details class="toc"><summary>On this page</summary>{md.toc}</details><article>{body}</article><nav class="pagination">{pager}</nav><footer class="endnote"><span class="author-credit">{CREDIT_ART}<span>Created by manaykaistha.</span>{CREDIT_ART}</span></footer></div></main></div></body></html>'''
        (out/url).write_text(page)
        parsed=Inspect();parsed.feed(page);pages[url]=parsed
        text=Inspect();text.feed(body);search.append(dict(title=title,url=url,text=' '.join(text.text)))
    (out/'search.json').write_text(json.dumps(search))
    for url,page in pages.items():
        for link in page.links:
            p=urlsplit(link)
            if p.scheme or p.netloc: continue
            path=p.path or url
            assert (out/path).is_file(),f'Broken link: {url}: {link}'
            if p.fragment and path in pages: assert unquote(p.fragment) in pages[path].ids,(url,link)
    print(f'Built and link-checked {len(pages)} session pages; notebook: {len(json.loads((SOURCE/"wandb_tutorial.ipynb").read_text())["cells"])} cells → {out}')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'runs/live_session_preview');build(ap.parse_args().output.resolve())
