# /// script
# requires-python = ">=3.10"
# dependencies = ["Markdown==3.8.2", "pymdown-extensions==10.16.1"]
# ///
"""Build separate teaching notes and notebook; no milestone source is modified."""
import argparse, ast, hashlib, html, json, re, shutil
from pathlib import Path
from urllib.parse import urlsplit, unquote
import markdown
from build_student_pages import CSS, JS, CREDIT_ART, Inspect, normalize_markdown, version_theme_assets
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/live_session_materials'
MODULES=['README','wandb','01_data','02_heads','03_rle']
TITLES=['Session overview','W&B notebook lab','1 · Count the right things','2 · Decide what a pixel means','3 · Turn masks into numbers']
URLS=['index.html','wandb.html','01_data.html','02_heads.html','03_rle.html']

def notebook():
    path=SOURCE/'wandb_tutorial.py'
    source=path.read_text(encoding="utf-8")
    lines=source.splitlines(keepends=True)
    tree=ast.parse(source)
    first=tree.body[0]
    last=tree.body[-1]
    if not (isinstance(first,ast.Expr) and isinstance(first.value,ast.Constant) and isinstance(first.value.value,str)):
        raise ValueError('W&B tutorial must begin with its introductory docstring')
    if not (isinstance(last,ast.Expr) and isinstance(last.value,ast.Constant) and isinstance(last.value.value,str)):
        raise ValueError('W&B tutorial must end with its run-inspection notes')
    intro=first.value.value
    intro=intro[intro.index('# W&B:'):].strip()
    closing=last.value.value.strip()
    code=''.join(lines[first.end_lineno:last.lineno-1]).strip()+'\n'
    code='\n'.join(line for line in code.splitlines() if line not in ('# Commented out IPython magic to ensure Python compatibility.', '# %pip install scikit-learn==1.7.2 joblib==1.5.2 matplotlib==3.10.9 wandb==0.23.1')).strip()+'\n'
    cells=[]
    for kind,src in [('markdown',intro),('code','%pip install scikit-learn==1.7.2 joblib==1.5.2 matplotlib==3.10.9 wandb==0.23.1'),('code',code),('markdown',closing)]:
        cell=dict(cell_type=kind,metadata={},source=src.splitlines(keepends=True))
        if kind=='code': cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    doc=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.11')),nbformat=4,nbformat_minor=5)
    for i,c in enumerate(cells): c['id']=f'session-{i:02d}'
    (SOURCE/'wandb_tutorial.ipynb').write_text(json.dumps(doc,indent=1), encoding="utf-8")

def build(out, *, source_dir=None, modules=None, titles=None, urls=None,
          downloads=None, extra_files=(), identity=None, nav_groups=None,
          optional_urls=(), page_routes=None):
    # Existing session adapters keep their defaults; new sessions pass explicit data.
    if source_dir is None:
        notebook()
    source_dir = SOURCE if source_dir is None else Path(source_dir)
    modules = MODULES if modules is None else modules
    titles = TITLES if titles is None else titles
    urls = URLS if urls is None else urls
    downloads = ('wandb_tutorial.ipynb', 'wandb_tutorial.py') if downloads is None else downloads
    design = dict(title='Pixels, Masks & Experiments', brand='Pixels, Masks<br>& Experiments',
                  description='Interactive live lab: experiment tracking, label sparsity, output heads and run-length encoding.',
                  intro='Explore segmentation data, masks and experiment tracking.',
                  topline='Industrial surface segmentation', kicker='Observe. Represent. Measure.',
                  hero='From pixels<br>to experiments.', caption='A live lab in industrial segmentation.',
                  download_label='Download notebook ↓')
    design.update(identity or {})
    if not (len(modules) == len(titles) == len(urls)):
        raise ValueError('Session modules, titles and URLs must align')
    optional_urls = set(optional_urls)
    page_routes = page_routes or {}
    if not optional_urls.issubset(urls):
        raise ValueError('Optional session URLs must be rendered pages')
    for page_url, route in page_routes.items():
        if page_url not in urls or len(route) != 2:
            raise ValueError('Session routes must provide a previous/next pair for a rendered page')
        if any(target is not None and target not in urls for target in route):
            raise ValueError('Session routes must point to rendered pages')
    out.mkdir(parents=True,exist_ok=True)
    # Keep optional lessons in source, but never leave their old rendered pages published.
    for hidden in ['04_geometry.html', '05_texture.html', 'instructor.html']:
        (out / hidden).unlink(missing_ok=True)
    # Publish only figures referenced by the active student pages.
    if (out/'figures').exists(): shutil.rmtree(out/'figures')
    (out/'figures').mkdir(parents=True)
    for module in modules:
        text=(source_dir/f'{module}.md').read_text(encoding="utf-8", errors="ignore")
        for asset in re.findall(r'!\[[^\]]*\]\((figures/[^)]+)\)', text):
            shutil.copy2(source_dir/asset, out/asset)
    shutil.copytree(ROOT/'styles/generative',out/'styles/generative',dirs_exist_ok=True)
    for name in (*downloads, *extra_files):
        (out/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir/name, out/name)
    (out/'site.css').write_text(CSS+'\n'+(ROOT/'styles/generative/session.css').read_text(encoding="utf-8"), encoding="utf-8")
    (out/'site.js').write_text(JS, encoding="utf-8")
    (out/'.nojekyll').touch()
    def asset(path):
        version=hashlib.sha256((out/path).read_bytes()).hexdigest()[:12]
        return f'{path}?v={version}'
    extra_head = ''.join(f'<link rel="stylesheet" href="{asset(name)}">' if name.endswith('.css') else f'<script defer src="{asset(name)}"></script>' for name in extra_files if name.endswith(('.css', '.js')))
    mapping={f'{m}.md':u for m,u in zip(modules,urls)}
    pages={};search=[]
    for i,(module,title,url) in enumerate(zip(modules,titles,urls)):
        md=markdown.Markdown(tab_length=2,extensions=['tables','toc','sane_lists','md_in_html','pymdownx.superfences','pymdownx.arithmatex'],extension_configs={'pymdownx.arithmatex': {'generic': True}})
        source=normalize_markdown((source_dir/f'{module}.md').read_text(encoding="utf-8", errors="ignore"))
        source=re.sub(r'<details(?=\s|>)','<details markdown="1"',source)
        body=md.convert(source)
        for name,target in mapping.items():
            body=body.replace(f'href="{name}"',f'href="{target}"')
            body=body.replace(f'href="{name}#',f'href="{target}#')
        nav='<nav class="nav" aria-label="Session pages">'
        for j, (t, u) in enumerate(zip(titles, urls)):
            if nav_groups and j in nav_groups:
                nav += f'<span class="nav-group-label">{html.escape(nav_groups[j])}</span>'
            optional_class = ' class="optional-page"' if u in optional_urls else ''
            nav += f'<a href="{u}"{optional_class}'+(' aria-current="page"' if i==j else '')+f'>{html.escape(t)}</a>'
        nav += '</nav>'
        palette=''.join(f'<button class="theme-option" data-palette="{p}" aria-pressed="{str(p=="paper").lower()}">{p.title()}</button>' for p in ['paper','midnight','moss','butter','blush','mono'])
        if url in page_routes:
            previous_url, next_url = page_routes[url]
            title_by_url = dict(zip(urls, titles))
            pager = f'<a href="{previous_url}">← {html.escape(title_by_url[previous_url])}</a>' if previous_url else '<span></span>'
            if next_url:
                next_label = 'Continue to ' if url in optional_urls else ''
                pager += f'<a href="{next_url}">{next_label}{html.escape(title_by_url[next_url])} →</a>'
        else:
            pager=(f'<a href="{urls[i-1]}">← {html.escape(titles[i-1])}</a>' if i else '<span></span>')
            if i + 1 < len(urls):
                if nav_groups and i > 0 and i + 1 in nav_groups:
                    pager += f'<a href="{urls[0]}">Back to session overview →</a>'
                else:
                    pager += f'<a href="{urls[i+1]}">{html.escape(titles[i+1])} →</a>'
        page=f'''<!doctype html><html lang="en" data-theme="paper"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)} · {html.escape(design["title"])}</title>
<meta name="description" content="{html.escape(design["description"], quote=True)}">
<link rel="stylesheet" href="{asset('site.css')}"><link rel="stylesheet" href="{asset('styles/generative/theme.css')}"><link rel="stylesheet" href="{asset('styles/generative/navigation.css')}">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,700;1,700&display=swap" rel="stylesheet">
<script>try{{let t=localStorage.getItem('guide-garden-theme');if(['paper','midnight','moss','butter','blush','mono'].includes(t))document.documentElement.dataset.theme=t}}catch{{}}</script>
<script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}},options:{{ignoreHtmlClass:'tex2jax_ignore',processHtmlClass:'arithmatex'}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"></script>
<script defer src="{asset('site.js')}"></script><script defer src="{asset('styles/generative/theme.js')}"></script><script defer src="{asset('styles/generative/live_session_interactions.js')}"></script>{extra_head}</head>
<body data-overview="{str(i==0).lower()}"><div class="reading-progress" aria-hidden="true"></div><a class="skip" href="#main">Skip to content</a><div class="layout"><aside class="sidebar"><span class="eyebrow">DLGenAI · live lab</span><a class="brand" href="index.html">{design["brand"]}</a><p>{html.escape(design["intro"])}</p>
<details class="palette-panel"><summary>Choose your palette</summary><div class="theme-options">{palette}</div><button class="motion-toggle" aria-pressed="true">Pause header motion</button></details>
<div class="desktop-nav"><input class="search" type="search" aria-label="Search session" placeholder="Search session…"><div class="results" aria-live="polite"></div>{nav}</div><details class="mobile-menu"><summary>Browse session</summary>{nav}</details><footer><a href="https://manyakaistha.github.io/DLGenAI-26T3/">Milestone study guides ↗</a></footer></aside>
<main id="main"><div class="content"><div class="topline"><span>{html.escape(design["topline"])}</span><a href="{downloads[0]}" download>{html.escape(design["download_label"])}</a></div>
<div class="garden-hero"><canvas aria-hidden="true"></canvas><div class="hero-copy"><span class="hero-kicker">{html.escape(design["kicker"])}</span><p class="hero-title">{design["hero"]}</p><p class="hero-caption">{html.escape(design["caption"])}</p></div></div>
<span class="section-label">{html.escape(title)}</span><details class="toc"><summary>On this page</summary>{md.toc}</details><article>{body}</article><nav class="pagination">{pager}</nav><footer class="endnote"><span class="author-credit">{CREDIT_ART}<span>Created by manaykaistha.</span>{CREDIT_ART}</span></footer></div></main></div></body></html>'''
        (out/url).write_text(page, encoding="utf-8")
        parsed=Inspect();parsed.feed(page);pages[url]=parsed
        text=Inspect();text.feed(body);search.append(dict(title=title,url=url,text=' '.join(text.text)))
    (out/'search.json').write_text(json.dumps(search), encoding="utf-8")
    for url,page in pages.items():
        for link in page.links:
            p=urlsplit(link)
            if p.scheme or p.netloc: continue
            path=p.path or url
            assert (out/path).is_file(),f'Broken link: {url}: {link}'
            if p.fragment and path in pages: assert unquote(p.fragment) in pages[path].ids,(url,link)
    version_theme_assets(out)
    print(f'Built and link-checked {len(pages)} session pages; notebook: {len(json.loads((source_dir/downloads[0]).read_text(encoding="utf-8"))["cells"])} cells → {out}')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'runs/live_session_preview');build(ap.parse_args().output.resolve())
