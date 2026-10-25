"""Resource directories and navigation for the combined student website."""
import html
import json
import os
import re
import shutil
from pathlib import Path
from urllib.parse import unquote, urlsplit

MILESTONES = [
    ('milestones/milestone-1/index.html', 'Milestone 1', 'Understand the data & build a baseline',
     'Nine concept guides on annotations, masks, evaluation and modeling, with 12 assessment questions and a Q11–Q12 starter notebook.'),
]
SESSIONS = [
    ('session/index.html', '12 October 2026 · Session 1', 'Pixels, Masks & Experiments',
     'Revisit label counts, output heads and run-length encoding. Practice experiment tracking in the W&B notebook lab.'),
    ('session-2/index.html', '19 October 2026 · Session 2', 'Dataset, DataLoader & Augmentations',
     'Follow a sample from CSV row to training batch. Explore the split, paired transformations, interactive labs and notebook.'),
    ('session-3/index.html', '26 October 2026 · Session 3', 'MRI Transfer Learning: From a Baseline to a Pretrained Encoder',
     'Explore patient-level splits, MRI sequence channels, pretrained features, freezing and fine-tuning, then run a fair initialization comparison.'),
]



def relative(target, current):
    return os.path.relpath(target, Path(current).parent).replace(os.sep, '/')


def exits(parent, home):
    return f'<nav class="nav section-exits" aria-label="Leave this session"><a href="{parent}">← All live session resources</a><a href="{home}">← App homepage</a></nav>'


def cards(items, current, action):
    return '<div class="resource-list">' + ''.join(
        f'<a class="resource-card" href="{relative(url, current)}"><span class="resource-meta">{html.escape(meta)}</span>'
        f'<h2>{html.escape(title)}</h2><p>{html.escape(description)}</p><span class="resource-action">{action} →</span></a>'
        for url, meta, title, description in items) + '</div>'


def build_hub(output):
    milestone = output / 'milestones/milestone-1'
    template = (milestone / 'index.html').read_text()
    # Directory pages share the existing accessible shell, palettes and garden header.
    directories = [
        ('index.html', 'Student Resource Hub', 'Choose your next step.',
         'Study guides and live session resources for Industrial Surface Anomaly Segmentation.',
         [('milestones/index.html', 'Study at your own pace', 'Milestone Guides', 'Find the guides, assessment questions and starter resources for your milestone.'),
          ('live-sessions/index.html', 'Revisit a live lesson', 'Live Session Resources', 'Browse session notes, interactive activities and notebook labs by session.')],
         'Open resources', 'Choose a section below. Use Milestone Guides for assessment preparation, or Live Session Resources to revisit a class and practice its activities.'),
        ('milestones/index.html', 'Milestone Guides', 'One milestone at a time.',
         'Find your milestone, understand its goals, and work through its study resources.', MILESTONES,
         'Open Milestone 1', 'Choose a milestone below. Start with its overview, read the concept guides, then use the assessment questions to apply what you have learned. Guides, questions and starter code stay together inside that milestone.'),
        ('live-sessions/index.html', 'Live Session Resources', 'Return to the lesson.',
         'Notes, guides and hands-on activities, organised by live session.', SESSIONS,
         'Open session', 'Choose the session you want to revisit. Its overview introduces the topics and links to the detailed notes, interactive guides and notebook. Use the session menu to move between resources.'),
    ]
    for current, title, hero, description, items, action, instruction in directories:
        page = template.replace('data-ai-help="study"', 'data-ai-help="off"')
        # Rebase the shell's assets and links before substituting directory content.
        page = re.sub(r'(href|src)="([^"#]+)"', lambda m: m[0] if urlsplit(m[2]).scheme or urlsplit(m[2]).netloc else f'{m[1]}="{relative("milestones/milestone-1/" + m[2], current)}"', page)
        nav_items = [('index.html', 'App homepage'), ('milestones/index.html', 'Milestone Guides')]
        if current == 'milestones/index.html':
            nav_items += [(url, meta) for url, meta, _, _ in MILESTONES]
        nav_items += [('live-sessions/index.html', 'Live Session Resources')]
        if current == 'live-sessions/index.html':
            nav_items += [(url, title) for url, _, title, _ in SESSIONS]
        nav = '<nav class="nav" aria-label="Resource navigation">' + ''.join(
            f'<a href="{relative(url, current)}"' + (' aria-current="page"' if url == current else '') + f'>{html.escape(label)}</a>' for url, label in nav_items) + '</nav>'
        page = re.sub(r'<div class="desktop-nav">.*?</details>\s*<footer>', lambda _: f'<div class="desktop-nav">{nav}</div><details class="mobile-menu"><summary>Browse resources</summary>{nav}</details><footer>', page, count=1, flags=re.S)
        page = re.sub(r'<title>.*?</title>', f'<title>{title} · DLGenAI</title>', page)
        page = re.sub(r'<meta name="description"[^>]+>', f'<meta name="description" content="{html.escape(description)}">', page)
        page = re.sub(r'<a class="brand".*?</a>', f'<a class="brand" href="{relative("index.html", current)}">Industrial Surface<br>Anomaly Segmentation</a>', page)
        page = page.replace('A guided path from raw annotations to your first segmentation baseline.', 'Your study materials, organised by milestone and live session.')
        page = re.sub(r'<div class="topline">.*?</div>', '<div class="topline"><span>DLGenAI · 26T3</span><span>Student learning resources</span></div>', page)
        page = re.sub(r'<p class="hero-title">.*?</p>', f'<p class="hero-title">{hero}</p>', page)
        page = page.replace('A closer look at the details', 'A place to learn').replace('Observe. Understand. Experiment.', 'Read. Practice. Revisit.')
        body = f'<h1>{title}</h1><p>{description}</p><div class="hero-note">{instruction}</div>' + cards(items, current, action)
        page = re.sub(r'<span class="section-label">.*?<footer class="endnote">', lambda _: f'<article>{body}</article><footer class="endnote">', page, flags=re.S)
        page = page.replace('Study the concepts. Practice on toy examples. Compute your own assessment results.', 'Learn the concepts. Try the activities. Build your understanding.')
        path = output / current
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page)
    # Keep session URLs stable; add exits inside both desktop and mobile menus.
    for folder in (Path(item[0]).parts[0] for item in SESSIONS):
        for path in (output / folder).glob('*.html'):
            page = path.read_text()
            page = page.replace('</nav>', '</nav>' + exits('../live-sessions/index.html', '../index.html'), 2)
            page = page.replace('https://manyakaistha.github.io/DLGenAI-26T3/', '../index.html').replace('Milestone study guides ↗', 'App homepage')
            path.write_text(page)
    # Retain old deep links, including fragments, and direct download URLs.
    for path in milestone.glob('*.html'):
        if path.name == 'index.html':
            continue
        target = 'milestones/milestone-1/' + path.name
        (output / path.name).write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Guide moved</title><meta http-equiv="refresh" content="0;url={target}"><script>location.replace({json.dumps(target)}+location.search+location.hash)</script></head><body><p>This guide has moved. <a href="{target}">Continue to Milestone 1</a>.</p></body></html>''')
    for folder in ('downloads', 'assets'):
        shutil.copytree(milestone / folder, output / folder, dirs_exist_ok=True)


def validate_site(output):
    """Check local links and anchors across the entire generated website."""
    from build_student_pages import Inspect
    pages = {}
    for path in output.rglob('*.html'):
        parsed = Inspect()
        parsed.feed(path.read_text())
        pages[path.resolve()] = parsed
    for path, page in pages.items():
        for link in page.links:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc:
                continue
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if target.is_dir():
                target /= 'index.html'
            if not target.is_relative_to(output.resolve()) or not target.is_file():
                raise ValueError(f'Broken local link from {path}: {link}')
            if parts.fragment and target in pages and unquote(parts.fragment) not in pages[target].ids:
                raise ValueError(f'Broken anchor from {path}: {link}')
    print(f'Checked all local links and anchors across {len(pages)} pages → {output}')
