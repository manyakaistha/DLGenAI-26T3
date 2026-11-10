# /// script
# requires-python = ">=3.10"
# dependencies = ["Markdown==3.8.2", "pymdown-extensions==10.16.1", "beautifulsoup4==4.14.2"]
# ///
"""Check M4/M5 assessment boundaries, resource rebasing and deterministic builds."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from bs4 import BeautifulSoup
import build_milestone_four as four
import build_milestone_five as five


def manifest(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def check(builder, number, questions):
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        builder(out)
        initial = manifest(out)
        assessment = BeautifulSoup((out / f'milestone-{number}.html').read_text(), 'html.parser')
        assert assessment.body['data-ai-help'] == 'off'
        assert len([h for h in assessment.select('article h2') if h.get_text().startswith('Question ')]) == questions
        assert len(json.loads((out / 'search.json').read_text())) == 10
        for f in out.glob('*.html'):
            doc = BeautifulSoup(f.read_text(), 'html.parser')
            assert doc.select_one('a[href="../index.html"]')
            assert doc.select_one('a[href="../../index.html"]')
            for image in doc.select('article img'):
                assert image.get('alt', '').strip(), f'{f}: missing image alternative'
                assert (out / image['src']).is_file()
            for a in doc.select('article a[href]'):
                href = a['href'].split('#')[0].split('?')[0]
                if href and not href.startswith(('https:', 'http:', 'mailto:')):
                    assert (out / href).is_file(), f'{f}: missing {href}'
        (out / 'assets/stale.txt').write_text('stale')
        builder(out)
        assert manifest(out) == initial, 'Repeated build changes bytes or retains stale resources'
    print(f'PASS M{number}: {questions} questions, AI disabled on assessment, resources, alt text, navigation, repeatability')


if __name__ == '__main__':
    check(four.build_milestone_four, 4, 10)
    check(five.build_milestone_five, 5, 9)
