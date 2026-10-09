# Generative guide style: feasibility study

Status: implemented as an isolated style experiment. All lesson Markdown and starter downloads are unchanged. The live website and main branch are unchanged.

Local branch: `experiment/generative-guide-style`, based on publication commit `b115673`. Isolated checkout: `runs/student_pages_style_experiment`. The experiment is maintained on its own branch; it is not deployed to GitHub Pages.

## What is feasible

The supplied Type Garden reference is plain HTML, CSS and browser JavaScript using Canvas 2D. Its design can be adapted to the existing static guide builder without a server, framework or change to the Markdown lesson content. Fonts currently come from Google Fonts; provide fallback fonts or locally hosted font assets if appropriate.

Use Playfair Display for large headings, DM Mono for small labels/code, and a comfortable reading font for body text. Start with the reference's Paper palette (`#F3EEE4` background, `#111111` text, `#FF3B1F` and `#1C2B8F` accents). Treat decorative colors separately from text/link colors and verify contrast. Other palettes can be optional themes.

Keep headings, lessons, equations, code and navigation in semantic HTML. Add a bounded decorative canvas to the overview header and small motion to cards or controls. Illustrations could use the same curve/animation system for pixel grids, mask overlays or threshold demonstrations using invented practice data, without disclosing milestone answers.

## How the math works

- Seeded Mulberry32 generates stable variation in stems, petals and positions. Generate geometry once, then reuse it each frame.
- Timed growth uses clamped normalized progress `u = (now - birth) / duration`. Cubic ease-out `1 - (1-u)^3` slows motion on arrival.
- The visual spring `1 - exp(-k*t)*cos(w*t)` combines an exponentially shrinking oscillation with an approach to 1. Here `k` is decay rate, not physical stiffness. It is an easing approximation; it does not generally have zero initial velocity or finish at exactly 1 at a finite endpoint.
- Smoothstep `3*u^2 - 2*u^3` has zero slope at both ends. It softens cursor influence as distance changes.
- Curves are sampled paths, including cubic Bézier interpolation. Drawing an increasing subrange makes stems grow; local tangent angles orient leaves and flowers.
- Sine/cosine offsets produce sway and petal shapes. Small deterministic jitter stepped every 120 ms gives a hand-drawn appearance. Background, text and foreground passes create depth.
- Canvas backing dimensions scale by device-pixel ratio, while drawing coordinates stay in CSS pixels. Text measurement waits for fonts. Integrated motion uses bounded delta time.

## Adaptation needed

The full-screen reference locks page scrolling, takes input focus automatically, and continuously animates. Those behaviors suit an art tool but need to be removed or contained for a reading site. The reference has some unseeded generation, and the starter does not implement all behaviors claimed by the original skill. Neither is evidence of cross-device perfection.

Provide reduced-motion and static fallbacks, keyboard-accessible controls, normal zoom and selection, touch scrolling, and pause rendering while hidden/offscreen. The hidden sentinel input belongs only in an optional typing demo, not normal guide navigation. Video/ZIP export is unnecessary for the guide styling experiment.

## Proposed experiment

1. On this branch, introduce an optional style module and separate preview output; leave the main publishing workflow alone.
2. Prototype the overview plus one representative guide containing equations, code, tables and worked-solution panels. Reuse the existing Markdown conversion.
3. Add a Paper theme, editorial headings and one restrained generative header. Keep the lessons unchanged.
4. Verify existing links/math, desktop/mobile layout, contrast, keyboard behavior, reduced motion and performance before deciding whether to extend the experiment.
5. Review a local preview first. Branch publishing, merging and live deployment are separate decisions.

Installed reusable skill: `/Users/manyakaistha/.codex/skills/generative-micro-interactions/SKILL.md`. Invoke with `$generative-micro-interactions`.

## Implemented version

- `styles/generative/theme.css`: Paper, Midnight, Moss, Butter, Blush and Mono palettes, editorial typography, responsive navigation and print styling.
- `styles/generative/theme.js`: seeded botanical header, DPR scaling, cubic growth, spring bloom, persistent palette and motion controls, reduced-motion static rendering, offscreen/hidden pause and reading progress.
- Branch-only builder integration loads the module on all 12 pages. No new build dependencies.
- The publication workflow has an explicit main-branch job guard, so this branch cannot deploy to the production Pages environment through a manual run.

Build: `uv run --no-project scripts/build_student_pages.py --output _site`
Preview: `uv run --no-project python -m http.server 8767 --bind 127.0.0.1 --directory _site`

Verification: all 12 lesson HTML bodies and both downloadable starter files match the original build byte for byte. Built-in links/anchors pass. Browser checks found no math errors, missing images, raw Markdown markers or document overflow on all 12 desktop pages and all 12 pages in a 390-pixel browser iframe. This is responsive layout verification, not a physical-device touch test. Palette persistence, motion pause and search work. Body, muted and link colors exceed 4.5:1 contrast in all six themes. JavaScript syntax check passes.

## Small garden details

The initial design is preserved in commit `8c760da`. Three tiny Canvas butterflies now flutter near the flowers/title, using the existing motion toggle and visibility/reduced-motion behavior. Small decorative SVG flowers flank the author attribution on each page; colors follow the selected palette and the credit remains semantic text.
