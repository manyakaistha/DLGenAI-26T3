# 26 October · Interactive classroom presentation

This is a separate presentation companion to the detailed revision guides. The
study Markdown, seven existing reading labs, and notebooks are unchanged.

- Build just the presentation: `uv run --no-project python scripts/build_transfer_presentation.py`
- Build it with the website: `uv run --no-project scripts/build_student_pages.py`
- Website route: `/session-3/presentation/index.html`; linked from Live Session Resources.
- Edit `index.html` (shell), `presentation.css` (layout), or `presentation.js`
  (slide content and activities). No framework, runtime packages, CDN, or font download.
- The header palette picker follows the main site themes (`paper`, `midnight`,
  `moss`, `butter`, `blush`, `mono`) and remembers the selected theme locally.
  The deck keeps a seeded, theme-aware botanical canvas decorative and clear of
  slide content. Its stems, leaves and spiral blooms share the main study
  site's garden language, with Retina scaling, deterministic geometry,
  line-boil jitter, reduced-motion/static fallback, and offscreen pausing.
- The builder embeds styles, JavaScript and the attributed existing MRI contact
  sheet into one HTML file. Download it from Slide index for offline presentation.
  Links to the full study guides and external dataset still require internet.

## Presenting

Designed on a 1280×720 stage and fitted to 16:9 laptop/projector screens. Use
fullscreen for maximum space. On phones, slides become a scrollable single-column
layout. Browser zoom remains available. Print shows every slide.

Use Next/Previous, left/right arrows, Space, Page Up/Down, Home/End. Sliders and
selects keep their native arrow keys; Space still activates focused buttons.
`I` opens the slide index; `F` requests fullscreen. After a slider or select,
Tab to a button to resume slide-arrow navigation. The index opens any slide and optionally includes the
three interleaved MRI stops; the default 21-slide route skips them. Optional pages
have explicit continue buttons. Hash links restore the current slide.

## Interaction contract

DOM and SVG preserve readable, selectable text and crisp diagrams. All toy arrays,
filter values, examples and curves are fixed. Numerical activities cover shapes,
cross-correlation, decoder concatenation, weight provenance, matched decoder starts,
MRI normalization, SGD switches, BatchNorm buffers, optimizer membership, learning
rates, stable logit BCE, soft Dice, hard threshold Dice and experiment confounders.
Illustrative curves and hand-set filters are labelled; no model training or measured
score is implied. The original figure’s attribution and license remain visible.

The botanical frame uses a bounded, seeded decorative loop; it pauses while hidden
or offscreen and becomes a static fully drawn frame when motion is disabled. The
user-started finite convolution scan and BatchNorm-buffer update remain the only
interactive activity animations. Time-normalized timelines settle immediately
when the slide changes, the tab hides, the menu opens, another activity starts,
or reduced motion is requested. There is no random regeneration or automatic
slide advancement. Motion off supports a static teaching mode and follows OS
preference changes.
