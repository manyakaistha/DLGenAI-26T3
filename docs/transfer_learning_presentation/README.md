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
  Three miniature blooms occupy a dedicated 112 × 42 px footer canvas (34 px
  high on phones), clear of slide content and navigation. Their scalloped
  silhouettes, spiral linework and leaves match the main site. Seeded geometry,
  Retina scaling, subpixel sway, reduced-motion/static fallback and hidden/
  offscreen pausing keep the decoration quiet. No background rules or pollen.
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

## Presentation design and navigation

The Session 3 redesign follows the installed `ppt-design` skill's audience-first
layout and rendered-review guidance, adapted to the existing interactive HTML
artifact. It preserves the course palette, lesson order and all 24 teaching stops.
The cover combines a synthetic image-to-mask schematic with a four-part roadmap.
Desktop chapter shortcuts jump to Model, Weights, Fine-tune and Evidence; the
cover roadmap provides the same entry points on phones. Explanatory notes sit
beside focused workbenches; comparison, code and MRI pages retain their own roles.

The slide index groups topics by chapter and searches titles, descriptions and
lesson text. Empty searches show a clear message; opening the index resets the
filter. Slider outputs show their current values and reset with their activities.
Selected activity tabs have one consistent highlight. Quizzes show explicit check
or cross markers and feedback. Native keyboard behavior is preserved in inputs,
selects and the palette. A brief opacity entrance respects Motion off and OS
reduced-motion preferences. All visual assets remain embedded or code-native;
there are no additional runtime dependencies.
