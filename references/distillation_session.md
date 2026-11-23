# Distillation, pseudo-labeling & data-centric live session · 23 November 2026

Session 6 is a 40-minute live lesson covering ensemble distillation (soft targets, temperature scaling, OOF teacher rule), pseudo-labeling & Noisy Student self-training (confidence gating, confirmation bias, leakage rules), data-centric tricks (domain shift, resolution/aspect ratio, copy-paste augmentation & edge-halo trap, annotation jitter), and probability calibration (temperature / Platt scaling).

- Sources: `docs/distillation_session/README.md`, `01_distillation.md`–`05_reference.md`, `activities.js`, `presentation/{index.html,presentation.css,presentation.js,content.js}`.
- Deck: 18 core slides + 1 optional multi-generation slide, using the shared Type Garden 6-palette 16:9 shell, keyboard/index/fullscreen/offline-save controls, 4 animated interactive labs (temperature slider, teacher→student flow animation, confidence-gate simulator, calibration toggle), and exit quiz.
- Local instructor script: `docs/distillation_session/presentation/SPEAKER_SCRIPT.md` (excluded from public HTML/navigation).
- Standalone build: `uv run --no-project scripts/build_distillation_session.py` (default `runs/distillation_preview/`).
- Full site build: `uv run --no-project scripts/build_student_pages.py --output runs/student_pages_preview` creates `/session-6/` and `/session-6/presentation/` and lists Session 6 in `/live-sessions/`.
