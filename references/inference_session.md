# Inference live session · 9 November 2026

Session 5 is a 60-minute Milestone 5 lesson: TTA spatial inversion, EMA/SWA versus prediction ensembles, threshold decisions and component filtering, leakage-free validation, runtime and RLE submission integrity.

- Sources: `docs/inference_session/README.md`, `01_pipeline.md`–`07_reference.md`, paired `inference_lab.py` / `inference_lab.ipynb`, `activities.js`, `presentation/{index.html,presentation.css,presentation.js,content.js}`.
- Deck: 23 core slides + one optional scale/tile slide. Existing course palette/16:9 shell, keyboard/index/fullscreen/offline-save controls, four deterministic native-control activities and exit quiz. No real-model results or private-data findings.
- Notebook: NumPy 2.2.6 only, 15 cells, CPU, H8/W50 synthetic tensors preserving aspect ratio. TTA aligned/broken and positional bias, scalar EMA, nonlinear parameter/prediction average contrast, threshold metrics, 4/8-connected components, diagnostic empty support and F-order RLE round trips. No training/data downloads/outputs in delivered notebook.
- Metric caveat: published 70/20/10 weights retained, unspecified organizer reductions not invented. Notebook positive-pair Dice and empty-pair specificity explicitly diagnostics. EMA buffer policy, successful optimizer steps and OOF-versus-final-ensemble distinction are explicit.
- Instructor file `presentation/SPEAKER_SCRIPT.md` is local only and excluded from publication, navigation, downloads and HTML.
- Build notebook: `uv run --no-project python scripts/build_inference_notebook.py`.
- Run lab: `uv run --no-project docs/inference_session/inference_lab.py`.
- Build standalone: `uv run --no-project scripts/build_inference_session.py` (default `runs/inference_preview/`).
- Full build: `uv run --no-project scripts/build_student_pages.py`; hub/main integration creates `/session-5/` and `/session-5/presentation/`.
- Publish only the new student sources and builders, this focused reference, and surgical hub/main changes to isolated `runs/student_pages_publish`. Never overwrite that checkout's newer milestone routes or copy project data/history/instructor script.

## Verification

- All lab code cells executed in NumPy 2.2.6; output-free notebook retained. JS syntax checks passed.
- Headless Chromium: all 25 slides checked at 1280×720, 390×844 and 320×740, no horizontal overflow or desktop footer overlap. Six palettes checked. TTA inverse/reset, EMA steps/decay, blend native keyboard input, threshold arithmetic, quiz, index/search and arrow navigation passed. No JavaScript errors.
- Full isolated publication build validates 168 HTML pages and local links/anchors. QA evidence is private under `runs/inference_checks/`; projector/physical-device behavior not tested.

- Slide 2 (the one-hour agenda) removed at user request. The cover now begins directly at the inference contract; index and counters derive from the remaining 23 core + one optional slides.
