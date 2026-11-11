# Deployment Guide (Hugging Face Space)

Student-facing, 20-minute guide: turn trained weights into a Gradio app on a free CPU Hugging Face Space. Upload an image, get predicted masks overlaid. Published at `/deployment/` and listed first under Live Session Resources.

- Source: `docs/deployment_guide/` — `README.md` (overview), `01_prepare.md`, `02_publish.md`, `figures/` (3 synthetic PNGs), `space_template/` (`app.py`, `model.py`, `requirements.txt`, `README.md` Space header).
- Builder: `scripts/build_deployment_session.py` (reuses `build_live_session.render`; zips `space_template/` into `surface_defect_space.zip`, the page's download). Hub entry is the first item of `SESSIONS` in `scripts/build_student_hub.py`; `build_student_pages.py` calls the builder. Local: `uv run --no-project scripts/build_deployment_session.py --output runs/deployment_preview`.
- Figures: `uv run --no-project scripts/generate_deployment_figures.py` (synthetic only; no competition data).
- Template contract: 4 independent sigmoid channels (a–d), grayscale 800×128 input, `THRESHOLD`, `preprocess()` flagged as "must match training"; state_dict `weights.pt` loaded on CPU. Gradio pinned via `sdk_version: 6.31.0` in the Space README.
- Verified 11 Nov 2026: template launched locally with Gradio 6.31.0, called through `gradio_client`, returned an 800×128 RGB overlay. Full-site build checks 191 pages.
- The page has no Ask-AI controls. Published to GitHub Pages at /deployment/ on the user's instruction (11 Nov 2026).
