# Instructor notes & deeper practice

## Use one thread through the lesson

“Which pixels on this strip are defects, how do we represent them, and how do we prove the experiment did what we think?” Connect every topic back to that question. Ask students to predict an outcome before changing a control. Reveal the calculation after the prediction.

## Coverage map for Guides 1–5

| Guide | Covered concepts | Live illustration | Deeper follow-up |
|---|---|---|---|
| 1 | Long table; row/image/pixel units; presence; clean frames; multi-label cardinality; imbalance; accuracy paradox; weighting | Toy labels and three actual-data charts | Write a schema validator; compare image prevalence with pixel prevalence |
| 2 | Logits; sigmoid; softmax; background; tensor shapes; pixel overlap audit; target conversion | Overlap map and probability sliders | Construct integer targets after asserting disjointness |
| 3 | Sparse masks; starts/lengths; 1-based indexing; F vs C order; coordinate mapping; empty masks; boundaries/clamping | Paintable numbered grid; real pixel crop; scrambled overlay | Encode/decode round trip; reject malformed tokens; compare compression patterns |
| 4 | Sensor aspect ratio; axis scale factors; anisotropic distortion; aliasing; interpolation; round-trip loss | Resampled crack and sampling-phase chart | Compare nearest/area/bilinear at fixed positions; test tiles without split leakage |
| 5 | Inside/outside statistics; pooled SD; signed d; effect-size interpretation; contrast vs texture; enhancement; receptive field | d calculator; real histogram; equal-histogram shuffle | Compare pooled-pixel and per-image statistics; test linear-transform invariance |

## Prompts with answers

- **W&B:** Where did the “validation” target come from? A separately generated fixed holdout; never the last training batch. Why save locally first? Logs and cloud upload can fail independently of training.
- **Guide 1:** Does sum of class positives equal number of defective images? Only if no image has multiple labels. Does foreground pixel accuracy reveal rare-class detection? Not by itself.
- **Guide 2:** Can an image be multi-label while pixels are exclusive? Yes; place labels in separate regions. Does a softmax head require a universal physical non-overlap law? No; it requires suitable target semantics and evidence.
- **Guide 3:** Height 4, index 6 maps where? x=1,y=1. What is `3 4` as a Python slice? `[2:6]`. Does correct foreground count imply correct decode? No; order can scramble location.
- **Guide 4:** Does aspect-ratio preservation prevent all loss? No; isotropic downsampling also discards detail. Will every 0.64-pixel crack vanish? No; phase/filter/threshold matter.
- **Guide 5:** Can identical means imply identical texture? No. Can identical histograms imply identical texture? Still no. Does small d prove no brightness information? No; distributions may differ beyond the mean.

## Optional exercises (outside the 65-minute walkthrough)

1. Make a 4 × 6 empty mask, a compact patch and a checkerboard. Encode each in F order. Count run pairs and compare textual bytes with 24 binary bits. Verify round trips and an empty string.
2. Decode `23 4` on 4 × 6. Report requested indices, retained indices under clamping, and why strict validation should reject it. Reject `0 3`, `5 -1` and `3 4 7` before writing any buffer.
3. Compare softmax of logits `[0,2,1,-1,-2]` with the four defect sigmoid probabilities. Argmax includes background; sigmoid thresholds can select multiple labels.
4. Resize a 2-pixel crack at ten positions, then restore. Measure Dice and overlay each mask. Compare target shapes 256 × 256 and 400 × 64. Explain shape distortion separately from sampling loss.
5. Permute intensities in a texture patch. Verify mean, SD and histogram remain identical. Compute mean absolute horizontal neighbor difference before/after.
6. Repeat the notebook at a lower learning rate. Record one config change, comparable validation curves, an example failure and a versioned artifact. Explain why threshold selection on validation is not an unbiased test estimate.

## Fallbacks

If W&B cannot authenticate or the network fails, use offline mode and show a previously completed dashboard. Do not announce an offline artifact as uploaded. If setup takes time, show the prepared notebook execution and restore check, then give students the download for later. Browser labs require no accounts and use deterministic synthetic patterns; actual-data figures are static.

## Build and publication

`uv run --no-project scripts/generate_live_session_figures.py --data-root /path/to/data/public` regenerates actual-data figures and provenance from local competition data. `uv run --no-project scripts/build_live_session.py` converts paired notebook cells and builds standalone notes with link checks. `uv run --no-project scripts/build_student_pages.py` also includes the separate session under `session/` and adds a navigation link, without altering milestone guide sources.

Publish only the session source, figures, builder and styles through the isolated guide-only publication checkout described in `references/student_pages.md`. Raw CSVs, raw image directories, model weights and runs remain local. The generated site is a preview until explicitly published.

## Live facilitation script

- **Minute 0:** “By the end, you should be able to trace a label from CSV to pixels, choose a legal model output, and preserve an experiment.”
- **Minute 3:** Show the notebook dataset, then initialize W&B. Point to recipe fields; ask what must stay fixed for a fair comparison.
- **Minute 8:** Run the training cell. The CPU demo finishes quickly; use the time to inspect the dashboard. At epoch 8, loss is falling but Dice is zero. Ask students to explain the high pixel accuracy.
- **Minute 13:** Open the validation table and confusion counts. Show a clean strip and a positive strip. Move to the threshold curve without retraining.
- **Minute 17:** Inspect artifact contents and checkpoint restore. Ask which alias is mutable and which version identifier should be recorded.
- **Minute 20:** Count the four toy strips. Toggle one extra class and have students predict cardinality before the readout changes.
- **Minute 28:** Move the C region onto A. The image-level label stays unchanged while pixel overlap changes. Show the tensor diagram and explain background.
- **Minute 36:** Trace `3 4` down a column boundary. Switch to C order, then show the actual numbered image crop. Explain why matching areas does not prove matching locations.
- **Minute 44:** Check `23 4` on 24 pixels. Show strict rejection versus documented training-only clamping; keep detailed parser code for later.
- **Minute 48:** Select square resize and move the crack by one pixel. Distinguish geometric width from sampled survival. Select isotropic half-size, then native size.
- **Minute 57:** Calculate d=0.5 from equal SDs, then shuffle the same texture intensities. Ask what changed if the histogram did not.
- **Minute 65:** Exit check: one representation bug, one geometry bug, one tracking safeguard. Use remaining time for questions.

## 60-minute route

Use 0–2 introduction, 2–17 W&B, 17–24 data, 24–31 heads, 31–42 RLE, 42–50 geometry, 50–57 texture, and 57–60 exit questions. Every core topic appears, but skip the numerical pooled-SD exercise, live student painting, and detailed decoder walkthrough. Give those as follow-up reading. A first-time hands-on W&B setup plus every derivation will not fit.
