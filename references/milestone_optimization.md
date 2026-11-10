# Milestones 4 and 5: study guides and assessment boundaries

Sources: `MILESTONE_4.md`, `MILESTONE_5.md`, and `docs/milestone_{4,5}_student_guide/` (overview, seven topic modules, synthetic figures, examples and paired notebook/Python downloads).

Public routes: `/milestones/milestone-4/` and `/milestones/milestone-5/`. Each contains an overview, an assessment (10 and 9 questions respectively), seven concept guides and a walkthrough page. The milestone directory lists both; navigation stays scoped to the selected milestone and provides exits to the directory/homepage.

`build_milestone_four.py` and `build_milestone_five.py` render the resources using the shared garden theme. `build_student_pages.py` integrates them into the full site. Assessment bodies have `data-ai-help="off"`; the shared helper adds no controls there. Practice and task heading exclusions remain active on supporting guides.

The guides explain methods and use distinct synthetic exercises. They contain no computed assessment results or benchmark outcomes. M4 covers subset/counting units, combination-group sampler masses, optimizer-step schedules, strict early stopping, AMP state/activation memory, accumulation, the stipulated 70/20/10 metric, size diagnostics and OFAT seed noise. M5 covers strict probability thresholds, connected-component filtering, inverse-aligned TTA, soft/hard/weighted ensembles, decimal memory/time accounting, validation discipline and Fortran-order RLE integrity. The assessment screenshots define the metric contract for these milestones.

Build/check with:

```sh
uv run --no-project scripts/check_milestone_optimization.py
uv run --no-project scripts/build_student_pages.py --output _site
```

The focused checker validates question counts, assessment AI exclusion markers, local resources/alternative text, scoped exits, stale-resource cleanup and repeatable output. The full build validates every local link and anchor. Notebook demonstrations use synthetic inputs and do not train competition models or supply assessed data-derived outputs.
