# Report fixes (Overleaf draft)

Record of the edits made to the Overleaf report: 2026-11-19 (sections A
and B) and 2026-11-22 (section C). After section C the report compiles with
no errors, undefined references or overfull boxes (only underfull lines in
the bibliography). Each citation fix was checked against the primary
source (PDF text, the IEEE Xplore abstract, or Crossref metadata).

## A. Citation fixes (applied 2026-11-19)

1. **Kim et al. bib month:** "May 2014" → "Oct. 2014" (Crossref: issued
   2014-10, vol. 30, no. 5, pp. 1049–1065).
2. **"Forgiving gripper" premise removed** (abstract and intro). The cited
   catching papers say the opposite: Salehian et al. write that any small
   imprecision may let the object escape the fingers, and Kim et al. that
   the required hand orientation tremendously limits the possible catching
   configurations. The abstract and intro now say catching is usually
   scored end to end, as a completed catch.
3. **"Three steps"** is now cited to Kim alone. Salehian's abstract
   describes catching as two tasks: precise estimation of the object's
   motion and control of the robot's motion.
4. **Critique of Kim softened:** "Their headline measure is catch success,
   which combines the stages, although they also report trajectory-
   prediction error" (Kim et al. report prediction error per object, Fig. 12).
5. **Fitter & Kuchenbecker:** the unsupported claims ("the human partner
   adapts to the robot", "highlights that alignment and timing matter") are
   replaced with what the paper's abstract (IEEE Xplore) states. It recorded
   seven pairs of people clapping and found their hand motions were well
   described by sinusoids. It programmed a Baxter to clap with a human
   partner, detecting hand impacts with wrist-mounted accelerometers.
6. **Fitter bib:** added "Aug." to the date (conference 26–31 Aug. 2016).
   The venue and Drake URL were already in the draft.

## B. Text updated to the current setup (applied 2026-11-19)

The draft described the old design. These passages were rewritten to match
the code (docs/setup.md):

- **Clasp removed** (abstract, intro, contributions, System, Fig. 1 and 2
  captions, Table I, Fig. 3): rigid flat hands with fixed fingers; a trial
  ends at the first palm touch or collision; success = the palms touch
  antiparallel (orientation within 15°, marked `\todo` as a placeholder).
  "Contact and Clasp" became "Contact and Scoring": touch detected from
  ground-truth geometry, all scores computed from the logged ground truth.
- **Checkpoints removed** everywhere (abstract, contributions, hypothesis,
  System, Pipeline, Trajectory Optimization, Metrics, Table I, Fig. 1, Fig. 4).
- **Velocity matching removed** as a requirement (abstract, contribution 2,
  Pipeline figure, Trajectory Optimization). The Salehian paragraph now says
  relative palm speed at contact is reported as a score.
- **Simulation environment:** randomized layouts, start configurations and
  meeting point with an IK feasibility check; partner path and peak speed
  randomized and executed within the partner iiwa's limits; D435-style head
  camera with depth noise and latency; noisy pose measurements stand in
  until the perception pipeline exists. ("Perturbed by added noise" on the
  partner trajectory was dropped: the path is smooth, the measurements are
  noisy.)
- **Experiments:** paired seeds; partner peak speed 0.15–0.9 m/s drawn per
  scenario; three noise levels × 150 paired seeds; a separate 300-seed run
  for success vs. speed (`\todo` to confirm counts and the replanning rate).
- **Metrics:** success rate with Wilson intervals; position, orientation
  (tilt and finger rotation) and timing error at contact; arm clearance,
  failure type, relative palm speed; McNemar and paired bootstrap.

Not changed: the method's estimation and meeting-point text, the results
and discussion placeholders, the AI acknowledgment, and the undefined
`fig:meeting` / `fig:replan` references.

## C. Rewrite to the perception focus (applied 2026-11-22)

The trajectory planner is no longer the project's focus (`docs/phase2_plan.md`),
so everything after the author block except the AI acknowledgment was
rewritten to the current setup and plan. The same changes went into the
proposal doc.

- **Title:** "... with Motion Planning" → "Angle and Alignment Matching for
  Robotic Hands from Depth Images".
- **Abstract, introduction, contributions:** the arm sees the partner only
  through the head camera; perception is the bottleneck; the hypothesis is
  that a depth policy distilled from a latency-matched privileged teacher
  beats the classical depth pipeline on paired evaluation scenarios, with
  the cost of perception and out-of-distribution results (`wide`) reported
  separately. Planner hypotheses (H1, H2) removed.
- **Related work:** adds Bäuml et al. 2010, Agile Catching (Abeyruwan et al.
  2023), visual servoing (Chaumette & Hutchinson 2006, 2007; Whitney 1969;
  Nakamura & Hanafusa 1986; Wampler 1986), Christen et al. 2019 and 2023,
  GenH2R, and the teacher–student line (Chen et al. 2019; Ross et al. 2011;
  Agarwal et al. 2022; Zhuang et al. 2023; Weihs et al. 2021).
- **System and method:** the camera model; train / validation / evaluation
  seed splits; the methods ladder (new Fig. `fig:methods` placeholder); the
  classical depth baseline (`dap/perception.py`); the teacher and student
  (details left as `\todo`). Meeting-point selection and trajectory
  optimization removed, which also removes the undefined `fig:meeting` and
  `fig:replan` references.
- **Experiments:** presets, metrics, paired tests with Holm's correction,
  and a baseline-check table (Table `tab:check`) with the validation-seed
  numbers from `docs/setup.md` section 1b, labelled as a check, not a
  result. The results table and figures now compare methods instead of
  plan-once vs. replan.
- **Bibliography:** 27 entries. Bibliographic details were checked against
  Crossref (DOIs, volumes, pages, author initials) or the arXiv listings in
  `research_notes/`; Holm 1979 has no DOI and is cited as in
  `docs/setup.md`.

