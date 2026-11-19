# Report fixes (Overleaf draft)

Record of the edits made to the Overleaf report on 2026-11-19 (sections A
and B). The report compiles; the only warnings are the pre-existing
undefined references to `fig:meeting` and `fig:replan` (figures the draft
no longer has). Each citation fix was checked against the primary
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
