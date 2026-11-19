# Report fixes (Overleaf draft)

Working note on the Overleaf report. Section A was **applied in Overleaf
on 2026-11-19** (the report compiles; the only warnings are the
pre-existing undefined references to `fig:meeting` and `fig:replan`).
Section B is still open. Each citation fix was checked against the primary
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

## B. Text that no longer matches the setup (team decisions since the draft)

The task is now purely kinematic and checkpoints are gone. These passages
describe the old design:

- **Clasp / hinged finger plate / compliant controller:** abstract ("and then
  clasp it", "clasp success"), intro ("and then clasp it", "before the clasp
  closes"), System ("hinged finger plate that can close"), the whole
  "Contact and Clasp" subsection, Metrics (1), Fig. 1(b), Table I, Fig. 3.
  Now: rigid flat hands; the trial ends at the first palm contact; success =
  contact with the palms antiparallel (rotation error ≤ 15°, placeholder).
- **Checkpoints:** abstract, contribution 1 and 3, hypothesis ("large numbers
  of checkpoints"), System ("places a few checkpoints"), Method (iii) and the
  trajectory-optimization constraints and fallback, Metrics (2), Fig. 1(a),
  Fig. 4. All removed.
- **Velocity matching:** abstract ("hand velocities matched"), contribution 2
  ("velocity-matched"), Salehian paragraph ("We apply the same idea by
  matching the partner hand's velocity"), Metrics (3). Relative palm speed at
  contact is now only a reported score (weight 0 in the placeholder loss).
- **Partner motion:** "perturbed by added noise" — the partner follows a
  smooth randomized path executed by the partner iiwa; the *measurements*
  are noisy. Partner speed is drawn per trial (peak palm speed 0.15–0.9 m/s),
  not a fixed set of speed conditions.
- **Observations:** the system now has a head-mounted RGB-D camera (D435
  model with depth noise); a noisy pose sensor stands in until the
  perception pipeline exists.
- **Experiments:** conditions are planner × sensor-noise level, paired on
  the same seeds (150 per condition for H1; 300 seeds analysed against
  partner speed for H2), with exact McNemar tests, Wilson intervals and a
  paired bootstrap on the loss (docs/setup.md section 9).
