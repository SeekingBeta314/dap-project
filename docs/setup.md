# Problem setup

This document records the simulated problem: scene, randomization, the
"a dap is possible" check, the partner's motion, the head camera, depth
noise, and the RL environment. It gives every parameter and where it comes
from. Nothing here is final.

Status labels on sources: **V** means verified in the primary paper, datasheet
or released code. **S** means secondary (a review, or a paper citing the
original). **U** means unverified (only the abstract was available). **E**
means measured in this repo (Drake 1.58, SNOPT, Apple silicon). **Ours**
means our own engineering choice.

Code map:

| Topic | Module |
| --- | --- |
| Scene, layout, collision queries | `dap/scene.py` |
| Head camera, frame timing | `dap/camera.py` |
| Depth noise | `dap/depth_noise.py` |
| Feasibility check | `dap/feasibility.py` |
| Partner motion | `dap/partner.py` |
| Scenario sampler | `dap/scenario.py` |
| Trial loop (arm limits, logging) | `dap/trial.py` |
| Events, scores, loss | `dap/scoring.py` |
| Result record | `dap/metrics.py` |
| Statistics | `dap/stats.py`, `scripts/analyze.py` |
| Scripted references | `dap/baselines.py` |
| Gymnasium env | `dap/envs/` |

---

## 1. Task and meeting type

Our iiwa must meet the partner iiwa's hand palm to palm. The task is **purely
kinematic**: the hands are rigid (a palm plus fixed fingers, no moving
parts), there is no clasp and no force, and the trial ends at the touch. For
now the meeting type is fixed. **The palms are parallel and face each other, and
the fingers point in opposite directions.** In palm frames (+z is the palm
normal, +x points toward the fingers) the partner palm frame is our palm
frame rotated by `RotY(pi)`. That gives `x_partner = -x_ours`,
`z_partner = -z_ours` and `y_partner = y_ours`.

The finger plate sits inside the palm's thickness, 5 mm behind the palm face
(`models/dap_hand.sdf`), so in a correct meeting only the two palm boxes
touch.

## 1a. Events: contact and collision (`dap/scoring.py`)

One rule decides how a trial ends. The trial loop and the RL env run it live
(`EventMonitor`); the scorer runs the same per-sample rule over the logged
trial (`evaluate`), and the trial raises an error if the two ever disagree.
`tests/test_scoring.py` checks the agreement on 300 random logs covering all
three outcomes, and `tests/test_baselines.py` checks that the env ends where
a trial ends (same event, same time, same arm configuration) for timeouts,
collisions and a replayed oracle dap. Everything comes from ground truth: palm poses
from forward kinematics of both arms, and signed distances from Drake's
scene graph.

Geometry at one sample, with our palm face center `p_o` and frame `R_o`, and
the partner's `p_p`, `R_p` (partner normal `n` points toward us):

| Quantity | Definition |
| --- | --- |
| `gap` | `(p_o − p_p) · n`: our face center's signed distance in front of the partner face plane |
| `in_plane` (`pos_err`) | `|(p_o − p_p) − gap · n|`: center offset along the palm plane |
| `rot_err` | angle of `RotY(pi)ᵀ R_oᵀ R_p`: the full rotation error from antiparallel (0 = normals opposite **and** fingers opposite) |
| `angle_err` | angle between `n_o` and `−n` (the "palms parallel" part) |
| `finger_err` | angle between our finger axis and the reversed partner finger axis, both projected on the palm plane |

Events, checked at every 10 ms sample:

- **Contact (the palms touch):** the palm boxes are within
  `touch_dist` = 2 mm of each other (signed distance, so overlap counts),
  the face centers overlap in the palm plane (`in_plane ≤ 5 cm`; the palms
  are 10 × 12 cm), our face center is at most 5 cm behind the partner's
  face plane, and on the previous sample it was in front (`gap ≥ 0`; the
  first sample uses its own gap), so the touch came from the front, not the
  back of the partner's hand. Hovering close without touching is not a
  contact.
- **Collision:** (a) any geometry pair other than palm–palm has signed
  distance < 0 (hand–hand pairs count: fingers into a palm, a palm edge into
  the partner's wrist), or (b) the palm boxes overlap without a contact
  (edge-on, from behind, off to the side).
- **Same-sample precedence:** a contact wins over a collision first seen on
  the same sample. A fast approach can cross several millimetres in one
  10 ms step, and the finger plates (5 mm behind the face) then penetrate
  the other palm on the very sample where the palms first touch; the touch
  came first within that step. A collision on any earlier sample has
  already ended the trial. `tests/test_baselines.py` checks this in the
  real scene with a 1.5 cm step-through.

The first event ends the trial. No event within the horizon (the partner's
motion plus `t_extra` = 1 s, `dap.trial.horizon_steps`) is a **timeout**.

## 1b. Success, scores and loss

**Success = the trial ended in a contact and the palms are antiparallel:
`rot_err ≤ 15°`** (`Tolerances.rot_tol`, a placeholder, **Ours**). Position
and timing are not success criteria; they are scores:

| Score (at contact) | Meaning |
| --- | --- |
| `pos_err` | palm-center offset in the palm plane (m) |
| `rot_err`, `angle_err`, `finger_err` | alignment (deg), full and decomposed |
| `time_err` | contact time minus the partner's **true** meeting time (s) |
| `gap` | face-center normal gap at the contact sample (m) |
| `rel_speed` | palm–palm relative speed at contact (m/s) |
| `min_palm_dist`, `closest_rot_err` | closest face-center approach and its rot_err (also defined for misses) |
| `min_arm_dist`, `vmax_frac` | closest arm–arm distance; fraction of steps with a joint at ≥ 95% of its speed limit |

**Loss** (`dap_loss`, vectorized over a results table, lower is better;
weights are placeholders in `LossWeights`, **Ours**). It is tiered, so the
outcomes keep a fixed order whatever the errors: every success < every
failed contact < every miss < every collision. With
`e = (pos_err/3 cm)² + (rot_err/15°)² + (time_err/0.1 s)²` (+ an optional
relative-speed term, weight 0 for now) and `f(e) = 5e/(e+1) ∈ [0, 5)`:
- success: `f(e)`, in [0, 5);
- failed contact (rotation error over the tolerance): `5 + f(e)`, in [5, 10);
- timeout: `10 + 5 · min(min_palm_dist / 0.3 m, 1)²`, in [10, 15], so a
  near miss scores better than a far one;
- collision: 20;
- crashed trials: NaN here; the analysis gives controller crashes the
  collision loss (section 9).

Changing the weights re-scores stored results without re-running anything,
since `trials.jsonl` keeps every component.

**Nothing the controller reports is scored.** The result record
(`dap/metrics.py`) holds only harness measurements: the outcome and scores
above, scenario covariates (drawn and realized partner speed, true meeting
time, layout, reachability), and controller compute time measured by the
harness around each `step` call.

**Sanity references** (`dap/baselines.py`; 40 `nominal` seeds, **E**):
- **Oracle** (knows the scenario; joint-space min-jerk to our side of the
  certified dap, arriving at the true meeting time; its commands are clipped
  to the joint speed limits like any controller's, and its path is not
  collision-checked): 38/40 successes, 2 collisions. Both collisions are
  scenarios where it had to run at its speed limits (`vmax_frac` ≈ 0.8) and
  the clipping pushed it off its path. At contact: median `rot_err` 2.4°,
  `pos_err` 6 mm, `|time_err|` ≤ 15 ms (median 0), `rel_speed` 0.35 m/s.
- **Holding still:** 40/40 timeouts (the partner never touches a resting
  arm, by construction; section 5).

So the task is solvable but not trivial.

**Baselines** (`dap/baselines.py`). Each answers one question about the
pixel policy's result; all run through `run_trial` on the paired seeds
(`run_experiment.py --controller <name>`).

| Name | Sees the partner through | Question |
| --- | --- | --- |
| `hold` | nothing | floor |
| `blind` | nothing: a fixed palm pose | does the task need perception at all? |
| `servo` | the stand-in pose sensor (section 10) | what does a simple classical controller achieve? |
| `servo_depth` | the head camera's depth frames, the student's input | the same controller with real perception |
| `servo_gt` | true current state | the same controller with perfect state (reference) |
| `oracle` | the scenario, including the future meeting | knows the answer in advance (reference) |

- **`servo`:** reactive differential IK (resolved-rate control with damped
  least squares, damping 0.05, gain 10/s; **Ours**). Each step it aims our
  palm face at the antiparallel pose of the partner palm predicted 0.1 s
  ahead (the filter's prediction; `servo_gt` extrapolates the true state
  the same way), feeding the partner's velocity forward. It first lines up
  5 cm in front of the partner palm along its normal, then (latched once
  within 2 cm and 10°) closes along the normal. Joint velocities are scaled
  together to the limits. No collision avoidance beyond the approach along
  the normal. The stand-in sensor reports the palm's finger direction as
  well as its normal (same 2° noise), and the filter low-passes it.
- **`servo_depth`:** the same servo on the partner palm perceived from
  the frames the pixel policy sees (212×120, 30 fps, one frame late,
  `med` depth noise, the trial's camera stream; `run_trial(camera=...)`
  renders them exactly as the env does). Perception (`dap/perception.py`)
  uses no labels:
  - Segmentation: render the depth of what our robot knows (floor,
    pedestals, our own arm at its joint angles at capture) and keep pixels
    more than 3 cm + 3% of depth in front of it (**Ours**), within the
    partner arm's reach of its shoulder (1.1 m, **E** from the iiwa model).
  - Hand: points within 15 cm of the filter's predicted plate middle, or,
    before the first fix and after 0.5 s without one (a fresh filter), of
    the partner point farthest from its shoulder.
  - Plate fit: RANSAC (Fischler & Bolles 1981) with the MSAC score (Torr &
    Zisserman 2000), inlier band 2σ of the D435 axial noise (section 7), at
    least 1 cm. Normal toward the camera; finger axis = the plate's long
    axis, signed away from the partner's arm (the arm holds the 18 cm plate
    5 cm from its wrist end).
  - Accepted only if the fit spans the plate (5–95% extents within 20% of
    0.9 × 18 × 12 cm; partial views bias the plate's middle) and its
    normal is within 30° of the filter's while tracking (**Ours**), or
    within 70° of the horizontal direction to our base when acquiring (the
    facing cone plus the start tilt, sections 3–4; measured max 46° nominal,
    55° wide, **E**). While tracking, the finger sign stays continuous
    unless two fixes in a row disagree (**Ours**).
  - The fixes update the same filter as `servo` at their capture times
    (measurement std 1 cm, **Ours**; normal and finger smoothing at the
    stand-in filter's time constant). Without fixes the servo predicts at
    most 0.1 s past the last one and then holds (**Ours**): extrapolating
    the constant-acceleration filter through a blackout runs away.
  - What it may know: the camera's configuration and where the partner
    stands (as `blind`), nothing about the partner's motion.
- **`blind`:** the same servo driving to a fixed pose and waiting. The pose
  uses only what a controller may know without looking (the nominal
  meeting point and the direction of the partner's base) plus the best
  constant guess fitted on 30 training scenarios (`train_rngs`, disjoint
  from evaluation): the mean offset of our meeting palm from the nominal
  point and the chordal mean of its orientation, in the frame facing the
  partner base.
- **First look** (12 seeds per preset, **E**; a check that the baselines
  work, not a result):

  | | `nominal` | `wide` |
  | --- | --- | --- |
  | `blind` | 0/12 successes (5 contacts, all > 15° off; 5 collisions) | 0/12 (11 collisions) |
  | `servo` | 12/12 | 11/12 (1 collision) |
  | `servo_depth` | 3/12 (8 collisions, 1 timeout) | 2/12 (7 collisions, 3 timeouts) |
  | `servo_gt` | 12/12 | 12/12 |

  Meeting poses vary too much to guess: over the 30 training scenarios
  the blind prior is fitted on, our meeting palm face lies 2–14 cm from the
  nominal point, and finger directions differ by up to 179° between
  scenarios (**E**), against tolerances of 5 cm and 15°. So
  the task needs perception. But with a good pose estimate, a simple
  reactive servo nearly always daps. Two consequences, both open
  decisions (section 11):
  - Success does not require timing. The servo's contacts came 1.6 s early
    to 0.6 s late against the partner's meeting time, and about a third
    happened after the partner's palm had stopped. Timing is scored
    (`time_err`, the loss) but not required.
  - The stand-in sensor (no latency, no occlusion) makes the classical
    baseline strong. On the student's depth frames the same servo mostly
    collides hand on hand. Camera timing and visibility are not the cause:
    fed the true palm pose at the camera's capture times (one frame late,
    same 0.1 s coast), it daps 12/12 and 10/12, and 11/12 and 10/12 when
    given it only on frames where at least 15 partner points lie within
    10 cm of the palm (**E**, scratch diagnostic on the same seeds). The gap is the
    perception. On the frames of 16 `servo` trials on training scenarios,
    65% of frames give a fix; of the fixes, 84% have the palm normal within
    15° (median 3.2°), 69% the position within 5 cm (median 1.9 cm), and
    15% the finger direction flipped. In 2 of the 16 trials the hand is
    mostly hidden at the start and the tracker locks onto a plate-sized
    patch of the partner's arm (**E**).
  - Its design choices were first explored on evaluation seeds 0–15 (a
    mistake) and then re-checked on training scenarios
    (`dap.experiment.train_rngs`), where the final choices were made. The
    first-look row above uses evaluation seeds 0–11, so it is not a clean
    held-out estimate; experiment seeds beyond 15 are untouched.
- **Not built:** a plan-once variant of the servo. Freezing the current
  partner estimate would be a strawman; a fair one needs a predictor of
  where the partner's reach will end.

## 2. Scene and layout randomization (`dap/scene.py`, `dap/scenario.py`)

- **Robots:** two KUKA LBR iiwa 7 R800 (`iiwa7_with_box_collision.sdf` from
  drake_models). Drake's joint position, velocity and torque limits match the
  KUKA datasheet exactly (**V**/**E**):

  | Joint | A1 | A2 | A3 | A4 | A5 | A6 | A7 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | Range (deg) | ±170 | ±120 | ±170 | ±120 | ±170 | ±120 | ±175 |
  | Max speed (deg/s) | 98 | 98 | 100 | 130 | 140 | 180 | 180 |

  The reach is 800 mm. Do not mix in Drake's iiwa14 SDFs: they carry the
  slower iiwa 14 limits.
- **Our base and floor:** our base is at the world origin. Both bases stand
  on 0.3 m square pedestals over a floor 0.75 m below our base (a table
  height; **Ours**). The floor and pedestals give the camera a background.
  Geometry on the world body is automatically excluded from collision checks
  against the world-welded `iiwa_link_0` (**E**).
- **Hand-to-hand contact:** only the palm-box vs. palm-box pair is excluded
  from collision checks, because palm-to-palm contact is the goal; the
  scorer judges that pair (section 1a). Every other pair counts, hand–hand
  included. IK that constructs a meeting filters hand–hand pairs in its own
  context only (`Scene.new_context(filter_hands=True)`) and its result is
  re-checked in the full scene.
- **Partner base layout:** sampled per trial (`LayoutRanges`). Distance is
  along +x; yaw is added to facing us (180°).

  | Preset | Distance (m) | Lateral (m) | Yaw (deg) | Height (m) |
  | --- | --- | --- | --- | --- |
  | `fixed` | 1.4 | 0 | 0 | 0 |
  | `nominal` | U[1.2, 1.75] | U±0.3 | U±30 | U±0.15 |
  | `wide` | U[1.0, 1.9] | U±0.45 | U±45 | U±0.25 |

  Sources for these ranges:
  - Facing arms sit about 1.5 m apart in ManiSkill3 and about 1.62 m apart in
    robosuite's TwoArmHandover (**V**).
  - Habitat perturbs base yaw with a 30° standard deviation (**V**).
  - The two arms can make a palm-to-palm dap up to about 1.88 m between bases
    (**E**). The cheap analytic gate rejects anything beyond 1.85 m.
  - `nominal` passes the full IK check about 100% of the time (**E**). `wide`
    relies on rejection sampling.
- **Nominal meeting point:** the midpoint between the bases, at palm height
  0.6 m above the mean base height. Add N(0, 5 cm) per axis, clipped to
  ±10 cm (`wide`: 8 cm, clipped to ±15 cm). In human-to-robot handovers
  (20 participants, robot receiver), the giver placed the object "close to
  half" the human–robot separation (Nemlekar et al., ICRA 2019, **V**); we
  carry that over to two arms, which is our assumption. The controller
  may know this prior (`Scenario.meet_prior`); the true meeting point is
  ground truth.

## 3. "A dap is possible" check (`dap/feasibility.py`)

One `InverseKinematics` program over both arms (14 joints), solved with SNOPT:

- **Partner palm frame expressed in our palm frame:** the faces touch, the
  centers lie within ±in-plane tolerance, and the gap is at most the gap
  tolerance.
- **Relative orientation:** within the angle tolerance of `RotY(pi)`.
- **Facing direction:** our palm normal is within 45° of the direction toward
  the partner base.
- **Meeting location:** our palm face lies within ±3 cm (per axis) of the
  sampled meeting point.
- **Clearance:** `MinimumDistanceLowerBoundConstraint` of 1 cm over every
  pair except hand–hand (filtered in the IK's own context). Each solution is
  then re-checked in the full scene and rejected if any pair other than
  palm–palm penetrates.
- **Restarts:** 5 attempts per tier in total: the first starts from
  `Q_READY` for both arms, the other 4 from uniform random configurations.

There are two tiers. The first tier that succeeds sets `witness.quality`.

| Tier | In-plane | Gap | Angle | Use |
| --- | --- | --- | --- | --- |
| `tight` | ±5 mm | ≤2 mm | ≤2° | a clean dap; the partner aims for this pose |
| `loose` | ±2 cm | ≤1 cm | ≤10° | a low-quality dap that still counts as "possible" |

Both tiers are within the success tolerance (section 1b), and the partner's
path continues 3–10 cm past the meeting pose along its palm normal, so it
closes any remaining gap (≤ 1 cm in the loose tier): "feasible" means a
successful dap exists. The oracle confirms this end to end (section 1b).

- **Cost:** about 3 ms median per check, 5.5 ms at the 90th percentile; a
  certified failure costs 0.2–0.5 s (**E**). Building the scene for a layout
  costs about 56 ms, more than the solve.
- **Related approaches** (all **V**):
  - Vahrenkamp et al. (IROS 2009) solve dual-arm IK for re-grasping by
    sampling; their Table 2 reports 47–220 ms per solve, depending on the
    variant and on obstacles.
  - Kim et al. (T-RO 2014) gate catching with a learned reachable-space GMM.
  - Bäuml et al. (2010) pose catching as an NLP over the catch
    configuration, with t ≥ t_min(q).
- **Arrival in time:** a min-jerk move peaks at 1.875× its average speed
  (Flash & Hogan 1985, **V**), so a move needs
  `T ≥ max_i 1.875 |Δq_i| / v_max,i` (`min_jerk_time`; Bäuml-style). We use
  90% of the velocity limits. This only sets the `reachable_in_time`
  covariate; it never rejects a scenario.

## 4. Partner motion (`dap/partner.py`)

**Speed first.** Each scenario draws a target **peak palm speed**
`v ~ U[0.15, 0.9] m/s` (`PathConfig.peak_speed`) before anything else, and
keeps it for every retry, so the speeds that reach an experiment follow that
distribution exactly. Only the geometry is conditioned on being executable
at that speed (see "Speed and geometry" below).

**Reference path of the partner palm frame:**
- Start position: meeting point + U[0.15, 0.45] m back toward the partner
  base + U±0.25 m sideways + U±0.2 m vertically.
- Start orientation: the meeting orientation tilted by U[0, 25°] about a
  random axis.
- Geometry: a cubic Bézier to the meeting point. The first control point is
  pushed out by a random "bulge" N(0, σ²) with σ ~ U[3, 8] cm, a scaled-down
  version of GenH2R's random key point (std 0.2; Wang et al., CVPR 2024, App.
  A.1.1, **V**). The last control point makes the path **arrive moving along
  the partner's palm normal**.
- Overshoot: a straight extension of U[3, 10] cm past the contact point, so
  the hand is still moving when the palms touch (people dap *through*
  contact; **Ours**). Partner palm speed at its meeting time: median
  0.35 m/s, range 0.12–0.75 m/s over 40 `nominal` seeds (**E**).
- Timing: min-jerk along arc length (Flash & Hogan 1985, **V**). Its peak
  speed is 1.875 L/T, so the duration follows from the drawn speed and the
  path length: `T = 1.875 L / v`; `t_meet` is when the contact point is
  reached. Durations outside `t_meet ∈ [0.4, 4] s` are re-drawn (geometry
  only). Realized `t_meet` over 40 `nominal` seeds: 0.46–3.2 s, median
  0.9 s (**E**).
  - **Caveat on min-jerk.** Chan et al. (Humanoids 2020, **V**) fitted
    1195 unconstrained human handover reaches: the conic model fit best and
    plain min-jerk fit best in only one of them. We use min-jerk as a timing
    law along a curved (Bézier) path, not as a claim that human handover
    paths are min-jerk.
  - Human reference durations: handovers happen "within ∼1.2 s" in Chan et
    al.'s dataset (**V**); the giver-start to receiver-reach time was
    1.212 ± 0.051 s in Nemlekar et al. 2019 (2 participants, 30 handovers,
    **V**); reaching movements of about 0.6–2 s in the protocol of
    Controzzi et al. (Exp. Brain Res. 2018, **V**); greeting handshakes
    0.90 ± 0.26 s (Tagne et al. 2016, **S**). Our range is wider on purpose,
    to cover slow and brisk partners.
- Orientation: slerp on the same min-jerk clock.
- Optional target switch, with probability `switch_prob`: 0 in `nominal`,
  0.3 in `wide`. A second min-jerk displacement of 5–15 cm in the palm plane
  starts at 30–60% of t_meet, lasts 40–80% of t_meet and finishes before
  contact. This is the superposition model of Flash & Henis (1991, **U**).

**The partner arm executes the path** (it is a real iiwa, so it stays within
the iiwa's joint position and velocity limits everywhere):
- **Start configuration:** IK to the start pose, with a cost pulling toward
  `Q_READY + noise`. This randomizes the start joint configuration through
  the arm's spare (null-space) freedom.
- **Tracking:** IK at 20 Hz knots, within 1 mm per axis and 1°. Each knot
  minimizes its distance to the previous knot and is bounded to
  `q_prev ± 0.95 · v_max · dt`, so the elbow cannot flip branches.
- **Interpolation:** a cubic spline through the knots, rest to rest. Its
  joint velocities and positions are checked against the limits at 100 Hz
  (and tested at 500 Hz in `tests/test_scenario.py`).
- **Self-collision:** checked against the arm itself, the floor and the
  pedestals (5 mm), at every 10 ms sample of the final trajectory.
- **Ground truth:** the palm pose comes from forward kinematics of this joint
  trajectory. The camera renders this arm. After its motion ends the arm
  holds its last configuration (no jumps).

The realized peak palm speed matches the drawn one within about 5%, and is
slightly higher (ratios 0.99–1.05 over 12 seeds, **E**; tracking error and
finite differences). The analysis bins by the realized speed.

**Speed and geometry.** Above about 1 m/s the partner iiwa cannot follow
most sampled paths within its joint velocity limits: at 1.2 m/s, 7 of 8
seeds found no executable path in 200 attempts, and at 1.5 m/s none did
(**E**). Hence the 0.9 m/s cap. Up to it, sampling always succeeds, but fast
scenarios need more attempts (median attempts 1 / 1.5 / 3.5 / 7.5 / 14.5 at
0.2 / 0.6 / 0.7 / 0.8 / 0.9 m/s, 8–10 seeds each, **E**), because only some
layouts and paths let the arm move that fast. The slow end is filtered too:
at 0.15 m/s about half of the sampled paths would take longer than the 4 s
`t_meet` cap and are re-drawn, so the accepted slow paths are shorter (mean
0.38 m vs. 0.44 m, **E**). **So speed is confounded with geometry at both
ends**: fast scenarios are biased toward layouts that suit the partner arm,
slow ones toward short paths. Analyses against speed should keep this in
mind (section 10).

Rejected samples are retried and counted (`Scenario.rejections`; mostly
tracking IK, start IK and our start colliding). Sampling a `nominal`
scenario takes a median of about 0.35 s and up to about 7 s for the fastest
partners (**E**).

## 4b. Our arm's limits (`dap/trial.py`)

- **No teleporting.** The harness, not the controller, moves our arm: each
  10 ms step moves every joint at most `v_max · dt` toward the command and
  keeps it within its position limits (`limit_command`). A non-finite or
  wrongly shaped command holds the arm still and is counted
  (`n_bad_commands`). The RL env uses the same function.
  `tests/test_baselines.py` drives a controller that demands the far end of
  the joint range and sends NaNs, and checks the logged motion never exceeds
  the limits.
- **Causal loop.** At time t the controller sees q(t) and the sensor reading
  at t, and its command takes effect at t + dt.
- **Logged:** `vmax_frac`, the fraction of steps with some joint at ≥ 95% of
  its limit.
- **`reachable_in_time`** is the min-jerk check at 90% of the limits. It is
  conservative (the oracle succeeds in 15 of 17 scenarios flagged
  unreachable, **E**), so it is a covariate, not ground truth.
- **Resting-arm check.** At sampling time the partner's whole motion is
  checked against our arm held still at its start (every 10 ms sample): no
  arm or hand contact, and the palms stay at least 1 cm apart (well above
  the 2 mm touch distance). The partner never touches a resting robot, so
  doing nothing always times out.

## 5. Start configurations (`dap/scenario.py`)

Our arm starts at `Q_READY + noise`, clipped to the joint limits minus 5°.
The partner's start is randomized through its start IK, as above.

| Tier | Noise | Source |
| --- | --- | --- |
| `small` | N(0, 0.02) rad | robosuite `initialization_noise="default"` = Gaussian 0.02; ManiSkill3 `robot_init_qpos_noise` 0.02 (**V**) |
| `medium` (default) | U±0.3 rad | **Ours**, between the two others |
| `large` (`wide`) | U±0.5 rad | Isaac Lab Lift (`core/lift/lift_env_cfg.py` at commit `892354c`, Nov 2026): `reset_joints_by_offset` with `position_range` [−0.5, 0.5] rad (plus ±3 rad on the wrist), and reset states kept only with ≥ 2 cm robot–table clearance (**V**) |

A start sample is rejected when either of these holds:
- it is closer than 1 cm to anything, palm–palm included, with the partner
  at its start pose (the same keep-if-clear pattern as Isaac Lab's
  clearance criterion and RLBench's spawn checks, **V**);
- the partner's motion would touch our arm resting at that start (section
  4b).

Whether our arm can reach the dap in time is recorded, not filtered on
(`reachable_in_time`). "Our side" of the dap is the configuration closest to
the start that puts our palm at the witness pose (single-arm IK,
collision-free); the oracle moves there.

## 6. Head camera (`dap/camera.py`)

The camera is modelled on an **Intel RealSense D435**. GenH2R's real system
uses D435s (**V**), and the D435i is a common humanoid head camera (**S**).
D400 datasheet values below are for the D435 camera (Table 3-49) unless
noted.
The camera is fixed relative to our base.

| Parameter | Value | Source |
| --- | --- | --- |
| Resolution | 848×480 | Intel's recommended D435 depth resolution ("RealSense D435: 848×480", tuning whitepaper, **V**; the datasheet allows up to 1280×720); also the old Drake D415 model (Drake v1.20 `MakeD415CameraModel`, **V**) |
| Vertical FOV | 58° (gives 88.8° horizontal, fx = fy = 433 px) | D435 depth FOV H 87° / V 58° (datasheet Table 3-49, at HD 16:9, **V**). Drake's default of 45° leaves almost no margin around the workspace |
| Rate | 30 fps | D435, Azure Kinect, DexYCB (**V**). Use 15 fps as a stress case (DROID records at 15 Hz, **V**) |
| Depth range | 0.195–3.0 m | D435 Min-Z at 848×480 is 195 mm (datasheet Table 4-12, **V**); ideal range 0.3–3 m (**S**) |
| Clipping | 0.01–10 m | Drake default near (**V**); `manipulation` repo far (**V**) |
| Mount | (−0.4, 0, 1.45) m in our base frame, looking at (0.7, 0, 0.6), so pitched about 38° down | RoboCasa cameras sit 0.5–0.6 m behind the base, 0.95–1.15 m up, pitched 26–35° down (**V**, recomputed from its camera quaternions). Fetch's head camera is 1.06–1.45 m above its base link, torso down to up (**E**, forward kinematics of the Fetch URDF). For scale, overall robot heights: HSR 1.005–1.35 m (**V**), TIAGo about 1.1–1.45 m (**S**). Those are robot heights, not camera heights |
| Latency | 1 frame (33 ms) | **Ours**. Extreme Parkour uses a fixed 80 ms (**V**); Robot Parkour uses U[0.2, 0.26] s (**V**). HandoverSim notes that its timings leave out perception latency (**V**) |
| Segmentation | none for the pixel policy; `servo_depth` segments geometrically (`docs/perception_pipeline.md`) | The Drake label image (label = body index) is ground truth, used only by tests |

**Other benchmarks, for comparison** (all **V**):

| Benchmark | Resolution | FOV | Notes |
| --- | --- | --- | --- |
| HandoverSim (third-person) | 1600×900 | 60° vertical | offscreen visualization camera, not a policy input |
| GenH2R (third-person) | 1280×720 | 60° | offscreen visualization camera |
| HandoverSim / GenH2R wrist cam | 224×224 | 90° | |
| ManiSkill3 | 128×128 | 90° | 20 Hz control |
| robosuite | 256×256 | 45° | 20 Hz |
| RoboCasa | 128–256 px | 60° | |
| Drake `CameraConfig` default | 640×480 | 45° | 10 Hz |

**How our setup differs from the references.** The two catching papers
track the object with marker-based motion capture:
- Kim et al. 2014: OptiTrack at 240 Hz (**V**).
- Salehian et al. 2016: OptiTrack at 240 Hz (**V**).

(Fitter & Kuchenbecker 2016 is a hand-clapping study; we have not read its
sensing setup in the full text, so we make no claim about it.) A single
markerless RGB-D head camera at 30 Hz is a much harder perception setting.

**Measured:**
- Rendering an 848×480 depth + label frame takes about 80 ms. At the 0.25×
  scale (212×120) used by the RL env it is much cheaper.
- At the nominal layout the partner palm covers about 950 px at about 1.5 m.
- **From a centered head camera, our own forearm occludes the middle of the
  view.** This is realistic (see the perception doc), and the camera position
  is a knob.

## 7. Depth noise (`dap/depth_noise.py`)

The model is D435-like (stereo baseline B = 50 mm, **V**). Drake renders
perfect depth; `DepthNoiseModel` applies these steps in order:

| # | Step | Model | Source |
| --- | --- | --- | --- |
| 1 | Range gate | z < 0.195 m or z > 3 m → 0 (invalid) | D400 datasheet (**V**) |
| 2 | Lateral jitter | shift by N(0, σ_s²) px, bilinear resample; mixed/flying pixels at edges | Handa et al. 2014 (ICL-NUIM), σ_s = 0.5 px (**V**); same order as Nguyen et al. 2012, σ_L ≈ 0.8 px (**V**). **Both are Kinect v1 (structured light) models**, not D435 stereo; we borrow the order of magnitude |
| 3 | Axial noise | Gaussian noise in disparity d = f·B/z, σ = 0.08 px. This gives RMS_z = z²·σ/(f·B): 0.9 / 3.7 / 8.3 mm at 0.5 / 1 / 1.5 m (our f = 433 px) | Intel "Tuning depth cameras for best performance" whitepaper (**V**): its D435 RMS-error curves use subpixel 0.08; a well-calibrated camera should show < 0.1 (approaching 0.05 on well-textured targets); recalibrate above about 0.2. The RMS formula is the standard stereo relation used there. Baseline B = 50 mm: datasheet Table 3-49 (**V**) |
| 4 | Fixed-pattern noise | coarse (8× downsampled) Gaussian disparity field, bilinear-upsampled, **held for the episode** | Inspired by Dex-Net 2.0's Gaussian-process noise (Mahler et al. 2017, Sec. IV and App. A, **V**), which is in depth (m) and redrawn per image. Holding it for the episode and putting it in disparity are **Ours**, motivated by the datasheet's temporal noise ≤ 1% vs. spatial noise ≤ 2% (Table 4-15, at HD, ≤ 2 m, 80% ROI, **V**) |
| 5 | Scale bias | depth × α, α ~ N(1, 0.5%) per episode | D400 Z-accuracy ±2% (Table 4-15, same conditions, **V**); Dex-Net's gamma (shape 1000, scale 0.001: SD about 3%) is the harsh end (**V**). 0.5% is **Ours** |
| 6 | Grazing-angle dropout | invalid where the surface-to-ray angle exceeds 75° (normals from the depth image) | Nguyen et al. 2012: invalid pixels rise sharply at extreme angles (**V**); threshold is **Ours** |
| 7 | Stereo shadow | an invalid band f·B·(1/z_fg − 1/z_bg) px wide to the **left** of foreground edges | The left imager is the depth reference (datasheet §4.5, which gives the related invalid band B·F/Z at the left frame edge, **V**). The per-edge occlusion formula is our derivation from the same geometry (**Ours**); about 18 px for a palm at 0.6 m against 1.2 m |
| 8 | Holes | Bernoulli(1%) pixels + 2 rectangular blobs of 4–16 px | D400 fill rate ≥ 99% (Table 4-15, **V**). Blobs are inspired by the block artifacts in Robot Parkour's released code (`stereo_full_block_artifacts_prob` 0.008, blocks about 62×3 px at 424×240; `a1_field_distill_config.py`, ZiwenZhuang/parkour @ `5e1f413`, **V**); our sizes are **Ours** |
| 9 | Quantize | 1 mm | D4 default depth unit, 1000 µm (tuning whitepaper, **V**) |

Latency is handled by `camera.FrameBuffer`, not here.

**Resolution.** All pixel-valued parameters are in native 848-px sensor
pixels: disparity σ, jitter σ_s, the pattern grid and blob sizes.
`DepthNoiseModel.for_camera(cfg, preset)` scales them to the rendered
resolution, so the metric and angular noise stay the same at any resolution.
Disparity σ scales too, because f scales with resolution. Two more details:

- Grazing dropout skips depth discontinuities. Finite-difference normals are
  meaningless on silhouettes, which are handled by the jitter and shadow
  steps.
- The jitter resampling interpolates over valid neighbours only, so it does
  not grow holes.

Measured on the partner palm, the valid fraction is 0.96–0.99 at 848, 424
and 212 px wide (**E**).

**Presets for ablations (`NOISE_PRESETS`):**

| Knob | `low` | `med` (default) | `high` |
| --- | --- | --- | --- |
| σ disparity (px) | 0.05 | 0.08 | 0.2 |
| σ_s (px) | 0.25 | 0.5 | 1.0 |
| Fixed-pattern σ (px) | 0 | 0.04 | 0.2 |
| Scale σ | 0 | 0.5% | 2% |
| Grazing θ_max | off | 75° | 65° |
| Holes | 0.5% | 1% + 2 blobs | 5% + 5 blobs |

`none` turns noise off. Where the values come from:
- σ disparity: Intel's typical value and recalibration bound.
- σ_s: Handa's value is `med`.
- Scale σ: the 2% cap is the datasheet bound.
- The rest are **Ours**.

**Caveats.**
- This is a parametric surrogate. It misses material-dependent failures
  (specular, dark, transparent), matching errors on textureless or repetitive
  surfaces, multipath, motion blur and rolling shutter.
- Physics-based IR-stereo simulation (Zhang et al., T-RO 2023, used in
  ManiSkill; DREDS, ECCV 2022) closes more of the sim-to-real gap. Their
  gains are mostly on non-Lambertian objects, and our palms are matte boxes.
- Per-frame i.i.d. noise alone would make a Kalman filter look
  optimistically good. That is why steps 4–5 persist across frames.
- **Unverified:** D435 hardware latency; D4 subpixel bit depth.

## 8. RL environment (`dap/envs/`, `dap/bank.py`, `dap/policy_eval.py`)

`gymnasium.make("dap/PalmMeet-v0", obs_mode="pixels" | "privileged" | "state" | "state+depth", bank=..., ...)`

The project's method is an end-to-end policy acting from head-camera depth.
The env is built so that such a policy can only see what the robot itself
could: its own arm state and depth frames. Ground truth exists for training
(a privileged teacher or critic) and for scoring, never as the deployed
policy's input.

- **Base class:** a plain `gymnasium.Env`, not `pydrake.gym.DrakeGymEnv`.
  - `DrakeGymEnv` wraps a Simulator-driven Diagram with an action input
    port, but our loop is kinematic.
  - In Drake 1.57 and 1.58 it also ignores `info_handler` (fixed upstream in
    PR #25228) and rejects reset `options` (**V**).
  - Revisit it if the arm becomes dynamic.
- **Rates:** policy at 20 Hz over the 100 Hz kinematic loop. robosuite,
  ManiSkill3 and LIBERO all control at 20 Hz (**V**); the `manipulation`
  BoxFlipUp env runs at 10 Hz (**V**).
- **Action:** `Box(-1, 1, (7,))`: joint velocity as a fraction of each
  joint's limit (1 means full speed). There is nothing else to actuate: the
  hands are rigid. The env moves the arm through the trial loop's
  `limit_command` (no teleporting); non-finite actions count as 0 and are
  counted in `info["n_bad_actions"]`. Symmetric [-1, 1] actions are what
  Gymnasium's checker expects (**V**).
- **Observation modes** (all `Dict`s, float32, finite bounds):

  | `obs_mode` | Keys | Partner information | Use |
  | --- | --- | --- | --- |
  | `pixels` | `proprio` (31), `depth` (3, H, W) | only what the camera sees | the end-to-end policy (student) |
  | `privileged` | `state` (43) | true palm pose and velocity, as of the newest delivered frame | teacher or critic, training only |
  | `state` | `state` (37) | stand-in sensor's Kalman estimate | modular variant |
  | `state+depth` | `state` (37), `depth` (1, H, W) | both of the above | legacy |

  - `proprio` (`PROPRIO_LAYOUT`): q, q̇, our palm position, normal and
    finger direction (forward kinematics of q), t, and the previous action.
    It contains nothing about the partner (tested).
  - `privileged` (`PRIVILEGED_LAYOUT`) = `proprio` + the partner palm face's
    true position, velocity, normal and finger direction. The same vector
    is in every step's `info["privileged"]`, so distillation can label
    pixel-mode rollouts with the teacher's input.
  - **The partner part is latency-matched** (`privileged_delay=True`, the
    default): it is the true state at the capture time of the newest depth
    frame the `pixels` policy has received (`info["privileged_t"]`; before
    the first delivery, the capture at t = 0). So it lags the present by
    33–77 ms, like the student's newest frame (tested), and the privileged
    env follows the camera's timing without rendering (its observation
    equals the pixel env's `info["privileged"]`, tested).
    `privileged_delay=False` gives the current state (e.g. for a critic).
    **Ours**: a teacher acting on information the student cannot infer is
    imitated only on average (the "imitation gap", Weihs et al. 2021). The
    nearest precedent: Agarwal et al. (CoRL 2022) "simulate a latency of
    40ms in both the phases of training" for their teacher–student "Noisy"
    elevation-map baseline (Sec. 3, **V**); their main scandot teacher
    states no latency. `info["partner_p"]`, `partner_v` and `partner_n`
    stay current: they are ground truth for rewards, not an observation.
  - `depth`: metres, 0 = no return, from the head camera at `camera_scale`
    (default 0.25, giving 212×120) with `depth_noise` (default `med`) and
    one frame of latency. In `pixels` mode it is a stack of the
    `frame_stack` most recently delivered frames, oldest first, zeros before
    the first delivery. Frames are distinct 30 Hz captures, so the stack
    spans 67 ms at the default of 3. **Ours**: one frame has no velocity,
    and stacking recent frames is standard in pixel RL; DrQ-v2's code
    release stacks 3 (`frame_stack: 3`, facebookresearch/drqv2
    `cfgs/config.yaml`, **V**).
  - Float metres follows Isaac Lab; ManiSkill uses uint16 millimetres
    (**V**). Stable-Baselines3 only treats uint8 images as images, so a
    float depth observation needs `normalize_images=False` and a custom CNN
    feature extractor (**V**).
- **Episode end:** the scorer's event rule (`dap.scoring.EventMonitor`,
  section 1a) and the trial's horizon (`dap.trial.horizon_steps`), so an
  episode ends where the same motion would end a trial (tested).
  - `terminated` at the first contact or collision.
  - `truncated` at the horizon (1 s after the partner's motion ends).
- **Where scenarios come from:**
  - **Training: a scenario bank** (`dap/bank.py`,
    `scripts/build_bank.py`). Sampling a scenario is hundreds of IK solves
    (about 1 s on average with the camera, several seconds for fast
    partners), which made resets most of an episode's cost. A bank stores
    each sampled scenario's layout, path parameters and IK knots; a reset
    rebuilds the scene and the partner's spline from them in tens of ms.
    Rebuilt scenarios are identical to the sampled ones (tested).
  - **Training seeds never overlap evaluation seeds.** Banks and fresh
    env samples come from `dap.experiment.train_rngs` (entropy
    `[seed, k, TRAIN_TAG]`); experiments use `trial_rngs` (`[seed, k]`).
    Sampling from `np_random` directly would not do: numpy drops trailing
    zero words, so `default_rng([s, 0])` is `default_rng(s)`, the stream
    `reset(seed=s)` gets (tested).
  - **Evaluation:** `reset(options={"trial_seed": s})` samples experiment
    seed `s` exactly as `run_trial` does, with its stand-in-sensor and
    depth-noise streams, bypassing any bank.
  - Without a bank or a trial seed, a reset samples a fresh scenario from
    a training stream seeded by `self.np_random`.
- **Scoring a policy** (`dap/policy_eval.py`): `run_policy_trial(cfg,
  cond, seed, policy)` runs the policy in an env with the experiment's
  timing, tolerances and preset (checked), records the trial's ground-truth
  log, and scores it with `dap.scoring.evaluate` and the same live/offline
  agreement check as `run_trial`. Rows have the same fields and covariates
  as the planners', so `scripts/analyze.py` compares them on paired seeds.
  A constant-velocity motion scored this way gives the same row as through
  `run_trial` (tested). A policy crash is a trial outcome, as for
  controllers. Privileged policies are refused unless explicitly allowed (a
  reference, not a method). From the command line:
  `scripts/run_experiment.py <config> --policy module:function --policy-name <label>`.
- **Reward:** `reward_fn(info) -> float`, which returns 0 when not given. No
  reward shaping yet. `info` carries ground truth: the event and success,
  the section 1a geometry (`normal_gap`, `in_plane_offset`, `rot_err_deg`,
  `angle_err_deg`, `finger_err_deg`), face-center distance, clearance,
  palm–palm distance, arm distance, partner pose and velocity (current), the
  privileged vector and the time its partner part is from, `t_meet`, the
  partner's peak speed and the scenario id. Rewards are for learning only; results are always the harness scores.
- **Determinism:** the scenario, sensor noise and depth noise come from
  `self.np_random`, or from the trial seed's streams when one is given.
  `gymnasium.utils.env_checker.check_env` passes with no warnings in every
  mode (**E**).
- **Cost** (**E**, M3 Pro, one process, 10 episodes holding still, bank of
  40):

  | Mode | Reset | Policy step | Policy steps/s per core |
  | --- | --- | --- | --- |
  | `pixels` (212×120) | ≈130 ms | ≈8 ms | ≈95 |
  | `privileged` | ≈40 ms | ≈1 ms | ≈620 |

  Without a bank, resets sample fresh scenarios: about 1.1 s on average
  with the camera, up to several seconds. Building a bank takes about
  0.25 s per scenario per worker (40 scenarios in 10 s with 8 workers).
- **Training dependencies are not installed.** For later: stable-baselines3
  2.9.2 needs `gymnasium>=0.29.1,<2` and `torch>=2.8`; Drake's and the
  `manipulation` book's examples use SB3 PPO with `SubprocVecEnv`.

## 9. Experiments and statistics (`dap/experiment.py`, `dap/stats.py`, `scripts/analyze.py`)

**Design.** A condition is (planner, sensor-noise scale). Every condition
runs on the same seeds, and a seed fixes the whole scenario (layout, partner
path and speed, our start) and the sensor-noise stream. Comparisons between
conditions are therefore **paired**, seed by seed. Partner speed is not a
condition: it is drawn per scenario, so every run covers the whole speed
range, and speed effects are analysed against the realized partner peak
speed.

| Config | Question | Conditions | Seeds |
| --- | --- | --- | --- |
| `smoke` | plumbing | replan | 3 |
| `replan_vs_once` | H1: does replanning beat planning once? | {once, replan} × noise {0.5, 1, 2} | 150 |
| `speed_limits` | H2: how does success fall with partner speed? | replan | 300 |

`run_experiment.py --controller hold|blind|servo|servo_depth|servo_gt|oracle` runs the
baselines and references (section 1b) on the same seeds.

**Analysis** (`scripts/analyze.py`):
- **Success rates** with 95% Wilson score intervals (Wilson 1927;
  recommended over the Wald interval by Brown, Cai & DasGupta 2001), from
  pooled counts per condition. Contact, collision and timeout rates the
  same way.
- **Paired success** between two planners at each noise level: the exact
  McNemar test, a two-sided binomial test on the discordant seeds (McNemar
  1947).
- **Paired loss:** mean per-seed difference with a percentile bootstrap CI
  over seeds (Efron & Tibshirani 1993). The contact scores `rot_err`,
  `pos_err` and `|time_err|` are compared the same way, but only on seeds
  where both planners made contact (`n_both_contact` is reported).
- **Multiple comparisons:** Holm's step-down correction across the tests in
  one analysis (Holm 1979).
- **Speed:** success with Wilson intervals, and mean loss, per realized
  peak-speed bin (default 0.15 m/s wide; the outer bins are widened so every
  trial lands in one).
- **Crashes** are never silently dropped. Each crashed trial records who
  crashed (`error_kind`), and the analysis treats them differently:
  - `controller`: an exception inside the controller is a failure of the
    method. It stays in every statistic as a trial with success = False and
    the collision loss (the trial loop records it with all its scenario
    covariates).
  - `scenario`: no scenario could be sampled for that seed. This depends
    only on the seed, so the seed is dropped from every condition, and
    counted.
  - `harness`: a bug in our code. Excluded, counted, and flagged first in
    the output; such results should not be trusted.
  Crash counts per condition are in `summary.csv`, and per bin of the drawn
  speed in `by_speed.csv`. A resumed sweep retries crashed trials, and the
  last row per trial counts.

**Sample sizes.** With 150 paired seeds, McNemar has about 80% power
(α = 0.05) to detect a 10-point difference in success rate when about 20% of
seeds are discordant (Connor 1987 sample-size formula: n ≈ 155). With 300
seeds over five 0.15 m/s bins (about 60 per bin), each bin's 95% Wilson
interval has a half-width of at most about 0.12. These are planning numbers; the
discordance rate is unknown until the planner exists.

## 10. Methodology notes

- **Filter model mismatch.** The estimator (`dap/estimator.py`) is a
  constant-acceleration Kalman filter; the partner follows a min-jerk
  profile, whose acceleration changes sign mid-reach. The filter therefore
  lags a decelerating partner, and its velocity estimate is only roughly
  right mid-approach (`tests/test_estimator.py` tolerates 0.15 m/s for this
  reason). Real human reaches are not min-jerk either (Chan et al. 2020), so
  any fixed motion model is misspecified; we accept this for now and treat
  the choice of motion model as an experimental variable for later.
- **The stand-in sensor is idealized, and the pixel policy does not use
  it.** `NoisyPoseSensor` gives ground-truth palm position, normal and
  finger direction plus i.i.d. Gaussian noise (5 mm, 2°), at 100 Hz, with no
  latency, no dropouts
  and no occlusion. It feeds only the `state` modes, the planner
  interface and `servo` (`servo_depth` perceives from depth instead). The
  end-to-end policy (`pixels` mode, section 8) sees only its
  own arm state and the head camera's depth frames (30 fps, one frame of
  latency, D435-style noise; sections 6–7), so the camera's limits,
  including our own hand occluding the partner near contact, are part of
  what it has to learn. Results from the `state` modes are optimistic about
  perception.
- **Privileged information is for training only.** The `privileged` mode
  and `info["privileged"]` carry ground truth for a teacher or critic. A
  privileged policy is a reference, like the oracle; `dap.policy_eval`
  refuses to score one unless asked to. Its partner state is as old as the
  student's newest frame (section 8), so teacher vs. student measures the
  cost of perceiving from depth, not of latency. `servo_gt` and the
  stand-in sensor have no latency, so they are not latency-matched to the
  teacher.
- **Speed is confounded with geometry at both ends** (section 4). Success
  vs. speed is observational; if needed, adjust for layout covariates
  (`layout_*`, `reachable_in_time`, all stored per trial), e.g. by logistic
  regression.
- **Ground truth only.** Scores come from forward kinematics and Drake
  signed distances in the harness's own context. The controller plans in a
  separate context (`dap.planner.ControllerModel`) and is never asked what
  happened.
- **The oracle is a reference, not an upper bound.** It knows the scenario
  but does not avoid collisions and does not plan around its speed limits.

## 11. Open decisions

- **Does a dap need timing?** Success is "the palms touch antiparallel".
  The reactive servo meets that with contacts from 1.6 s early to 0.6 s
  late, about a third after the partner's palm has stopped (section 1b).
  Options: keep timing as a score only, or require contact within a window
  around the partner's meeting time or while the partner is still moving.
- **What makes the task hard for a good controller?** With the stand-in
  sensor, the servo nearly always succeeds on both presets; with
  perception from the student's depth frames it rarely does (section 1b).
  Other candidates: faster partners, mid-course switches, a timing
  requirement.

- **Loss weights and the success tolerance.** `rot_tol` (15°) and the loss
  scales and weights (`LossWeights`) are placeholders. Because every
  component is stored per trial, settling them later only re-scores.
- **Camera position.** Centered behind our base gives realistic
  self-occlusion; an offset of about 0.2–0.3 m sideways would reduce it.
  Should camera position be an experimental axis?
- **Segmentation.** `servo_depth` segments geometrically (no labels); the
  pixel policy gets no segmentation at all.
- **Meeting type.** Only the opposed-parallel dap so far. The facing cone
  (45°) and the free rotation about the palm normal are the knobs for adding
  more.
- **Partner speed range.** 0.15–0.9 m/s, capped by what the partner iiwa
  can execute. Faster "human" partners would need a faster partner arm
  model (e.g. scaled velocity limits for the partner only), which would no
  longer be a real iiwa.

## References

Bibliographic details below were checked against Crossref or the paper
itself, except where marked.

**Related work**
- Kim, Shukla, Billard. Catching objects in flight. IEEE T-RO 30(5):1049–1065, Oct. 2014.
- Salehian, Khoramshahi, Billard. A dynamical system approach for softly catching a flying object: theory and experiment. IEEE T-RO 32(2):462–471, Apr. 2016.
- Fitter, Kuchenbecker. Equipping the Baxter robot with human-inspired hand-clapping skills. Proc. IEEE RO-MAN 2016, pp. 105–112.
- Vahrenkamp, Berenson, Asfour, Kuffner, Dillmann. Humanoid motion planning for dual-arm manipulation and re-grasping tasks. IROS 2009, pp. 2464–2470.
- Bäuml et al. Kinematically optimal catching a flying ball with a hand-arm-system. IROS 2010.

**Human motion**
- Flash, Hogan. The coordination of arm movements: an experimentally confirmed mathematical model. J. Neurosci. 5(7), 1985.
- Flash, Henis. Arm trajectory modifications during reaching towards visual targets. J. Cogn. Neurosci., 1991 (abstract only).
- Chan, Tran, Sheikholeslami, Croft. An experimental validation and comparison of reaching motion models for unconstrained handovers: towards generating humanlike motions for human-robot handovers. IEEE-RAS Humanoids 2020 (published 2021), pp. 356–361; arXiv:2108.12780.
- Nemlekar, Dutia, Li. Object transfer point estimation for fluent human-robot handovers. ICRA 2019.
- Controzzi et al. Humans adjust their grip force when passing an object according to the observed speed of the partner's reaching out movement. Exp. Brain Res. 236:3363–3377, 2018.
- Tagne, Hénaff, Gregori. Measurement and analysis of physical parameters of the handshake between two persons according to simple social contexts. IROS 2016, pp. 674–679 (value cited from a secondary source).

**Handover benchmarks**
- Chao et al. HandoverSim. ICRA 2022, arXiv:2205.09747.
- Wang et al. GenH2R. CVPR 2024, arXiv:2401.00929.
- Christen et al. Learning human-to-robot handovers from point clouds. CVPR 2023, arXiv:2303.17592.

**Perception**
- Fischler, Bolles. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Commun. ACM 24(6):381–395, 1981.
- Torr, Zisserman. MLESAC: a new robust estimator with application to estimating image geometry. Comput. Vis. Image Underst. 78(1):138–156, 2000.

**Depth sensing and noise**
- Intel RealSense D400 Series Datasheet (Mar 2026); Intel, "Tuning depth cameras for best performance" (web whitepaper, accessed Nov 2026).
- Nguyen, Izadi, Lovell. Modeling Kinect sensor noise for improved 3D reconstruction and tracking. 3DIMPVT 2012.
- Khoshelham, Elberink. Accuracy and resolution of Kinect depth data. Sensors 12(2), 2012.
- Handa et al. A benchmark for RGB-D visual odometry, 3D reconstruction and SLAM (ICL-NUIM). ICRA 2014.
- Mahler et al. Dex-Net 2.0. RSS 2017, arXiv:1703.09312.
- Zhang et al. Close the optical sensing domain gap by physics-grounded active stereo sensor simulation. T-RO 2023, arXiv:2201.11924.
- Dai et al. DREDS. ECCV 2022, arXiv:2208.03792.
- Zhuang et al. Robot Parkour Learning. CoRL 2023; code: github.com/ZiwenZhuang/parkour @ 5e1f413 (`legged_gym/legged_gym/envs/a1/a1_field_distill_config.py`).
- Cheng et al. Extreme Parkour with legged robots. arXiv:2309.14341.

**Pixel RL**
- Agarwal, Kumar, Malik, Pathak. Legged locomotion in challenging terrains using egocentric vision. CoRL 2022, arXiv:2211.07638.
- Weihs, Jain, Liu, Salvador, Lazebnik, Kembhavi, Schwing. Bridging the imitation gap by adaptive insubordination (ADVISOR). NeurIPS 2021, arXiv:2007.12173.
- Yarats, Fergus, Lazaric, Pinto. Mastering visual continuous control: improved data-augmented reinforcement learning (DrQ-v2). arXiv:2107.09645, 2021; code: github.com/facebookresearch/drqv2 (`cfgs/config.yaml`: `frame_stack: 3`).

**Simulation benchmarks:** ManiSkill3, robosuite, RoboCasa, LIBERO, RLBench, DROID (arXiv:2403.12945), DexYCB (arXiv:2104.04631), Habitat, Isaac Lab (github.com/isaac-sim/IsaacLab @ 892354c, `source/isaaclab_tasks/isaaclab_tasks/core/lift/lift_env_cfg.py`).

**Statistics**
- Wilson. Probable inference, the law of succession, and statistical inference. JASA 22(158):209–212, 1927.
- Brown, Cai, DasGupta. Interval estimation for a binomial proportion. Statistical Science 16(2):101–133, 2001.
- McNemar. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika 12(2):153–157, 1947.
- Connor. Sample size for testing differences in proportions for the paired-sample design. Biometrics 43(1):207–211, 1987.
- Holm. A simple sequentially rejective multiple test procedure. Scand. J. Statist. 6(2):65–70, 1979 (no DOI; details not re-checked).
- Efron, Tibshirani. An Introduction to the Bootstrap. Chapman & Hall, 1993.
