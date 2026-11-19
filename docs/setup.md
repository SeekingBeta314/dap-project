# Problem setup

This document records the simulated problem: scene, randomization, the
"a dap is possible" check, the partner's motion, the head camera, depth
noise, and the RL environment. It gives every parameter and where it comes
from. Nothing here is final.

Status labels on sources: **V** means verified in the primary paper, datasheet
or released code. **S** means secondary (a review, or a paper citing the
original). **E** means measured in this repo (Drake 1.58, SNOPT, Apple
silicon). **Ours** means our own engineering choice.

Code map:

| Topic | Module |
| --- | --- |
| Scene, layout, collision queries | `dap/scene.py` |
| Head camera, frame timing | `dap/camera.py` |
| Depth noise | `dap/depth_noise.py` |
| Feasibility check | `dap/feasibility.py` |
| Partner motion | `dap/partner.py` |
| Scenario sampler | `dap/scenario.py` |
| Gymnasium env | `dap/envs/` |

---

## 1. Task and meeting type

Our iiwa must meet the partner iiwa's hand palm to palm, then clasp. For now
the meeting type is fixed. **The palms are parallel and face each other, and
the fingers point in opposite directions.** In palm frames (+z is the palm
normal, +x points toward the finger hinge) the partner palm frame is our palm
frame rotated by `RotY(pi)`. That gives `x_partner = -x_ours`,
`z_partner = -z_ours` and `y_partner = y_ours`. The palm faces touch, with
their centers aligned.

## 1a. Contact (`dap/metrics.py: palm_contact`)

Hand-hand collisions are filtered (section 2), so palm contact is decided
geometrically from the two face centers and the partner palm normal `n`,
which points toward us. With `offset = p_ours - p_partner`:

- normal gap `g = offset · n`: our face center's signed distance in front of
  the partner's face plane;
- in-plane offset `r = |offset - g n|`.

**Contact** holds when `-5 cm ≤ g ≤ contact_dist` (1 cm) and `r ≤ 5 cm`
(the faces overlap: the palm is 10 × 12 cm). A negative gap (passed through)
counts as contact, so a fast approach cannot step over the 1 cm band between
10 ms samples. `pos_err = r` at the first contact sample is then a real
check against `success_pos_tol` (3 cm). Before, contact required the face
*centers* to be within 1 cm, which made the 3 cm tolerance unreachable.

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
- **Hand-to-hand contact:** collisions between the two hands are filtered out,
  because palm-to-palm contact is the goal. Every other pair counts as a
  collision.
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
  ±10 cm (`wide`: 8 cm, clipped to ±15 cm). People hand over at "close to
  half" their separation (Nemlekar et al., ICRA 2019, **V**). The controller
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
  unfiltered pair.
- **Fingers:** open (hinges at 0).
- **Restarts:** 5 attempts per tier in total: the first starts from
  `Q_READY` for both arms, the other 4 from uniform random configurations.

There are two tiers. The first tier that succeeds sets `witness.quality`.

| Tier | In-plane | Gap | Angle | Use |
| --- | --- | --- | --- | --- |
| `tight` | ±5 mm | ≤2 mm | ≤2° | a clean dap; the partner aims for this pose |
| `loose` | ±2 cm | ≤1 cm | ≤10° | a low-quality dap that still counts as "possible" |

Both tiers are within the contact definition (section 1a) and the success
tolerances in `ExperimentConfig` (3 cm, 15°), so "feasible" means a scorable
dap exists. After the partner tracks its path and our arm solves for the
meeting (both within 1 mm per axis), the re-created meeting has a face gap of
a few millimetres. That is below the 1 cm contact distance, though not zero.

- **Cost:** about 3 ms median per check, 5.5 ms at the 90th percentile; a
  certified failure costs 0.2–0.5 s (**E**). Building the scene for a layout
  costs about 56 ms, more than the solve.
- **Related approaches** (all **V**):
  - Vahrenkamp et al. (IROS 2009) solve dual-arm hand-off by sampling joint
    IK, in 47–161 ms.
  - Kim et al. (T-RO 2014) gate catching with a learned reachable-space GMM.
  - Bäuml et al. (2010) pose catching as an NLP over the catch
    configuration, with t ≥ t_min(q).
- **Arrival in time:** a min-jerk move peaks at 1.875× its average speed
  (Flash & Hogan 1985), so a move needs `T ≥ max_i 1.875 |Δq_i| / v_max,i`
  (`min_jerk_time`; Bäuml-style). We use 90% of the velocity limits.

## 4. Partner motion (`dap/partner.py`)

**Reference path of the partner palm frame:**
- Start position: meeting point + U[0.1, 0.3] m back toward the partner base
  + U±0.25 m sideways + U±0.2 m vertically.
- Start orientation: the meeting orientation tilted by U[0, 25°] about a
  random axis.
- Geometry: a cubic Bézier to the meeting point. The first control point is
  pushed out by a random "bulge" N(0, σ²) with σ ~ U[3, 8] cm, a scaled-down
  version of GenH2R's random key point (std 0.2; Wang et al., CVPR 2024, App.
  A.1.1, **V**). The last control point makes the path **arrive moving along
  the partner's palm normal**.
- Overshoot: a straight extension of U[3, 10] cm past the contact point.
  People dap *through* contact, so the hand is still moving when the palms
  touch. The median contact speed is about 0.2 m/s and the maximum about
  0.55 m/s (**E**). This keeps velocity matching (H3) meaningful. The old
  spline stopped exactly at the meeting point. (**Ours**)
- Timing: min-jerk along arc length (Flash & Hogan 1985, **S**; used for
  handover reaches by Chan et al. 2021, **V**). It is scaled so the contact
  point is reached at `t_meet ~ U[0.8, 2.0] s / speed_scale`. Human values:
  - handovers take about 1.2 s (Chan et al. 2021, **V**; Nemlekar et al. 2019:
    1.21 ± 0.05 s, **V**);
  - reaches take 0.6–2 s (Controzzi et al. 2018, **V**);
  - greeting handshakes take 0.90 ± 0.26 s (Tagne et al. 2016, **S**).
- Speed cap: peak palm speed ≤ 1.0 m/s × `speed_scale`. Min-jerk with
  D = 0.5 m and T = 0.8–1.2 s gives 0.8–1.2 m/s. **No primary source was
  found for "about 1 m/s".**
- Orientation: slerp on the same min-jerk clock.
- Optional target switch, with probability `switch_prob`: 0 in `nominal`,
  0.3 in `wide`. A second min-jerk displacement of 5–15 cm in the palm plane
  starts at 30–60% of t_meet, lasts 0.4–0.8 s and finishes before contact.
  This is the superposition model of Flash & Henis (1991, **U**: abstract
  only).

**The partner arm executes the path:**
- **Start configuration:** IK to the start pose, with a cost pulling toward
  `Q_READY + noise`. This randomizes the start joint configuration through
  the arm's spare (null-space) freedom.
- **Tracking:** IK at 20 Hz knots, within 2 mm and 1°. Each knot minimizes
  its distance to the previous knot and is bounded to
  `q_prev ± 0.95 · v_max · dt`, so the elbow cannot flip branches.
- **Interpolation:** a cubic spline through the knots. Its velocity is
  checked against the limits at 100 Hz.
- **Self-collision:** checked against the arm itself, the floor and the
  pedestals (5 mm), at every 10 ms sample of the final trajectory, not just
  at the IK knots.
- **Ground truth:** the palm pose comes from forward kinematics of this joint
  trajectory. The camera renders this arm.

Rejected samples are retried and counted (`Scenario.rejections`). Typical
reasons are tracking IK and start IK. Each scenario takes about 0.3 s to
sample (`nominal`), with 2–3 attempts on average (**E**).

## 5. Start configurations (`dap/scenario.py`)

Our arm starts at `Q_READY + noise`, clipped to the joint limits minus 5°.
The partner's start is randomized through its start IK, as above.

| Tier | Noise | Source |
| --- | --- | --- |
| `small` | N(0, 0.02) rad | robosuite `initialization_noise` 0.02; ManiSkill3 `robot_init_qpos_noise` 0.02 (**V**) |
| `medium` (default) | U±0.3 rad | **Ours**, between the two others |
| `large` (`wide`) | U±0.5 rad | Isaac Lab Lift resets ±0.5 rad, rejection-sampled for ≥ 2 cm clearance (**V**) |

A start sample is rejected when either of these holds:
- it is closer than 1 cm to anything, including the partner's start pose
  (Isaac Lab / RLBench rejection pattern, **V**);
- it cannot reach our side of the dap by `t_meet` under the min-jerk check.
  "Our side" is the configuration closest to the start that puts our palm at
  the witness pose (single-arm IK, collision-free).

## 6. Head camera (`dap/camera.py`)

The camera is modelled on an **Intel RealSense D435**. GenH2R's real system
uses D435s (**V**), and the D435i is a common humanoid head camera (**S**).
The camera is fixed relative to our base.

| Parameter | Value | Source |
| --- | --- | --- |
| Resolution | 848×480 | D435 native depth mode (D400 datasheet, Mar 2026, **V**); also the old Drake D415 model (Drake v1.20 `MakeD415CameraModel`, **V**) |
| Vertical FOV | 58° (gives 88.8° horizontal, fx = fy = 433 px) | D435 depth FOV 87°×58° (datasheet, **V**). Drake's default of 45° leaves almost no margin around the workspace |
| Rate | 30 fps | D435, Azure Kinect, DexYCB (**V**). Use 15 fps as a stress case (DROID records at 15 Hz, **V**) |
| Depth range | 0.195–3.0 m | D435 Min-Z at 848×480 is 195 mm (datasheet Table 4-12, **V**); ideal range 0.3–3 m (**S**) |
| Clipping | 0.01–10 m | Drake default near (**V**); `manipulation` repo far (**V**) |
| Mount | (−0.4, 0, 1.45) m in our base frame, looking at (0.7, 0, 0.6), so pitched about 38° down | RoboCasa cameras sit 0.5–0.6 m behind the base, 0.95–1.15 m up, pitched 26–35° down (**V**). Head heights: Fetch 1.06–1.45 m (URDF, **V**), TIAGo 1.1–1.45 m (**V**), HSR 1.0–1.35 m (**V**) |
| Latency | 1 frame (33 ms) | **Ours**. Extreme Parkour uses a fixed 80 ms (**V**); Robot Parkour uses U[0.2, 0.26] s (**V**). HandoverSim notes that its timings leave out perception latency (**V**) |
| Segmentation | Drake label image (label = body index) | Ground truth. This is a known simplification; see `docs/perception_pipeline.md` |

**Other benchmarks, for comparison** (all **V**):

| Benchmark | Resolution | FOV | Notes |
| --- | --- | --- | --- |
| HandoverSim (third-person) | 1600×900 | 60° vertical | |
| GenH2R (third-person) | 1280×720 | 60° | |
| HandoverSim / GenH2R wrist cam | 224×224 | 90° | |
| ManiSkill3 | 128×128 | 90° | 20 Hz control |
| robosuite | 256×256 | 45° | 20 Hz |
| RoboCasa | 128–256 px | 60° | |
| Drake `CameraConfig` default | 640×480 | 45° | 10 Hz |

**How our setup differs from the references.** All three of our references
sense without vision:
- Kim et al. 2014 used OptiTrack motion capture (**S**).
- Salehian et al. 2016 used OptiTrack at about 240 Hz (**S**).
- Fitter & Kuchenbecker 2016 used wrist accelerometers and no vision (**S**).

A single markerless RGB-D head camera at 30 Hz is a much harder perception
setting.

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
| 2 | Lateral jitter | shift by N(0, σ_s²) px, bilinear resample; mixed/flying pixels at edges | Handa et al. 2014 (ICL-NUIM), σ_s = 0.5 px (**V**); matches Nguyen et al. 2012 σ_L ≈ 0.8 px (**V**) |
| 3 | Axial noise | Gaussian noise in disparity d = f·B/z, σ = 0.08 px. This gives RMS_z = z²·σ/(f·B): 0.9 / 3.7 / 8.3 mm at 0.5 / 1 / 1.5 m (our f = 433 px) | Intel "Tuning depth cameras for best performance" whitepaper: subpixel 0.08 typical, < 0.1, recalibrate if > 0.2 (**V**) |
| 4 | Fixed-pattern noise | coarse (8× downsampled) Gaussian disparity field, bilinear-upsampled, **held for the episode** | Dex-Net 2.0 Gaussian-process noise (Mahler et al. 2017, App. A, **V**). The datasheet's temporal noise ≤1% vs spatial noise ≤2% (**V**) suggests part of the error is fixed |
| 5 | Scale bias | depth × α, α ~ N(1, 0.5%) per episode | D400 Z-accuracy ±2% (**V**); Dex-Net's gamma (k = 1000, about 3%) is the harsh end (**V**) |
| 6 | Grazing-angle dropout | invalid where the surface-to-ray angle exceeds 75° (normals from the depth image) | Nguyen et al. 2012: invalid pixels rise sharply at extreme angles (**V**); threshold is **Ours** |
| 7 | Stereo shadow | an invalid band f·B·(1/z_fg − 1/z_bg) px wide to the **left** of foreground edges (the left imager is the D400 reference) | D400 datasheet §4.5 (**V**); about 18 px for a palm at 0.6 m against 1.2 m |
| 8 | Holes | Bernoulli(1%) pixels + 2 rectangular blobs of 4–16 px | D400 fill rate ≥ 99% (**V**); block artifacts as in Zhuang et al. 2023 (Robot Parkour code, **V**) |
| 9 | Quantize | 1 mm | D400 depth unit (**V**) |

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

## 8. RL environment (`dap/envs/`)

`gymnasium.make("dap/PalmMeet-v0", obs_mode="state" | "state+depth", scenario="nominal", ...)`

- **Base class:** a plain `gymnasium.Env`, not `pydrake.gym.DrakeGymEnv`.
  - `DrakeGymEnv` wraps a Simulator-driven Diagram with an action input
    port, but our loop is kinematic.
  - In Drake 1.57 and 1.58 it also ignores `info_handler` (fixed upstream in
    PR #25228) and rejects reset `options` (**V**).
  - Revisit it if the arm becomes dynamic.
- **Rates:** policy at 20 Hz over the 100 Hz kinematic loop. robosuite,
  ManiSkill3 and LIBERO all control at 20 Hz (**V**); the `manipulation`
  BoxFlipUp env runs at 10 Hz (**V**).
- **Action:** `Box(-1, 1, (8,))`.
  - Dims 0–6: joint velocity as a fraction of each joint's limit (1 means
    full speed).
  - Dim 7: finger-hinge target, from −1 (open) to +1 (closed).
  - Symmetric [-1, 1] actions are what Gymnasium's checker expects (**V**).
- **Observation:** a `Dict`.
  - `state`: a float32 vector of 39 values (`STATE_LAYOUT`): q, q̇, hinge,
    our palm position and normal, the KF estimate of the partner palm's
    position, velocity and normal, t, and the previous action. All bounds are
    finite.
  - `depth` (in `state+depth` mode): (1, H, W) float32 metres, 0 meaning
    invalid. It comes from the head camera at `camera_scale` (default 0.25,
    giving 212×120), with `depth_noise` (default `med`) and one frame of
    latency.
  - Float metres follows Isaac Lab; ManiSkill uses uint16 millimetres
    (**V**). Stable-Baselines3 only treats uint8 images as images, so a
    float depth observation needs `normalize_images=False` and a custom CNN
    feature extractor (**V**).
- **Episode end:**
  - `terminated` on palm contact (faces within 1 cm) or any collision
    (clearance < 0).
  - `truncated` 1 s after the partner's motion ends.
- **Reward:** `reward_fn(info) -> float`, which returns 0 when not given. No
  reward shaping yet. `info` carries ground truth: palm gap, angle error,
  arm distance, clearance, contact, collision, partner pose and velocity,
  and t_meet.
- **Determinism:** the scenario, sensor noise and depth noise all come from
  `self.np_random`. `gymnasium.utils.env_checker.check_env` passes with no
  warnings in both modes (**E**).
- **Cost:** about 1.2 ms per policy step in state mode and about 15 ms with
  depth at 212×120 (**E**). Reset is dominated by scenario sampling (about
  0.3 s).
- **Training dependencies are not installed.** For later: stable-baselines3
  2.9.2 needs `gymnasium>=0.29.1,<2` and `torch>=2.8`; Drake's and the
  `manipulation` book's examples use SB3 PPO with `SubprocVecEnv`.

## 9. Open decisions

- **Camera position.** Centered behind our base gives realistic
  self-occlusion; an offset of about 0.2–0.3 m sideways would reduce it.
  Should camera position be an experimental axis?
- **Segmentation.** Ground-truth labels vs. colour or geometric segmentation.
- **Checkpoints.** They are still supported (sampled toward the nominal
  meeting prior, so they no longer leak the true meeting point), but every
  config except the old H1/H2 sweeps uses 0.
- **Meeting type.** Only the opposed-parallel dap so far. The facing cone
  (45°) and the free rotation about the palm normal are the knobs for adding
  more.
- **Speed sweeps.** Faster partners are rejected more often, because of
  partner joint-speed limits and the palm speed cap. Conditions with
  different `speed_scale` therefore see different scenario distributions, as
  well as different speeds.

## References

- Kim, Shukla, Billard. Catching objects in flight. T-RO 30(5), 2014.
- Salehian, Khoramshahi, Billard. A dynamical system approach for softly catching a flying object. T-RO 32(2), 2016.
- Fitter, Kuchenbecker. Equipping the Baxter robot with human-inspired hand-clapping skills. RO-MAN 2016.
- Flash, Hogan. The coordination of arm movements. J. Neurosci. 5(7), 1985.
- Flash, Henis. Arm trajectory modifications during reaching towards visual targets. J. Cogn. Neurosci., 1991.
- Chan et al. arXiv:2108.12780, 2021 (min-jerk handover reaches).
- Nemlekar, Dutia, Li. Object transfer point estimation for fluent human-robot handovers. ICRA 2019.
- Controzzi et al. Humans adjust their grip force when passing an object according to the observed speed of the partner's reaching out movement. 2018.
- Chao et al. HandoverSim. ICRA 2022, arXiv:2205.09747.
- Wang et al. GenH2R. CVPR 2024, arXiv:2401.00929.
- Christen et al. Learning human-to-robot handovers from point clouds. CVPR 2023, arXiv:2303.17592.
- Vahrenkamp et al. Humanoid motion planning for dual-arm manipulation and re-grasping tasks. IROS 2009.
- Bäuml et al. Kinematically optimal catching a flying ball with a hand-arm-system. IROS 2010.
- Intel RealSense D400 Series Datasheet (Mar 2026); Intel, "Tuning depth cameras for best performance."
- Nguyen, Izadi, Lovell. Modeling Kinect sensor noise for improved 3D reconstruction and tracking. 3DIMPVT 2012.
- Khoshelham, Elberink. Accuracy and resolution of Kinect depth data. Sensors 12(2), 2012.
- Handa et al. A benchmark for RGB-D visual odometry, 3D reconstruction and SLAM (ICL-NUIM). ICRA 2014.
- Mahler et al. Dex-Net 2.0. RSS 2017, arXiv:1703.09312.
- Zhang et al. Close the optical sensing domain gap by physics-grounded active stereo sensor simulation. T-RO 2023, arXiv:2201.11924.
- Dai et al. DREDS. ECCV 2022, arXiv:2208.03792.
- Zhuang et al. Robot Parkour Learning. CoRL 2023; Cheng et al. Extreme Parkour. arXiv:2309.14341.
- Benchmarks: ManiSkill3, robosuite, RoboCasa, LIBERO, RLBench, Isaac Lab, DROID (arXiv:2403.12945), DexYCB (arXiv:2104.04631).
