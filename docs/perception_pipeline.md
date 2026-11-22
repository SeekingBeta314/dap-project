# Perception pipeline (the `servo_depth` baseline)

The main method is an end-to-end policy that learns its own perception from
depth frames (`pixels` mode, `docs/setup.md` section 8). This explicit
pipeline exists for one classical comparison on the same input: the
diff-IK servo fed by perception from the student's depth frames
(`servo_depth`, `docs/setup.md` section 1b). Code: `dap/perception.py`;
tests: `tests/test_perception.py`.

The `state` env modes and the `servo` baseline still get the partner palm
from `dap.partner.NoisyPoseSensor`: ground truth plus i.i.d. Gaussian noise
(5 mm, 2°) every 10 ms, with no latency, dropouts or occlusion.

## Input

`run_trial(..., camera=SERVO_DEPTH_CAMERA)` renders the head camera exactly
as the env renders it for the student: 212×120, captured at 30 fps with
`med` D435-style noise from the trial's camera stream, delivered one frame
later (`camera.FrameBuffer`), together with our joint positions at capture.

## Per frame

1. **Segment without labels.** Render the depth of what our robot knows:
   the floor, both pedestals and our own arm at its joint positions at
   capture (`KnownWorld`, grown by one pixel). Pixels more than 3 cm + 3% of
   depth in front of it are foreign; keeping those within the partner arm's
   reach of its shoulder (1.1 m) also drops depth artifacts. On clean depth
   this keeps exactly the partner's arm and hand (tested).
2. **Pick the hand.** Points within 15 cm of the filter's predicted plate
   middle; before the first fix, or after 0.5 s without one, of the partner
   point farthest from its shoulder (the end of its arm). Re-acquiring
   starts a fresh filter, so the jump is not read as motion.
3. **Fit the plate.** The palm and finger plate form one flat 18×12 cm
   rectangle. RANSAC plane with the MSAC score; the inlier band is 2σ of
   the axial noise at the points' depth (`depth_noise.axial_rms`), at least
   1 cm. Refit by SVD. Normal toward the camera (the palm faces us).
4. **Accept or drop.** The fit must span the plate (5–95% extents within 20%
   of 0.9 × 18 × 12 cm): partial views (an occluded or edge-on plate) bias
   the plate's middle, so they are dropped and the filter predicts. Its
   normal must be within 30° of the filter's while tracking, or face us
   (within 70° of the horizontal direction to our base) when acquiring.
5. **Pose.** Finger axis: the plate's long axis, signed away from the rest
   of the partner's arm (the arm holds the plate 5 cm from its wrist end and
   13 cm from its fingertip end). While tracking the sign stays continuous
   unless two fixes in a row disagree. Palm face center: the plate's middle,
   4 cm back toward the wrist.

Fixes update `dap.estimator.PalmEstimator` at their capture times. The servo
predicts at most 0.1 s past the last fix and then holds that pose.

## Measurements and what was dropped

On the frames of 16 `servo` trials on training scenarios (**E**): 65% of
frames give a fix; of the fixes, 84% have the palm normal within 15° (median
3.2°), 69% the position within 5 cm (median 1.9 cm), and 15% a flipped
finger direction. On clean depth (no noise, our arm still) the tests hold
the median position error under 2.5 cm and the normal error under 5°.

These gave no better fixes and are not in the code: a penalty for points in
front of the plane, a smaller arm radius for the finger cue, and rejecting
fits next to our own arm's silhouette (which also cut the fixes by 40%).
The facing check when acquiring and the two-fix rule for the finger sign
guard against failures found in review (a lock onto the partner's elbow, a
single wrong sign turning the target 180°); their effect on end-to-end
success was within noise on 24 training trials. Design choices were first
explored on evaluation seeds 0–15 and then re-checked on training
scenarios (`docs/setup.md` section 1b).

## Known limits

- A third of the frames give no fix: near contact our own hand hides much
  of the partner's plate, and the head camera sees the palm at a slant.
- When the hand is mostly hidden at the start, the tracker can lock onto a
  plate-sized patch of the partner's arm (2 of 16 training trials).
- The finger sign is wrong in about 15% of fixes. The plate is a symmetric
  rectangle, so the arm's position is the only cue to the fingers'
  direction, and a bent wrist can mislead it.
- The filter's constant-acceleration model is misspecified for the partner
  (`docs/setup.md` section 10), and biased fixes look like motion to it.
