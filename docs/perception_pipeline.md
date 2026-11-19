# Perception pipeline (planned, not implemented; optional)

The main method is now an end-to-end policy that learns its own perception
from depth frames (`pixels` mode, `docs/setup.md` section 8), so this
explicit pipeline is no longer on the critical path. It is still useful
for a modular comparison (depth -> palm pose -> filter -> controller), and
its palm estimate could serve as an auxiliary training target for the pixel
policy.

Until it exists, the planner interface and the `state` env modes get the
partner palm pose from `dap.partner.NoisyPoseSensor`: ground truth plus
i.i.d. Gaussian noise (5 mm, 2°), every 10 ms, with no latency, no dropouts
and no occlusion. That is more favourable than the camera will be
(`docs/setup.md`, section 10). The head camera, depth noise and frame
timing the pipeline will consume already exist (sections 6 and 7).

## Per frame (30 Hz, delivered with one frame of latency by `camera.FrameBuffer`)

1. **Render** depth (and labels) from the head camera, then apply
   `DepthNoiseModel`.
2. **Segment the partner palm.**
   - First version: the Drake label image (`Scene.palm_label(PARTNER)`). This
     is ground-truth segmentation and must be stated as such in the report.
   - Later: colour segmentation (give the palm a distinct colour), or a
     geometric crop around the KF's predicted position.
   - Labelled frames double as auto-labelled training data for a learned
     segmenter or pose estimator.
3. **Back-project** the masked pixels to 3D with the camera intrinsics
   (`HeadCameraConfig.intrinsics()`). Transform them to the world frame with
   `HeadCameraConfig.X_WC()`.
4. **Fit the palm.** The palm is a flat 10×12 cm box, so:
   - RANSAC plane fit, with the inlier threshold tied to the axial noise
     σ_z(z) (`depth_noise.axial_rms`) rather than a constant;
   - palm normal = plane normal, oriented toward our side (the camera sees
     the palm face);
   - palm center = centroid of the inliers, corrected for edge pixels lost to
     dropout and shadow;
   - optional: ICP against the hand model for the rotation about the normal.
     The finger plate breaks the box's symmetry.
5. **Measurement covariance** from the fit residuals and the number of
   inliers. Report "no measurement" below a minimum inlier count.

## Filter (`dap/estimator.py`)

- **Update** with each delivered measurement, stamped with its *capture*
  time, not its delivery time.
- **Gate outliers** with a Mahalanobis distance test against the predicted
  measurement.
- **Dropouts:** when a frame has no measurement, run predict only. Most
  dropouts happen near contact, when our own hand occludes the partner's palm
  from the head camera, so the last moments before the touch will often run
  on prediction alone.

## Metrics to add

- Visible palm pixels over time.
- Length of the last blackout before contact.
- Estimation error vs. time-to-contact.
- Perception wall time per frame.
