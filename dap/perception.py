"""Label-free perception of the partner palm from head-camera depth, for the
`servo_depth` baseline (dap/baselines.py). Description and measurements:
docs/perception_pipeline.md.

Per delivered frame, stamped with its capture time:
  1. Segment without labels: render the depth the camera would see of what
     our robot knows (the static scene and our own arm at its joint angles
     at capture) and keep the pixels clearly in front of it, within the
     partner's reach. They are the partner's arm and hand.
  2. Pick the hand: the points near the filter's predicted plate middle
     while tracking, otherwise near the end of the partner's arm (the point
     farthest from its shoulder).
  3. Fit the hand plate (palm + fingers, 18 x 12 cm, models/dap_hand.sdf):
     RANSAC plane with a truncated-quadratic (MSAC) score and an inlier band
     tied to the D435 axial noise, refit by SVD. Normal: toward the camera.
     Finger axis: the plate's long axis.
  4. Accept the fit only if it spans the whole plate and its normal is near
     the filter's (tracking) or faces us (acquiring, which starts a fresh
     filter).
  5. Finger sign: away from the partner's arm, which holds the plate 5 cm
     from its wrist end and 13 cm from its fingertip end; while tracking it
     stays continuous unless consecutive fixes disagree. Palm face center:
     the plate's middle, 4 cm back toward the wrist.
Accepted fixes update a PalmEstimator; other frames only predict.
"""

from dataclasses import replace

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LoadModelDirectivesFromString,
    ProcessModelDirectives,
)

from dap.camera import add_head_camera
from dap.depth_noise import NATIVE_WIDTH, NOISE_PRESETS, axial_rms
from dap.estimator import PalmEstimator
from dap.scene import OURS, _add_static_geometry, _arm_directives, make_parser

SHOULDER_Z = 0.3405  # m, iiwa7 joint-2 axis above its base (Drake iiwa7 model)
# The partner's arm and hand lie within this of its shoulder: 0.945 m to the
# palm face with the arm straight, plus 0.143 m to a plate corner (E, model).
REACH = 1.1
MARGIN = (0.03, 0.03)  # foreign = nearer than the known world by 3 cm + 3% of depth (Ours)
GATE = 0.15  # m, hand points lie within this of the plate middle (corner: 10.8 cm)
ARM_RADIUS = 0.25  # m, partner-arm points used for the finger sign (Ours)
MIN_INLIERS = 15  # (Ours)
RANSAC_ITERS = 100
# A fit counts only if it spans the plate: the 5-95% extents of a uniform
# 18 x 12 cm plate are 0.9 x its sides, matched within 20% (Ours). Partial
# views (an occluded or edge-on plate) bias the middle, so they are dropped.
PLATE_EXTENT = (0.9 * 0.18, 0.9 * 0.12)
EXTENT_TOL = 0.2
NORMAL_GATE_DEG = 30.0  # while tracking (Ours)
# When acquiring, the palm must face us: the partner's palm normal stays
# within the facing cone (45 deg) plus the start tilt (25 deg) of the
# horizontal direction to our base (dap.feasibility, dap.partner); measured
# max 46 deg nominal, 55 deg wide over 30 training scenarios each (E).
FACING_DEG = 70.0
FLIP_AFTER = 2  # consecutive fixes against the filter's finger sign flip it (Ours)
LOST_AFTER = 0.5  # s without a measurement -> re-acquire the hand (Ours)
PALM_FROM_MIDDLE = 0.04  # m: the plate spans x = -0.05..0.13 in the palm frame
POS_NOISE = 0.01  # m, measurement std given to the filter (Ours)
# The stand-in filter's smoothing time constant (alpha 0.2 per 10 ms update)
# at 30 Hz: 1 - 0.8 ** (100 / 30).
NORMAL_ALPHA = 1 - 0.8 ** (100 / 30)


class KnownWorld:
    """What our robot knows of the scene: the floor, both pedestals and our
    own arm and hand (no partner), seen by the same head camera."""

    def __init__(self, layout, camera):
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        directives = "directives:" + _arm_directives(OURS, (0, 0, 0), 0)
        ProcessModelDirectives(LoadModelDirectivesFromString(directives), plant,
                               make_parser(plant))
        _add_static_geometry(plant, layout)
        plant.Finalize()
        sensor = add_head_camera(builder, plant, scene_graph, replace(camera, label=False))
        diagram = builder.Build()
        self.context = diagram.CreateDefaultContext()
        self.plant, self.plant_context = plant, plant.GetMyMutableContextFromRoot(self.context)
        self.sensor, self.sensor_context = sensor, sensor.GetMyContextFromRoot(self.context)

    def render(self, q_ours):
        """Depth (H, W) with our arm at `q_ours` (inf = nothing in range)."""
        self.plant.SetPositions(self.plant_context, q_ours)
        img = self.sensor.depth_image_32F_output_port().Eval(self.sensor_context)
        return np.array(img.data[:, :, 0], copy=True)


def _min3x3(z):
    """3 x 3 minimum filter: the known world grown by a pixel, so jittered
    and mixed pixels at our arm's silhouette are not taken for the partner."""
    h, w = z.shape
    p = np.pad(z, 1, mode="edge")
    return np.min([p[i:i + h, j:j + w] for i in range(3) for j in range(3)], axis=0)


class DepthPalmTracker:
    """Partner palm pose from depth frames. Uses only the head camera's
    configuration and where the partner stands (`layout`); `estimator` is
    its filter, replaced by a fresh one whenever the hand is re-acquired."""

    def __init__(self, layout, camera):
        self.world = KnownWorld(layout, camera)
        self.fx, self.fy, self.cx, self.cy = camera.intrinsics()
        X = camera.X_WC()
        self.R_WC, self.p_WC = X.rotation().matrix(), X.translation()
        self.v, self.u = np.mgrid[0:camera.height, 0:camera.width]
        # Disparity noise of the `med` preset in this image's pixels.
        self.sigma_px = NOISE_PRESETS["med"].sigma_subpixel * camera.width / NATIVE_WIDTH
        base = layout.partner_base().translation()
        self.shoulder = base + [0.0, 0.0, SHOULDER_Z]
        self.toward_us = -base * [1.0, 1.0, 0.0] / np.linalg.norm(base[:2])
        self.estimator = PalmEstimator(pos_noise=POS_NOISE, normal_alpha=NORMAL_ALPHA)
        self.disagree = 0  # consecutive fixes whose finger sign disagrees with the filter
        self.rng = np.random.default_rng(0)  # RANSAC samples

    def observe(self, t, depth, q_ours):
        """A frame captured at time `t`, with our joint positions then.
        Updates the filter and returns the fix (palm face center, normal,
        finger direction), or None."""
        est = self.estimator
        tracking = est.x is not None and t - est.t <= LOST_AFTER
        fit = self.fit_plate(self.foreign_points(depth, q_ours), tracking)
        if fit is None:
            return None
        middle, n, axis, cue = fit
        if tracking:
            sign = 1.0 if axis @ est.finger >= 0 else -1.0  # continuity
            self.disagree = self.disagree + 1 if cue == -sign else 0
            if self.disagree >= FLIP_AFTER:
                sign, self.disagree = -sign, 0
                est.finger = -est.finger  # low-pass blending opposite vectors breaks
        elif cue:
            self.estimator = est = PalmEstimator(pos_noise=POS_NOISE, normal_alpha=NORMAL_ALPHA)
            sign, self.disagree = cue, 0
        else:
            return None
        x = sign * axis
        fix = middle - PALM_FROM_MIDDLE * x, n, x
        est.update(t, *fix)
        return fix

    def foreign_points(self, depth, q_ours):
        """World points of the pixels clearly in front of the known world,
        within the partner's reach (which also drops depth artifacts)."""
        z = depth
        mask = (z > 0) & (z < _min3x3(self.world.render(q_ours)) - (MARGIN[0] + MARGIN[1] * z))
        z, u, v = z[mask], self.u[mask], self.v[mask]
        P = np.stack([(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z], axis=1)
        P = P @ self.R_WC.T + self.p_WC
        return P[np.linalg.norm(P - self.shoulder, axis=1) < REACH]

    def fit_plate(self, P, tracking):
        """The hand plate among the foreign points P: (middle, normal toward
        the camera, long axis, finger-sign cue in {-1, 0, 1} along that
        axis), or None if no fit is accepted."""
        est = self.estimator
        if len(P) < MIN_INLIERS:
            return None
        # Search around the predicted plate middle, or the end of the partner's arm.
        center = (est.predict(0.0)[0] + PALM_FROM_MIDDLE * est.rotation()[:, 0] if tracking
                  else P[np.argmax(np.linalg.norm(P - self.shoulder, axis=1))])
        near = np.linalg.norm(P - center, axis=1) < GATE
        if near.sum() < MIN_INLIERS:
            return None
        inl = self._ransac(P[near])
        if inl.sum() < MIN_INLIERS:
            return None
        plate = P[near][inl]
        c = plate.mean(axis=0)
        _, _, Vt = np.linalg.svd(plate - c, full_matrices=False)
        axis, n = Vt[0], (Vt[2] if Vt[2] @ (self.p_WC - c) > 0 else -Vt[2])
        # While tracking, the normal stays near the filter's; when acquiring,
        # the palm must face us.
        ref, max_deg = (est.normal, NORMAL_GATE_DEG) if tracking else (self.toward_us, FACING_DEG)
        if n @ ref < np.cos(np.deg2rad(max_deg)):
            return None
        side = np.cross(n, axis)
        lo, hi = np.percentile(np.stack([(plate - c) @ axis, (plate - c) @ side], axis=1),
                               [5, 95], axis=0)
        if np.any(np.abs((hi - lo) / PLATE_EXTENT - 1) > EXTENT_TOL):
            return None
        middle = c + 0.5 * (lo[0] + hi[0]) * axis + 0.5 * (lo[1] + hi[1]) * side
        # Finger-sign cue: away from the rest of the partner's arm.
        rest = np.ones(len(P), bool)
        rest[np.flatnonzero(near)[inl]] = False
        arm = P[rest & (np.linalg.norm(P - middle, axis=1) < ARM_RADIUS)]
        cue = -np.sign(np.mean((arm - middle) @ axis)) if len(arm) else 0.0
        return middle, n, axis, cue

    def _ransac(self, H):
        """Inliers of the best plane through 3 sampled points (MSAC score:
        each inlier counts 1 - (d / tol)^2). The band is 2 sigma of the
        axial depth noise at the points' depth, at least 1 cm (the finger
        plate's face sits 5 mm behind the palm's)."""
        z = (H - self.p_WC) @ self.R_WC[:, 2]
        tol = max(2 * axial_rms(np.median(z), self.fx, self.sigma_px), 0.01)
        best, best_score = np.zeros(len(H), bool), -np.inf
        for _ in range(RANSAC_ITERS):
            a, b, c = H[self.rng.choice(len(H), 3, replace=False)]
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            if norm < 1e-12:
                continue
            d = (H - a) @ (n / norm)
            inl = np.abs(d) < tol
            score = np.sum(1 - (d[inl] / tol) ** 2)
            if score > best_score:
                best, best_score = inl, score
        return best
