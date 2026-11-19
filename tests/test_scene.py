import numpy as np

from dap.scene import OURS, PARTNER, Q_READY, Scene, SceneLayout, build_scene


def test_palms_face_each_other():
    diagram, plant, _ = build_scene()
    pc = plant.GetMyContextFromRoot(diagram.CreateDefaultContext())
    assert plant.num_positions() == 16  # 7 + 7 arm joints, 2 finger hinges

    def palm(name):
        frame = plant.GetFrameByName("palm", plant.GetModelInstanceByName(name))
        return plant.CalcRelativeTransform(pc, plant.world_frame(), frame)

    ours, theirs = palm("iiwa_hand"), palm("partner_hand")
    n_ours = ours.rotation().matrix()[:, 2]
    n_theirs = theirs.rotation().matrix()[:, 2]
    assert np.dot(n_ours, n_theirs) < -0.99
    assert ours.translation()[0] < theirs.translation()[0]


def test_layout_moves_partner_base():
    layout = SceneLayout(distance=1.6, lateral=0.2, yaw_deg=20.0, height=0.1)
    scene = Scene(layout)
    base = scene.plant.GetFrameByName("iiwa_link_0", scene.models[PARTNER])
    X = scene.plant.CalcRelativeTransform(scene.plant_context, scene.plant.world_frame(), base)
    assert np.allclose(X.translation(), [1.6, 0.2, 0.1])
    # Partner faces us (yawed 180 deg) plus the extra 20 deg.
    yaw = np.degrees(np.arctan2(X.rotation().matrix()[1, 0], X.rotation().matrix()[0, 0]))
    assert abs(((yaw - 200.0) + 180) % 360 - 180) < 1e-6


def test_ready_pose_is_collision_free():
    scene = Scene()
    scene.set_ours(Q_READY)
    scene.set_partner(Q_READY)
    # Closest pair is a hand's finger plate vs. its own link 7 (1.55 cm).
    assert scene.clearance() > 0.01
    assert scene.min_arm_distance() > 0.1


def test_hand_hand_contact_is_filtered():
    from dap.feasibility import check_feasible

    scene = Scene()
    w = check_feasible(scene, np.random.default_rng(0), target=[0.7, 0.0, 0.6])
    scene.set_q(w.q)
    p_ours, _ = scene.palm(OURS)
    p_partner, _ = scene.palm(PARTNER)
    assert np.linalg.norm(p_ours - p_partner) < 0.01  # the palms are touching
    hands = {scene.hands[OURS], scene.hands[PARTNER]}
    inspector = scene.scene_graph.model_inspector()
    plant = scene.plant

    def model(gid):
        return plant.GetBodyFromFrameId(inspector.GetFrameId(gid)).model_instance()

    pairs = scene._query().ComputeSignedDistancePairwiseClosestPoints(0.05)
    for p in pairs:
        a, b = model(p.id_A), model(p.id_B)
        assert not (a in hands and b in hands and a != b), "hand-hand pair not filtered"
    assert scene.clearance() > 0.0
