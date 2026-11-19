import numpy as np

from dap.scene import OURS, PARTNER, Q_READY, Scene, SceneLayout, build_scene


def test_palms_face_each_other():
    diagram, plant, _ = build_scene()
    pc = plant.GetMyContextFromRoot(diagram.CreateDefaultContext())
    assert plant.num_positions() == 14  # 7 + 7 arm joints; the hands are rigid

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


def test_only_the_palm_pair_is_excluded():
    """At a palm-to-palm dap the two palm boxes touch; that pair is left to
    the scorer. Every other pair, hand-hand included, still counts as a
    collision."""
    from dap.feasibility import check_feasible

    scene = Scene()
    w = check_feasible(scene, np.random.default_rng(0), target=[0.7, 0.0, 0.6])
    scene.set_q(w.q)
    assert abs(scene.palm_palm_distance()) < 0.005  # the palms are touching
    hands = {scene.hands[OURS], scene.hands[PARTNER]}
    inspector = scene.scene_graph.model_inspector()

    def model(gid):
        return scene.plant.GetBodyFromFrameId(inspector.GetFrameId(gid)).model_instance()

    pairs = scene._query().ComputeSignedDistancePairwiseClosestPoints(0.05)
    hand_hand = [p for p in pairs if model(p.id_A) in hands and model(p.id_B) in hands
                 and model(p.id_A) != model(p.id_B)]
    assert any(scene._is_palm_pair(p) for p in hand_hand)
    assert any(not scene._is_palm_pair(p) for p in hand_hand)  # fingers etc. are checked
    assert scene.clearance() > 0.0

    # Push our arm 3 cm further into the partner: the finger plates now hit
    # the partner palm, which clearance must report.
    q = w.q.copy()
    scene.set_q(q)
    n = scene.palm(OURS)[1]
    X = scene.palm_pose(OURS)
    from dap.ik import arm_ik
    from pydrake.all import RigidTransform
    q_in = arm_ik(scene, OURS, RigidTransform(X.rotation(), X.translation() + 0.03 * n),
                  w.q_ours, w.q_ours, 0.001, 1.0)
    assert q_in is not None
    scene.set_ours(q_in)
    assert scene.clearance() < 0.0


def test_planning_context_can_filter_hands_without_touching_the_scene():
    scene = Scene()
    ctx, pc = scene.new_context(filter_hands=True)
    assert pc is not scene.plant_context
    sg = scene.scene_graph.GetMyContextFromRoot(ctx)
    q_obj = scene.scene_graph.get_query_output_port().Eval(sg)
    a, b = scene.palm_geoms[OURS], scene.palm_geoms[PARTNER]
    inspector = q_obj.inspector()
    assert inspector.CollisionFiltered(a, b)
    assert not scene.scene_graph.model_inspector().CollisionFiltered(a, b)
