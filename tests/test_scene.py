import numpy as np

from dap.scene import build_scene


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
