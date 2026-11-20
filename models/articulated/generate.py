"""Build the project's original right-hand meshes and SDF. Run with NumPy.

Distances are in metres except in the palm loft, which is sculpted in mm.
The palm, tapered phalanges and nail plates are closed, smooth-shaded meshes.
Dimensions and flexion limits are design choices for this kinematic model.
"""

from math import cos, pi, sin
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


HERE = Path(__file__).resolve().parent
SKIN = "0.71 0.44 0.31 1"
NAIL = "0.79 0.56 0.49 1"
# Index ranges in the 59 wrist-to-knuckle stations and 96 points per ring.
# The upper half-ring runs from the little-finger edge (0) to the thumb edge (48).
PALM_LENGTHS = (("heel", 0, 25), ("mid", 25, 42), ("knuckles", 42, 58))
PALM_WIDTHS = (("hypothenar", 0, 18), ("center", 18, 30), ("thenar", 30, 48))


def palm_sections():
    """Split the palm loft into convex collision patches at the hollow and pads."""
    for region, first, last in PALM_LENGTHS:
        for pad, edge, other in PALM_WIDTHS:
            name = f"palm_{region}_{pad}"
            if (region, pad) == ("mid", "center"):
                # A single convex central patch would span both sides of the
                # cavity. Four pieces meet at its lowest point (x≈0, y=0).
                for end, lo, hi in (("heel", first, 34), ("distal", 34, last)):
                    for side, a, b in (("ulnar", edge, 24), ("radial", 24, other)):
                        yield f"{name}_{end}_{side}", lo, hi, a, b
            else:
                yield name, first, last, edge, other


def surface(name, rows):
    """Triangulate cyclic rings ordered along the finger and average vertex normals."""
    rings = np.array(rows)
    n, m, _ = rings.shape
    verts = list(rings.reshape(-1, 3))
    verts.extend((rings[0].mean(axis=0), rings[-1].mean(axis=0)))
    faces = []
    for i in range(n - 1):
        for j in range(m):
            a, b = i * m + j, i * m + (j + 1) % m
            faces.extend(((a, b, a + m), (b, b + m, a + m)))
    for j in range(m):
        k = (j + 1) % m
        faces.extend(((n * m, k, j), (n * m + 1, (n - 1) * m + j, (n - 1) * m + k)))
    verts, faces = np.array(verts), np.array(faces)
    normals = np.zeros_like(verts)
    for a, b, c in faces:
        normal = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        normals[[a, b, c]] += normal
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12)
    with (HERE / f"{name}.obj").open("w") as f:
        f.write("# Original geometry. Regenerate with models/articulated/generate.py\n")
        for v in verts:
            f.write("v %.7f %.7f %.7f\n" % tuple(v))
        for v in normals:
            f.write("vn %.6f %.6f %.6f\n" % tuple(v))
        for tri in faces + 1:
            f.write("f " + " ".join(f"{i}//{i}" for i in tri) + "\n")


def palm():
    # Stations describe x, radial/ulnar extent, and half thickness. The wrist
    # narrows, then flares at the heel and rolls towards the little finger.
    profile = np.array(
        [
            [-68, 19, 17, 10],
            [-64, 24, 21, 12],
            [-56, 25, 22, 12],
            [-48, 27, 24, 13],
            [-39, 31, 28, 14],
            [-28, 36, 31, 14.5],
            [-15, 39, 35, 13.6],
            [0, 42, 38, 12],
            [15, 42.5, 39.5, 11],
            [26, 42, 39, 10.7],
            [34, 39, 35, 9.8],
            [39, 33, 28, 8],
            [43, 24, 19, 6.5],
            [46, 14, 11, 4.4],
            [47, 5, 4, 2],
        ]
    )
    rows = []
    for x in np.linspace(-68, 47, 59):
        radial, ulnar, height = [
            np.interp(x, profile[:, 0], profile[:, k]) for k in (1, 2, 3)
        ]
        ring = []
        for t in np.linspace(0, 2 * pi, 96, endpoint=False):
            c, s = cos(t), sin(t)
            y = (ulnar if c >= 0 else radial) * np.sign(c) * abs(c) ** 0.83
            z = height * np.sign(s) * abs(s) ** 0.72
            # The outer metacarpals roll toward the palm when the hand is cupped.
            # Apply the transverse arch to the back as well as the palm.
            z += 8.0 * (y / 40) ** 2 * np.exp(-((x + 2) / 40) ** 4)
            if s > 0:
                # The fleshy pads at the thumb, little finger and finger bases
                # surround the central hollow. The heel stays full toward the wrist.
                z += s * (
                    10.0 * np.exp(-(((x + 13) / 28) ** 2) - ((y + 24) / 13) ** 2)
                    + 7.6 * np.exp(-(((x + 8) / 32) ** 2) - ((y - 25) / 12) ** 2)
                    + 4.8 * np.exp(-(((x - 25 + 0.12 * y) / 8) ** 2) - (y / 33) ** 4)
                    - 7.8 * np.exp(-((x / 23) ** 2) - (y / 17) ** 2)
                )
                z -= (
                    0.55
                    * s
                    * np.exp(-(((x - 11 + 0.12 * y) / 1.4) ** 2) - (y / 31) ** 4)
                )
            else:
                # The rounded dorsal metacarpal heads give the open palm a knuckle line.
                for cy in (-30, -10, 11, 30):
                    z -= (
                        1.8
                        * (-s)
                        * np.exp(
                            -(((x - 30 + 0.12 * abs(cy)) / 11) ** 2)
                            - ((y - cy) / 6) ** 2
                        )
                    )
            ring.append(np.array([x, y, z]) * 0.001)
        rows.append(ring)
    surface("palm", rows)
    # A convex hull of the entire palm would bridge the hollow. Partition the
    # very same skin surface in x and y before convexifying each patch in Drake.
    # All patches extend from the palmar side to the dorsal side of the hand.
    for name, first, last, edge, other in palm_sections():
        stations = [*range(first, last, 2), last]
        top = [*range(edge, other, 2), other]
        bottom = [96 - i for i in reversed(top) if i not in (0, 48)]
        section = [[rows[i][j] for j in top + bottom] for i in stations]
        surface(name, section)


def finger(name, distal=False, thumb_web=False):
    # Normalized axial coordinate and normalized lateral/dorsopalmar radii.
    if distal:
        stations = [
            (-0.29, 0.05, 0.06),
            (-0.26, 0.44, 0.45),
            (-0.15, 0.84, 0.86),
            (0, 1, 1),
            (0.25, 0.95, 1),
            (0.49, 0.94, 1.01),
            (0.65, 0.90, 1.00),
            (0.83, 0.77, 0.91),
            (0.93, 0.55, 0.70),
            (0.98, 0.31, 0.45),
            (1, 0.04, 0.09),
        ]
    else:
        stations = [
            (-0.24, 0.06, 0.07),
            (-0.21, 0.43, 0.46),
            (-0.13, 0.81, 0.80),
            (0, 1.04, 1.03),
            (0.16, 1.02, 1.01),
            (0.36, 0.93, 0.96),
            (0.64, 0.85, 0.89),
            (0.80, 0.86, 0.90),
            (0.93, 0.90, 0.94),
            (1.0, 0.90, 0.96),
            (1.10, 0.72, 0.72),
            (1.18, 0.36, 0.36),
            (1.19, 0.07, 0.07),
        ]
    profile = np.array(stations)
    rows = []
    for u in np.unique(
        np.r_[np.linspace(stations[0][0], stations[-1][0], 28), profile[:, 0]]
    ):
        width, height = [np.interp(u, profile[:, 0], profile[:, k]) for k in (1, 2)]
        ring = []
        for t in np.linspace(0, 2 * pi, 32, endpoint=False):
            c, s = cos(t), sin(t)
            z = height * np.sign(s) * abs(s) ** 0.83
            if s > 0:
                z += 0.10 * s * np.exp(-(((u - 0.37) / 0.22) ** 2))
                z -= 0.075 * s * np.exp(-(((u - 0.11) / 0.04) ** 2))
            y = width * np.sign(c) * abs(c) ** 0.9
            if thumb_web and c > 0:
                # First web space fans from the thumb metacarpal back into
                # the radial palm; it travels with the thumb during opposition.
                y += 1.45 * c**3 * np.exp(-(((u - 0.46) / 0.46) ** 2))
                z -= 0.10 * c * c
            ring.append((u, y, z))
        rows.append(ring)
    surface(name, rows)


def nail():
    # A thin domed nail on the dorsal face, in distal-segment coordinates.
    rows = []
    for x, width, z in [
        (0.35, 0.08, -0.98),
        (0.37, 0.32, -0.985),
        (0.40, 0.52, -0.985),
        (0.45, 0.61, -0.985),
        (0.54, 0.65, -0.99),
        (0.67, 0.65, -0.963),
        (0.76, 0.63, -0.94),
        (0.82, 0.58, -0.91),
        (0.86, 0.45, -0.88),
        (0.89, 0.25, -0.85),
        (0.9, 0.06, -0.84),
    ]:
        rows.append(
            [
                (x, width * cos(t), z + 0.36 * (width * cos(t)) ** 2 + 0.075 * sin(t))
                for t in np.linspace(0, 2 * pi, 32, endpoint=False)
            ]
        )
    surface("nail", rows)


def tag(parent, kind, text=None, **attributes):
    child = ET.SubElement(parent, kind, attributes)
    if text is not None:
        child.text = text
    return child


def numbers(values):
    return " ".join(f"{v:.8g}" for v in values)


def inertial(link, mass, size, center=(0, 0, 0)):
    inertia = tag(link, "inertial")
    tag(inertia, "pose", numbers((*center, 0, 0, 0)))
    tag(inertia, "mass", str(mass))
    diag = tag(inertia, "inertia")
    a, b, c = size
    for key, value in zip(
        ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"),
        [
            mass * (b * b + c * c) / 12,
            mass * (a * a + c * c) / 12,
            mass * (a * a + b * b) / 12,
            0,
            0,
            0,
        ],
    ):
        tag(diag, key, str(value))


def visual(link, name, mesh, scale, color):
    obj = tag(link, "visual", name=name)
    geo = tag(tag(obj, "geometry"), "mesh")
    tag(geo, "uri", f"package://dap/articulated/{mesh}.obj")
    tag(geo, "scale", numbers(scale))
    tag(tag(obj, "material"), "diffuse", color)


def capsule(link, radius, length):
    obj = tag(link, "collision", name="pad")
    tag(obj, "pose", numbers((length / 2, 0, 0, 0, pi / 2, 0)))
    geo = tag(tag(obj, "geometry"), "capsule")
    tag(geo, "radius", str(radius))
    tag(geo, "length", str(max(0.001, length - 1.04 * radius)))


def phalanx(
    model, digit, part, joint_name, parent, pose, length, radius, axis, upper, terminal
):
    name = f"{digit}_{part}"
    link = tag(model, "link", name=name)
    tag(link, "pose", numbers(pose), relative_to=parent)
    inertial(
        link,
        radius * length * 40,
        (length, 2 * radius, 1.8 * radius),
        (length / 2, 0, 0),
    )
    scale = (length, radius, radius * 0.90)
    mesh = "tip" if terminal else "phalanx"
    if part == "metacarpal":
        mesh = "thumb_web"
    visual(link, "skin", mesh, scale, SKIN)
    if terminal:
        visual(link, "nail", "nail", scale, NAIL)
    capsule(link, radius * 0.85, length)
    joint = tag(model, "joint", name=f"{digit}_{joint_name}", type="revolute")
    tag(joint, "parent", parent)
    tag(joint, "child", name)
    axis_node = tag(joint, "axis")
    tag(axis_node, "xyz", numbers(axis))
    lim = tag(axis_node, "limit")
    # Thumb opposition has an abducted approach pose across neutral.
    tag(lim, "lower", "-0.70" if joint_name == "cmc" else "0")
    tag(lim, "upper", str(upper))
    tag(lim, "velocity", "8")
    tag(lim, "effort", "2")
    return name


def model():
    doc = ET.Element("sdf", {"version": "1.9", "xmlns:drake": "http://drake.mit.edu"})
    hand = tag(doc, "model", name="articulated_hand")
    base = tag(hand, "link", name="palm")
    inertial(base, 0.27, (0.115, 0.082, 0.026))
    visual(base, "skin", "palm", (1, 1, 1), SKIN)
    for name, *_ in palm_sections():
        col = tag(base, "collision", name=name)
        geo = tag(tag(col, "geometry"), "mesh")
        tag(geo, "uri", f"package://dap/articulated/{name}.obj")
        tag(geo, "drake:declare_convex")
    wrist = tag(hand, "frame", name="wrist", attached_to="palm")
    tag(wrist, "pose", "-.061 0 0 0 1.57079632679 0", relative_to="palm")
    contact = tag(hand, "frame", name="palm_face", attached_to="palm")
    tag(contact, "pose", "0 0 .0047 0 0 0", relative_to="palm")
    # Small roll differences aim flexion toward the transverse arch. Splay
    # remains small so adjacent digits can form the outer lip of the cup.
    fingers = [
        ("index", 0.038, -0.0295, 0.002, -0.12, -0.015, (0.043, 0.026, 0.020), 0.0093),
        ("middle", 0.042, -0.0098, -0.001, -0.04, 0.0, (0.047, 0.029, 0.021), 0.0095),
        ("ring", 0.039, 0.0107, 0.0, 0.07, 0.015, (0.044, 0.028, 0.020), 0.0089),
        ("little", 0.031, 0.0295, 0.004, 0.18, 0.03, (0.035, 0.021, 0.017), 0.0076),
    ]
    for digit, x, y, z, roll, splay, lengths, r in fingers:
        parent = "palm"
        pose = (x, y, z, roll, 0, splay)
        for i, (part, joint, limit) in enumerate(
            zip(
                ("proximal", "middle", "distal"),
                ("mcp", "pip", "dip"),
                (1.55, 1.70, 1.35),
            )
        ):
            length = lengths[i]
            parent = phalanx(
                hand,
                digit,
                part,
                joint,
                parent,
                pose,
                length,
                r * (1, 0.84, 0.72)[i],
                (0, -1, 0),
                limit,
                i == 2,
            )
            pose = (length, 0, 0, 0, 0, 0)
    parent = "palm"
    pose = (-0.035, -0.026, 0.003, 0, -0.18, -0.78)
    for i, (part, joint, length, r, axis, limit) in enumerate(
        [
            ("metacarpal", "cmc", 0.036, 0.012, (0, -0.65, 0.75993421), 0.95),
            ("proximal", "mcp", 0.025, 0.010, (0, -0.50, 0.86602540), 1.25),
            ("distal", "ip", 0.022, 0.0086, (0, -0.50, 0.86602540), 1.40),
        ]
    ):
        parent = phalanx(
            hand, "thumb", part, joint, parent, pose, length, r, axis, limit, i == 2
        )
        pose = (length, 0, 0, 0, -0.06, 0.06)
    ET.indent(doc, space="  ")
    (HERE.parent / "articulated_hand.sdf").write_text(
        '<?xml version="1.0"?>\n<!-- Generated by articulated/generate.py. -->\n'
        + ET.tostring(doc, encoding="unicode")
        + "\n"
    )


if __name__ == "__main__":
    palm()
    finger("phalanx")
    finger("tip", distal=True)
    finger("thumb_web", thumb_web=True)
    nail()
    model()
