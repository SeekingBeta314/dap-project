# Two hands dapping

[Watch the full-resolution motion](hand_dap.mp4) · [GIF preview](hand_dap.gif)

The two simulated iiwa arms wear anatomically shaped right hands. Each hand has 15 moving joints, including three for thumb opposition and flexion. The palm, finger segments, and nails are original meshes generated in the repository. Each body has a collision shape so the controller can query distances as the hands close. [Hand model](../models/articulated_hand.sdf) · [Mesh generator](../models/articulated/generate.py)

## How the motion was made

The robots meet with open palms, curl their fingers and thumbs around each other's hands, and pull up together. Once they release, they close their hands into fists and return for a knuckle bump. Drake solves the arm joints for the desired wrist poses. The finger controller tests the clearance to the other hand as it closes each joint. The resulting motion is a kinematic demonstration with fixed choreography. The learned depth policy and its 43/150 success count belong to the rigid-palm study. [First-contact study](setup.md)

The timing of the open-hand approach and release uses a human reference. In the public Inter-X handshake preview, we tracked the horizontal gap between the two colored silhouettes in a band around the hands. At 30 frames/s, frames 0–26 show the gap closing and frames 91–108 show it opening. We normalized each displacement to its total travel and fitted a blend of two smooth minimum-jerk movements. The root-mean-square fit errors were 0.026 and 0.020 of the observed travel, respectively. Those errors describe the fit to the video. The controller uses the curves to advance along its own wrist paths and stretches individual intervals to respect robot joint speed limits. [Inter-X project](https://liangxuy.github.io/inter-x/) · [Reference clip](https://liangxuy.github.io/inter-x/static/videos/handshake.mp4) · [Fitted curves](../dap/hand_timing.py)

Cabibihan and colleagues measured approach, contact, and return as distinct phases of human handshaking. We used that structure for the clasp. The follow-up knuckle bump is an authored choice drawn from documented variations on the dap. [Human handshake study](https://doi.org/10.3389/frobt.2022.758519) · [Smithsonian Center for Folklife and Cultural Heritage](https://folklife.si.edu/talkstory/2014/five-on-the-black-hand-sideorigins-and-evolutions-of-the-dap)

NVIDIA's GR00T research uses first-person footage of people performing manipulation tasks, and NVIDIA publishes a 100,000-hour subset. We chose Inter-X for this demo because its greeting clip shows both participants' hands in the same view. [NVIDIA GR00T](https://research.nvidia.com/labs/gear/gr00t-2/) · [NVIDIA Human-1M-100K dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Human-1M-100K)

## What meets during the dap

The 4.88-second replay uses 14 arm joints and 30 hand joints. Distances below are between Drake collision surfaces. A positive gap is clearance. The model represents a touch within 0.2 mm and does not simulate skin compression or grip force. [Motion and geometry audit](../dap/handshake_audit.py)

| Time | Event | Measurement |
| --- | --- | --- |
| 1.34 s | Open palms meet | 0.017 mm between palms |
| 1.66 s | Both hands clasp | Each index fingertip and each thumb tip is within 0.55 mm of the opposite palm |
| 1.90 s | Shared pull peaks | The clasp has risen 47 mm and moved 38 mm toward the first arm |
| 3.69 s | Hands have separated | 122 mm between the nearest hand surfaces |
| 4.09 s | Knuckle bump | 0.030 mm between the two middle-finger knuckles |
| 4.88 s | Hands withdraw | 145 mm of clearance |

The audit checked 1,234 poses, sampling every 5 ms and at each trajectory knot. All tested collision pairs had positive clearance, and the peak joint speed reached 98.5% of its model limit.

## Reproduce

```bash
make setup
make dap
.venv/bin/python -m dap.handshake_audit
```

To refit the timing curves from the public reference clip:

```bash
.venv/bin/python -m pip install scipy
curl -fsSL https://liangxuy.github.io/inter-x/static/videos/handshake.mp4 -o /tmp/inter-x-handshake.mp4
.venv/bin/python scripts/fit_hand_timing.py /tmp/inter-x-handshake.mp4
```

The fit script checks the source video's SHA-256 and prints the two sets of coefficients and their errors. Rendering uses the fitted constants and works offline once the simulation dependencies and arm assets are installed.
