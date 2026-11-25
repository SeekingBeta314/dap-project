# The cupped-hand dap

[Watch the replay](hand_dap.gif) · [Download the MP4](hand_dap.mp4)

The two cupped hands start about 66 cm apart. The striking arm draws back, then sweeps outward and upward before rolling the palm into the tap. The receiving arm takes a shorter arc to meet it. Both palms keep moving together briefly at contact. The arms then peel away along a lower curve and settle back into their starting poses. The fixed camera shows both elbow paths, and the second angle follows the palms.

## Human reference

In OMFGrant's *Contemporary Dap Up*, the striking hand rises early, then swings across into the other palm. We marked both palm centers in frames 20–25 of the 25 fps recording. The striker's hand gets about 17% of its start-to-contact distance above a straight line between those points. We fitted a cubic curve to that side-view arc and a second curve to the horizontal closing motion. The replay scales those shapes to the robots' reach. [Original film by OMFGrant, CC0](https://commons.wikimedia.org/wiki/File:Contemporary_Dap_Up.gif) · [Frame annotations and fit](../scripts/fit_hand_timing.py)

The observed segment spans 0.20 s. The simulated inward swing takes 0.93 s over the longer reach. Over the six source frames, the timing fit has a root-mean-square error of 3.8% of the horizontal travel, and the arc fit has an error of 0.56% of the projected start-to-contact distance. The robot's out-of-plane motion, wrist rotation, and recovery are authored for the iiwa.

## What touches

Each simulated hand has 15 finger and thumb joints held at constant angles. The palm mesh arches across the knuckles and has raised pads on either side of its center. Twelve collision patches follow that surface. Using several patches lets Drake represent the hollow between the pads. [Hand generator](../models/articulated/generate.py) · [Motion](../dap/handshake.py)

| Measurement | Result |
| --- | --- |
| Opening gap between the nearest hand surfaces | 65.5 cm, up from 17.5 cm in the previous replay |
| Inward palm travel after the wind-up | 56.8 cm for the striker and 41.7 cm for the receiver |
| Furthest distance from a straight inward path | 10.8 cm for the striker and 7.4 cm for the receiver |
| Finger and thumb motion over the 2.26 s replay | 0° across all 30 joints |
| Contact interval | One encounter, observed from 1.187 to 1.204 s |
| Span between two outer palm contacts at impact | 45.8 mm, with surface gaps of 0.011 and 0.012 mm |
| Center of the pocket at impact | 11.7 mm clearance to each hand |
| Closest distance involving a finger or thumb on opposite hands | 6.1 mm |
| Shared follow-through | The midpoint between the palms travels 3.8 mm during the contact phase |
| Palm approach and rebound speeds | 0.75 m/s closing and 0.09 m/s separating, averaged over the 40 ms before and after impact |

The audit checks every millisecond and every trajectory knot (2,419 poses). It counts surfaces within 0.2 mm as touching. The starting and ending joint angles match, so the GIF can loop without an elbow jump. The original depth-policy results use the separate first-contact benchmark. [Contact audit](../dap/handshake_audit.py) · [Learning study](setup.md)

## Reproduce

```bash
make setup
make dap
.venv/bin/python -m dap.handshake_audit
.venv/bin/python scripts/fit_hand_timing.py
```

The replay files are written to `results/hand_dap/`. To see where each point was placed, download the original GIF from the film's Wikimedia Commons page and run the following command.

```bash
.venv/bin/python scripts/fit_hand_timing.py Contemporary_Dap_Up.gif --overlay /tmp/dap-reference-points.png
```
