# The cupped-hand dap

[Watch the replay](hand_dap.gif) · [Download the MP4](hand_dap.mp4)

Both robots start with a thumb raised, the four fingers gently bent together, and a hollow in the palm. The striking arm draws back slightly, then sweeps up and inward. The receiving arm makes a shorter movement to meet it. The palm pads meet once and the arms rebound. The hand shape stays fixed. The profile view shows the curve of the fingers throughout the swing.

## Human reference

OMFGrant's *Contemporary Dap Up* shows two people turning toward each other and swinging their right hands together. We used the opening strike, from frames 20 through 25 of the 25 fps recording. We marked the approximate palm centers and measured how their horizontal separation changed. The normalized swing in the robot replay follows a cubic fit to those six observations. It begins from rest and still has inward speed when the palms meet. The fit error is 3.8% of the observed horizontal travel (root mean square). The reference segment lasts 0.20 seconds. The simulated arms take 0.41 seconds for the inward swing. Their wrist paths, cupped pose, and rebound are authored for the iiwa. [Original film by OMFGrant, CC0](https://commons.wikimedia.org/wiki/File:Contemporary_Dap_Up.gif) · [Frame annotations and fit](../scripts/fit_hand_timing.py)

## What touches

Each simulated hand has 15 finger and thumb joints held at constant angles. The palm mesh arches across the knuckles and has raised pads on either side of its center. Twelve collision patches follow that surface. Using several patches lets Drake represent the hollow between the pads. [Hand generator](../models/articulated/generate.py) · [Motion](../dap/handshake.py)

| Measurement | Result |
| --- | --- |
| Finger and thumb motion over the 1.12 s replay | 0° across all 30 joints |
| Contact interval | One encounter, from 0.590 to 0.610 s |
| Span between two outer palm contacts at impact | 45.8 mm, with surface gaps of 0.011 and 0.012 mm |
| Center of the pocket at impact | 11.7 mm clearance to each hand |
| Closest distance involving a finger or thumb on opposite hands | 6.1 mm |
| Palm approach and rebound speeds | 0.58 m/s closing and 0.33 m/s separating, averaged over 40 ms on each side of contact |

The audit checks every millisecond and every trajectory knot (1,139 poses). It counts surfaces within 0.2 mm as touching. These are geometric measurements from the kinematic replay. The original depth-policy results use the separate first-contact benchmark. [Contact audit](../dap/handshake_audit.py) · [Learning study](setup.md)

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
