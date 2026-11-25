# Held-out scenario archive

Each compressed JSON Lines file contains the 172 reserved seeds for one preset. Each line holds the SHA-256 digest and a JSON payload containing the seed identity and the simulator inputs. The payload includes the actual IK solution, joint-trajectory knots, arm starting pose and layout. Sensor and camera noise use the original per-seed streams. The archives are used only for evaluation and replay. The separate training bank is [documented here](../training/teacher/README.md).

During a preliminary six-controller run, an audit of ground-truth inputs found that re-solving IK for the same seed on different cluster nodes could yield different trajectories. Six wide seeds (1005, 1015, 1051, 1077, 1139 and 1156) differed beyond the archive audit tolerance. Three produced different layouts (1051, 1077 and 1156). The team discarded the complete preliminary run from the comparison, solved all 344 reserved scenarios once, and reran every method from the stored trajectories. This correction preceded opening the student's evaluation outcomes. The preregistered seeds, sample size, model selection and endpoints were preserved. Every published row records its scenario digest.

To reproduce the experiments, use the files alongside this note. To generate an independent archive in a separate location, use the commands below. A fresh solve can differ across machines and includes new timing metadata. The shipped files specify the exact inputs for the published measurements.

```bash
.venv/bin/python scripts/freeze_perception_scenarios.py nominal --workers 8 --out results/new_scenarios
.venv/bin/python scripts/freeze_perception_scenarios.py wide --workers 8 --out results/new_scenarios
```
