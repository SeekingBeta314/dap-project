# Closed-loop replay on a second CPU host

We repeated all 344 frozen-student trials on one additional compute node after closing the prespecified analysis. **Each seed retained its original aligned-contact success or failure classification.** One wide failure changed from timeout to collision. Continuous quantities could differ substantially even when the binary result was identical.

| Preset | Original aligned contacts | Repeat aligned contacts | Binary classifications changed | First-event classifications changed |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 114/172 | 114/172 | 0 | 0 |
| Wide | 93/172 | 93/172 | 0 | 1 |

The first-event classes are aligned contact, misaligned contact, collision and timeout. Wide seed 1115 originally timed out. The repeat collided at 3.31 s. Nominal seed 1009 produced aligned contact in both runs, with 5.116° original rotation error and 6.221° on the alternate host. Nominal seed 1007 produced misaligned contact in both, with 104.52° error at 2.07 s originally and 147.21° at 1.76 s in the repeat. [Per-seed measurements](per_seed.csv) · [Transition counts](transitions.csv)

Among the 132 nominal and 117 wide trials that made contact in both runs, median absolute rotation-error differences were 0.000044° and 0.000045°. The 95th percentiles were 0.1095° and 0.0537°. The largest differences were 42.70° and 0.78°. For trials with a contact or collision in both runs, median absolute event-time differences were zero on both presets. These results characterize this checkpoint on these machines. Small numerical differences can accumulate as each policy action changes future camera images. Isolating the responsible library or hardware instruction would require additional controlled runs. [Summary statistics](summary.csv)

## Machines and integrity checks

| Run | Host | Slurm array element | CPU |
| --- | --- | --- | --- |
| Original nominal | node2024 | 30461508_0 | AMD EPYC 7513 32-Core Processor |
| Original wide | node2501 | 30461508_1 | x86_64, 2 sockets and 60 cores per socket in Slurm, exact model unavailable in the original log |
| Nominal repeat | node1623 | 30461908_0 | AMD EPYC 9654 96-Core Processor |
| Wide repeat | node1623 | 30461908_1 | AMD EPYC 9654 96-Core Processor |

The original jobs used eight concurrent independent trial workers. The repeat used four workers on node1623. Each process used one numerical-library thread and CPU inference. The recorded simulator and student source hashes, checkpoint, Python package versions, camera configuration, evaluation options and scenario-archive hash agree. Each repeat used the same scenario digest as its official trial, and all eleven scenario covariates matched within the study audit tolerance. All 344 repeats completed. The student SHA-256 was `acc1952951026802b488b52c9391b8c0327ada9e912de9c46a27857f24af1d62`. The primary study retains its original 2,752 trials and paired results. [Original study](../README.md) · [Nominal repeat manifest](nominal_manifest.json) · [Wide repeat manifest](wide_manifest.json)

## Reproduce the comparison

The [nominal](nominal_repeat.jsonl.gz) and [wide](wide_repeat.jsonl.gz) repeat files contain the full trial records. Run the following command from the repository root to rebuild the CSVs and check the saved run identities.

```bash
.venv/bin/python scripts/compare_perception_reproduction.py docs/perception_results/portability \
    --out results/portability_rebuilt
```

To run a new repeat on the same alternative cluster node, request an allocation with four CPU cores and use the published checkpoints and canonical trajectories. The two presets may run as separate jobs.

```bash
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
for preset in nominal wide; do
    srun -p mit_preemptable,mit_normal --nodes=1 --ntasks=1 --nodelist=node1623 \
        --cpus-per-task=4 --mem=10G --time=00:06:00 \
        .venv/bin/python scripts/run_frozen_student.py "$preset" \
        --checkpoint models/policies/student.pt --workers 4 --out results/student_host_repeat
done
.venv/bin/python scripts/compare_perception_reproduction.py results/student_host_repeat \
    --out results/student_host_tables
```
