# Report figures and video footage

The study asks whether a robot can infer a partner's moving palm from delayed depth images accurately enough to meet it with the fingers pointing in opposite directions. The learned student and classical tracker operate in separate closed-loop replays of the same saved scenario. The student sees depth and its own arm state. The classical tracker also receives the static layout, which it uses to mask known geometry and locate the partner's shoulder. The privileged teacher supplies training labels and serves as a reference in the evaluation. It has an initial pose estimate before the first camera delivery, while the student holds until an image arrives. The teacher–student gap includes that startup difference. The [study results](../perception_results/README.md) and [protocol recorded before evaluation](../perception_protocol.md) define the comparisons.

## Paper

One possible title is *Meeting a Moving Palm from Depth: Learned and Geometric Controllers for a Robotic Dap*. In the abstract, introduce the difficulty of seeing a moving hand near contact, describe the two depth controllers and the matched simulation, then give the nominal and wide results. Close with what the change across layouts implies for using this method on unfamiliar reaches.

Use the IEEE conference template for the final manuscript. Introduce each numbered figure or table in the preceding paragraph and give it a caption that makes its main observation clear. The following assets have vector versions for the paper and raster versions for video.

| Design question | Evidence |
| --- | --- |
| What does the robot learn, and what can it observe at deployment? | [Methods figure, PDF](methods.pdf) · [PNG](methods.png). The upper row shows the privileged supervision available during training. The two lower rows identify the information and control rate of each deployed system. |
| Does learned perception increase aligned first contacts, on both familiar and wider reaches? | [Success rates and Wilson intervals](../perception_results/figures/success.pdf) · [Paired comparisons](../perception_results/paired.csv). Use the student-only and classical-only discordant pairs for the hypothesis test. |
| How do the trials end, and how does realized partner speed relate to performance? | [First-event breakdown](../perception_results/figures/outcomes.pdf) · [Results by speed](../perception_results/figures/speed.pdf) · [Timing sensitivity](../perception_results/timing.csv). The earliest valid touch ends the benchmark. Scoring allows aligned touches after the partner stops. |
| What did the network see during its final approach? | [Palm-estimation errors](../perception_results/diagnostics/pose_errors.pdf) · [Measurement definitions and trial-level data](../perception_results/diagnostics/README.md). The replay compares the estimate used for the action with the true palm at the corresponding camera timestamp. |

The results archive also includes a [LaTeX table](../perception_results/paired_table.tex) with the two primary comparisons, the actual success counts, and corrected p-values. To use these assets in Overleaf, upload `methods.pdf`, `paired_table.tex` and `references.bib` alongside the manuscript and load `graphicx` and `url` (or `hyperref`). This figure fits across both IEEE columns. Place each environment after its first reference in the text.

```latex
Figure~\ref{fig:perception-methods} shows where each controller obtains its palm estimate.
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{methods.pdf}
  \caption{Depth-based controllers for an aligned first contact. The student learns from
    a privileged teacher. During deployment it infers the partner from delayed depth
    and its own arm state. The geometric tracker also uses the known static layout.}
  \label{fig:perception-methods}
\end{figure*}
Table~\ref{tab:perception-paired} compares aligned touches on the same reaches.
\input{paired_table.tex}
```

**Motivation and design goals.** A moving contact target can disappear behind the robot's own hand just as accuracy matters most. This benchmark makes that perception problem measurable. Define an aligned touch as a front-side palm contact with at most 15° of full relative-rotation error. Report the comparison on nominal reaches, the change under wider layouts and possible mid-course switches, and the gap to controllers given accurate pose. The 20-percentage-point improvement in the power calculation is the design's target effect size.

**Findings to report.** On nominal reaches the student aligned 114/172 first touches (66.3%) and the tracker aligned 21/172 (12.2%). Only the student succeeded on 97 reaches, and only the tracker succeeded on 4. The paired improvement was 54.1 percentage points (95% interval 45.9–62.2). On wider reaches the counts were 93/172 and 17/172, with 81 student-only and 5 tracker-only successes. The improvement was 44.2 points (36.0–52.3). Both Holm-adjusted tests gave *p* < 0.001, and both gains exceeded the 20-point effect targeted by the study design. The privileged teacher still achieved 163/172 and 145/172. A single fit of each controller was tested, so the intervals describe variation across reaches. [Paired results](../perception_results/paired.csv)

**What an aligned touch means.** Of the student's 114 nominal successes, 86 occurred before the partner finished moving. On wide reaches, 66 of 93 did. Requiring an aligned contact within ±100 ms of the intended meeting left only 22/172 nominal and 15/172 wide successes. This distinction matters for a greeting where both people expect the tap on the same beat. A next experiment could make a timing window part of the training goal and compare the gain in on-time touches against any increase in collisions. [Motion at contact](../perception_results/motion_at_success.csv) · [Timing sensitivity](../perception_results/timing.csv)

**Why some reaches failed.** In the final 200 ms, the median partner-orientation estimation error on nominal reaches was 6.5° when the student achieved an aligned touch and 21.6° when it failed. Wide reaches showed a similar separation, 7.2° versus 21.9°. Misjudging the partner's orientation is one plausible contributor to failure. The grouping also includes collisions and timeouts, so it cannot identify a single cause for each miss. [Pose-error figure and measurements](../perception_results/diagnostics/)

**Design process.** On the same 64 nominal development reaches, the first student trained against the final teacher achieved 4 aligned contacts. Two rounds of collecting the learner's own trajectories raised this to 16 and then 22. Inspection showed that the learned finger direction could reverse, even when the estimated center was useful. The next architecture explicitly located the palm in the image, examined the surrounding pixels and learned which end of the finger axis was forward. Its first fit achieved 39 aligned contacts on the same reaches, increasing to 47 after collecting trajectories with this model. On the fresh 64-reach confirmation, the spatial network achieved 47 aligned touches to the earlier model’s 24. The spatial network was then frozen. Use this sequence to explain how observing a failure changed the design. The changes were tested together, so the result measures their combined effect. [Student design-process figure](../perception_results/training/student/validation_stages.png) · [Training and checkpoint selection](../perception_results/training/student/)

**Related work and trade-offs.** Fitter and Kuchenbecker give an example of robot hand-clapping inspired by human motion. Here a recorded greeting supplies the approach arc for the cupped-hand illustration. Pose-based visual servoing makes it possible to inspect what the geometric tracker accepts and to replace the inferred pose with accurate pose as a diagnostic. Teacher–student training allows accurate simulator state to supervise a controller that must infer the hand from depth at deployment. DAgger supplies labels on states visited by the student's own actions. Comparing the same policies on wider reaches tests sensitivity to changes in the scenario distribution, a concern also present in robotic catching. [Clapping](https://doi.org/10.1109/ROMAN.2016.7745097) · [Visual servoing](https://doi.org/10.1109/MRA.2007.339609) · [Privileged supervision](https://proceedings.mlr.press/v100/chen20a.html) · [DAgger](https://proceedings.mlr.press/v15/ross11a.html) · [Agile catching](https://proceedings.mlr.press/v211/abeyruwan23a.html) · [BibTeX](references.bib)

## Footage and spoken explanation

| Footage | What to point out in the voice-over |
| --- | --- |
| [Nominal success](nominal_success.mp4) · [terminal still](nominal_success.png) | Seed 1009. The student touches with 5.1° of rotation error. The classical controller collides. Use the circle and cross in the depth views to explain which measurements are available to the viewer and controller. |
| [Nominal failure](nominal_failure.mp4) · [terminal still](nominal_failure.png) | Seed 1000. Both trials end in collision, the student at 0.99 s and the classical controller at 1.03 s. Include this example when explaining the remaining 34 nominal student collisions. |
| [Wide target switch](wide_switch.mp4) · [terminal still](wide_switch.png) | Seed 1006. Watch the target-shift annotation. Both controllers touch. The student reaches 12.7° of rotation error and the tracker reaches 28.5°. The 15° alignment threshold explains the different scores. |

1. **Show the gesture.** The [human reference clip](human_reference.mp4) plays at the original 25 fps. Yellow circles mark the six frames used to fit the approach arc. Follow it with the [robot cupped-hand replay](../hand_dap.mp4) and identify it on screen as an authored motion illustration. Its hands start 65.5 cm apart. The fingers stay fixed while the arms meet and separate. The robot uses the approach portion of the greeting. [Motion and contact audit](../hand_dap.md)
2. **Show what the controller sees.** Use the methods figure and a paired replay. Point out the depth camera's view when the arm moves in front of the partner. Identify the left or right example, its outcome and the label on each video panel. In both side views the arm on the left is controlled and the arm on the right executes the recorded partner trajectory. The scored experiment uses rigid palms.
3. **State the measured result.** Put the two matched success counts and their denominator on screen. Explain what happened on wider reaches and show the paired uncertainty. Use one failure alongside any success example, then return to the complete outcome chart.
4. **Explain a design decision.** Use a recorded comparison from the training history. For instance, the initial from-scratch teacher made zero aligned touches on 100 validation scenarios. The team tested imitation of an explicit servo controller, then collected labels where the learned policy actually went. In the new run the frozen teacher achieved 182 aligned touches on 200 independent validation scenarios, compared with 164 using the servo supplied with accurate pose at camera timing. Explain what changed in the final training run, what remained difficult, and why that matters for oriented contact. [Initial training history](../teacher_design.md) · [Teacher design-process figure](../perception_results/training/teacher/training.pdf) · [Final training evidence](../perception_results/training/teacher/)
5. **Close with scope and sources.** Ground-truth geometry measures first contact in simulation. Testing compliance, impact forces or human adaptation would require a separate experiment. The [1920 × 1080 References slide](references_slide.png) lists the papers and human film, and its [PDF](references_slide.pdf) contains clickable source links. The film already carries an on-screen credit. Add a title-page acknowledgment describing the AI assistance actually used for the project and video, and a corresponding statement in the paper. For this branch, Codex assisted with implementation, training and evaluation, diagnostic plots, and annotated simulation footage. Update that description to reflect the team's final writing and editing process.

The robot clips and figures are project-generated. The human film is *Contemporary Dap Up*, OMFGrant, 2025, released under CC0. Use the credited export if showing this footage. [Original film and license](https://commons.wikimedia.org/wiki/File:Contemporary_Dap_Up.gif)

## Rebuild the methods figure and human b-roll

From the repository root, in the project environment:

```bash
python scripts/plot_perception_methods.py
python scripts/fit_hand_timing.py /path/to/Contemporary_Dap_Up.gif --video docs/presentation_assets/human_reference.mp4
```

The reference exporter checks the source GIF's SHA-256 against the version used for the fit.

## Paired replay selection

The examples use the first nominal student success and first nominal student failure in ascending seed order, and the first wide scenario containing a mid-course switch. Each clip has the same camera angle on both arms, the most recent depth image each controller actually used, and the terminal outcome. A circle marks the true palm center at image capture for the viewer. A cross marks the student estimate or a fresh classical tracker update. The markers are drawn after inference. Playback is real time. A completed trial freezes at its own endpoint while the other panel continues. The final second holds both endpoints for reading. The renderer checks both outcomes and contact measurements against the archived records before writing video.

CPU-dependent numerical differences can grow over a closed-loop rollout. Repeating all 344 student trials on another host reproduced every aligned-contact classification, although one wide timeout became a collision. The strict clip exporter can flag a change in contact angle even when the success label agrees. The released footage was checked against the original metrics. [Cross-machine reproduction](../perception_results/portability/)

```bash
python scripts/render_perception_pair.py --preset nominal --case success --rows docs/perception_results/nominal/trials.jsonl.gz --out docs/presentation_assets/nominal_success.mp4
python scripts/render_perception_pair.py --preset nominal --case failure --rows docs/perception_results/nominal/trials.jsonl.gz --out docs/presentation_assets/nominal_failure.mp4
python scripts/render_perception_pair.py --preset wide --case target-switch --rows docs/perception_results/wide/trials.jsonl.gz --out docs/presentation_assets/wide_switch.mp4
```
