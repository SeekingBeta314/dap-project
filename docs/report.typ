#set page(paper: "us-letter", margin: (x: 0.83in, y: 0.78in),
  footer: context [#set text(size: 8pt, fill: gray)
    MIT 6.4210 · Robotic Manipulation #h(1fr) Dap study · #counter(page).display()])
#set text(font: "Libertinus Serif", size: 11pt)
#set par(leading: .62em)
#set heading(numbering: none)
#let ink = rgb("#174c48")
#show heading: it => text(fill: ink, it)
#let summary = csv("results/matched_summary.csv", row-type: dictionary)
#let row(name) = summary.find(r => r.planner == name)
#let pretty = (pixels: "Depth and arm state", proprio: "Arm state only", replan: "Replan",
                once: "Plan once", oracle: "Oracle", hold: "Hold still")
#let names = ("pixels", "proprio", "replan", "once", "oracle", "hold")
#let rate(x) = str(calc.round(100*float(x), digits: 1))
#let num(r, field) = str(calc.round(float(r.at(field))*int(r.n)))
#let header(..items) = items.pos().map(x => text(weight: "bold", x))
#let tablefmt = (stroke: (x: none, y: .45pt + luma(210)), inset: (x: 5pt, y: 4pt))

#text(fill: ink, size: 10pt, weight: "bold")[MIT 6.4210  /  ROBOTIC MANIPULATION]
#v(2pt)
#text(size: 26pt, weight: "bold")[Learning to meet a moving palm]
#v(1pt)
#text(size: 13pt)[Depth-guided dapping with two simulated KUKA iiwa arms]
#v(3pt)
A dap requires two hands to meet face to face with their fingers pointing in opposite directions. The partner follows a different reach in each trial. Our arm receives a delayed, noisy depth image and its own joint state. Can it use that image to reach an aligned first contact? A second experiment asks how much a classical controller gains from revising its plan as it observes the partner.

*On 150 held-out trials, the depth policy achieved 43 aligned touches (28.7%). The matched arm-state-only policy achieved one (0.7%).* The single blind-policy success also occurred with depth, leaving 42 discordant trials, all favoring vision (Holm-adjusted exact McNemar $p < 10^(-10)$). All 43 depth successes occurred before the partner stopped. Eighteen landed within 100 ms of the intended meeting time and 31 within 200 ms.

Collisions ended 86 depth-policy trials. Twenty other trials ended in misaligned contact and one timed out. The classical replanner had 33 successes and 61 collisions on the same seeds. Its success-rate difference from the depth policy was inconclusive (29 depth-only wins, 19 replanner-only, unadjusted $p = 0.193$).

#figure(image("outcomes.png", width: 90%), caption: [First-event outcomes on identical held-out scenarios at nominal sensor noise. An aligned touch requires a full relative-rotation error of at most 15°.])

#text(size: 10pt)[
#table(columns: (1.9fr, .9fr, 1fr, 1.35fr, 1fr), ..tablefmt,
  ..header("Method", "Trials", "Successes", "95% rate CI", "Collisions"),
  ..names.map(n => {let r = row(n); (pretty.at(n), r.n, num(r, "success"),
     rate(r.success_lo) + "–" + rate(r.success_hi) + "%", num(r, "collision"))}).flatten())
]

#pagebreak()
= What counts as a dap?
Each hand is a rigid palm with fixed fingers. Drake computes the signed distances between the collision geometries at 10 ms intervals. A valid touch comes from the front, puts the palm boxes within 2 mm, and leaves the face centers within 5 cm in the palm plane. The first contact or collision ends the trial. Valid contact takes precedence when both events first appear in the same sample. A trial that runs one second past the end of the reach without an event times out. The scorer reads only ground-truth geometry.

The target rotation reverses both the palm normal and the finger direction. The 15° engineering tolerance applies to the full relative rotation. A touch can therefore have parallel palm faces and still fail because the fingers point the wrong way. Position, timing, and relative palm speed are measured at contact. The intended meeting time is the instant at which the partner reaches its sampled meeting pose. The binary rule permits earlier and later aligned touches.

== Comparing controllers
The partner follows a smooth reaching trajectory through an inverse-kinematics-certified meeting pose, continues past it, and then holds still. The sampler varies base placement, both initial arm configurations, meeting pose, and target peak speed (0.15–0.9 m/s). Both arms obey joint position and velocity bounds. The partner avoids contacting our resting arm by construction.

The learned policy fits examples of our goal configuration labeled by simulator inverse kinematics. The deployed network receives the latest noisy 212 × 120 depth frame, the arm's current and initial joint angles, and elapsed trial time. The selected model takes the closest depth return in each 2 × 2 block of a central crop. Three convolutional layers encode the resulting 56 × 68 image. Two output heads predict different wrist configurations. A temporally smoothed gate selects one to avoid averaging opposing finger orientations. A velocity servo drives the seven joints toward that goal at 20 Hz. The arm-state-only network uses the same training scenes, labels, two-head output, and loss. The loss includes an estimate of how joint-angle errors change palm position and orientation. The two checkpoints share the same output representation and action limits.

The classical controllers get 100 Hz measurements of partner palm position and normal. The nominal sensor has 5 mm position noise and 2° angular noise. A Kalman filter estimates motion. The controller predicts a meeting, estimates the unobserved finger direction from the known geometry, and solves for a reachable, aligned pose. The once controller plans at the start of the trial. The replanning controller can revise its trajectory at 10 Hz. The oracle receives the true meeting time and target joint angles and follows a joint-space minimum-jerk trajectory. Its intermediate path can collide, and it can reach its speed limit.

== Paired evaluation
The selected model used 4,096 labeled training-distribution scenarios (20 snapshots per scenario). A scenario-level split reserved 820 for epoch selection. Three candidate models were compared on 32 separate closed-loop development scenarios. The convolutional model made eight aligned contacts and each coarse-input candidate made three. The final comparisons use separate experiment seeds, with each controller replaying the same scenarios. The analysis records success rates with 95% Wilson intervals, applies exact paired McNemar tests to binary outcomes, and uses seed-level paired bootstrap intervals for mean loss differences. In total, the six sweeps logged 2,100 condition–seed trials with zero exceptions. The noise and speed sweeps share seeds. Holm adjustment covers the tests in each reported comparison family. The loss ranks an aligned touch above a misaligned touch, a timeout, and a collision, in that order. Its within-tier scales and weights are engineering choices.

#pagebreak()
= Replanning, speed, and timing
== One plan or repeated updates?
On 150 paired trials per noise level, replanning produced more aligned first contacts at all three pose-noise levels. At nominal noise, 19 seeds succeeded only with replanning and three only with one-shot planning. Replanning reduced the mean tiered loss by 2.57 points (paired bootstrap 95% interval 1.44–3.74). Because the one-shot controller acts on the first observation, it lacks a history from which to estimate velocity. The comparison measures the benefit of later observations and revised trajectories together.

#table(columns: (1.4fr, 1fr, 1fr, 1.4fr), stroke: (x: none, y: .45pt + luma(210)),
  [Pose noise], [Once], [Replan], [Adjusted $p$],
  [Half nominal], [17/150], [33/150], [0.0044],
  [Nominal], [17/150], [33/150], [0.0077],
  [Double nominal], [17/150], [32/150], [0.0104])
== Partner speed
The replanner succeeded in 77 of 300 trials (25.7%). From the slowest to fastest realized-speed bins, successes were 15/46, 21/47, 22/64, 14/68, and 5/75. The fastest bin had the lowest success rate. At the two extremes, the fraction of scenarios passing the benchmark's reach-in-time check fell from 44/46 to 17/75. This check uses one candidate meeting configuration and a conservative speed margin. Other paths can still reach contact. Actual near-limit joint speeds were rare: in the fastest bin, at least one joint reached 95% of its speed bound during an average of 0.21% of executed steps. The data do not establish sustained joint saturation as the cause of the decline.

#figure(image("results/speed_limits/figures/by_speed.png", width: 90%), caption: [Replanning at nominal pose noise. Points show results by realized partner peak speed. Error bars give 95% Wilson intervals for the success rate.])

== Timing sensitivity
The original rule counts aligned contact anywhere along the reach. Requiring it within 100 ms of the intended meeting time gave 12/150 successes for each classical controller at nominal noise. At 200 ms, replanning yielded 25/150 and planning once yielded 17/150. The larger gain on the original endpoint includes contacts outside a narrow timing window.


#pagebreak()
= Interpretation and scope
A noisy 100 Hz pose observation and a delayed 30 Hz camera image expose different information about the partner. A planner–policy comparison therefore measures the complete sensing and control systems. The arm-state comparison measures the difference between separately fitted systems. The vision policy waits for its first delivered image and the arm-state policy can move immediately. The classical noise multiplier applies only to pose measurements. The speed analysis is observational because the sampler retains arm geometries that are executable at the drawn speed.

The simulator advances a fixed amount of simulated time per control step regardless of solver runtime. A 10 Hz replanning schedule is therefore a simulated control rate. Across the 300 nominal-noise replanning trials, 156 had at least one controller call longer than 100 ms on the measured hardware.

The simulator models rigid first contact and omits forces, compliance, human adaptation, and post-contact clasping. Geometry sampling and simulated camera noise also limit transfer. The study measures generalization across the nominal simulated scenario distribution and a single fitted checkpoint per learned method.

== Two trial replays
In held-out seed 0, the depth policy met the moving partner at 0.72 s, 55 ms before the intended meeting, with 1.8° rotation error. Seed 2 was the first collision by seed order. Both animations in the repository show an external side view beside the depth image available at that instant. #link("https://github.com/SeekingBeta314/dap-project/blob/finish/dap-study/docs/success.gif")[Successful touch.] #link("https://github.com/SeekingBeta314/dap-project/blob/finish/dap-study/docs/collision.gif")[Collision.]

#figure(image("first_contact.png", width: 86%), caption: [Seed 0 at first contact. The left panel shows both arms. The right panel shows the latest depth image delivered to the policy. White pixels have no depth return.])

== Reproducibility
The repository includes the fitted weights, a demonstration replay, experiment commands, raw per-trial records, and code for reconstructing the paired tables. The full setup document lists parameter values and their sources. Build this PDF from the repository with `typst compile docs/report.typ`. Josh Nix, Oliver Wang, and Samuel Zhou are the named project team. The completion work, experiment analysis, and this report used AI assistance. #link("https://github.com/SeekingBeta314/dap-project")[Project repository.]

== Related work
Kim, Shukla, and Billard modeled motion prediction and reactive interception in _Catching Objects in Flight_ (IEEE Transactions on Robotics, 2014). #link("https://doi.org/10.1109/TRO.2014.2316022")[Paper.] Fitter and Kuchenbecker demonstrated human-inspired hand clapping with Baxter (IEEE RO-MAN, 2016). #link("https://osusharelab.com/publications/")[Author publication list.] The present study uses first palm touch as a measurable endpoint and reports orientation and timing separately.
