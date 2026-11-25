# Learned (RL / IL) approaches to dynamic interception with robot arms: catching, hitting, meeting moving targets

Scope note: notes for the dap project (KUKA iiwa7, kinematic Drake sim, joint-velocity actions at 20 Hz, meeting a moving palm moving at 0.15–0.9 m/s, head-mounted D435-like depth at 30 fps with 1-frame latency; plan is a PPO privileged-state teacher, DAgger depth student, and comparisons against a diff-IK servo and a blind baseline). Citations were checked against arXiv abstract pages, publisher or institutional listings, or the paper PDF/HTML. Anything I could not check against a primary listing is marked UNVERIFIED. Every number is quoted from the source named next to it.

### Verified citation list (for the report writer)

| Short name | Citation | Verified against |
|---|---|---|
| Mülling 2013 | K. Mülling, J. Kober, O. Kroemer, J. Peters, "Learning to select and generalize striking movements in robot table tennis," *IJRR* 32(3):263–279, 2013. DOI 10.1177/0278364912472380 | [MPI-IS listing](https://is.mpg.de/publications/mullingkkp2012_2), [Tübingen repository](https://publikationen.uni-tuebingen.de/xmlui/handle/10900/39898?show=full), [author PDF](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Muelling_IJRR_2013.pdf) |
| Kim 2014 | S. Kim, A. Shukla, A. Billard, "Catching Objects in Flight," *IEEE T-RO* 30(5):1049–1065, 2014. DOI 10.1109/TRO.2014.2316022 | [EPFL Infoscience](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1), [researchr](https://researchr.org/publication/KimSB14) |
| Gao 2020 | W. Gao, L. Graesser, K. Choromanski, X. Song, N. Lazic, P. Sanketi, V. Sindhwani, N. Jaitly, "Robotic Table Tennis with Model-Free Reinforcement Learning," arXiv:2003.14398 (2020). Venue: IROS 2020 is UNVERIFIED (the arXiv page lists none). | [arXiv](https://arxiv.org/abs/2003.14398) |
| Büchler 2020/22 | D. Büchler, S. Guist, R. Calandra, V. Berenz, B. Schölkopf, J. Peters, "Learning to Play Table Tennis From Scratch using Muscular Robots," arXiv:2006.05935. The arXiv page says "Submitted to T-RO"; final venue UNVERIFIED. | [arXiv](https://arxiv.org/abs/2006.05935) |
| i-Sim2Real 2022 | S. Abeyruwan, L. Graesser, D. B. D'Ambrosio, A. Singh, A. Shankar, A. Bewley, D. Jain, K. Choromanski, P. R. Sanketi, "i-Sim2Real: Reinforcement Learning of Robotic Policies in Tight Human-Robot Interaction Loops," arXiv:2207.06572 (2022). Venue: CoRL 2022 is UNVERIFIED (the arXiv page lists none). | [arXiv](https://arxiv.org/abs/2207.06572) |
| GoalsEye 2022 | T. Ding, L. Graesser, S. Abeyruwan, D. B. D'Ambrosio, A. Shankar, P. Sermanet, P. R. Sanketi, C. Lynch, "GoalsEye: Learning High Speed Precision Table Tennis on a Physical Robot," arXiv:2210.03662 (2022). Venue UNVERIFIED. | [arXiv](https://arxiv.org/abs/2210.03662) |
| RSS TT system 2023 | D. B. D'Ambrosio et al. (35 authors), "Robotic Table Tennis: A Case Study into a High Speed Learning System," RSS 2023, arXiv:2309.03315, DOI 10.15607/RSS.2023.XIX.006 | [arXiv](https://arxiv.org/abs/2309.03315) |
| Agile Catching 2023 | S. Abeyruwan, A. Bewley, N. M. Boffi, K. Choromanski, D. D'Ambrosio, D. Jain, P. Sanketi, A. Shankar, V. Sindhwani, S. Singh, J.-J. Slotine, S. Tu, "Agile Catching with Whole-Body MPC and Blackbox Policy Learning," L4DC 2023 (per the arXiv comments field), arXiv:2306.08205 | [arXiv](https://arxiv.org/abs/2306.08205), [HTML v2](https://arxiv.org/html/2306.08205v2) |
| Human-level TT 2024 | D. B. D'Ambrosio, S. Abeyruwan, L. Graesser, A. Iscen, H. Ben Amor, A. Bewley, … P. R. Sanketi (27 authors), "Achieving Human Level Competitive Robot Table Tennis," arXiv:2408.03906 (v1 Aug 2024, v3 May 2025). Peer-reviewed venue UNVERIFIED (the arXiv page lists none). | [arXiv](https://arxiv.org/abs/2408.03906), [HTML](https://arxiv.org/html/2408.03906) |
| Catch It! 2024 | Y. Zhang, T. Liang, Z. Chen, Y. Ze, H. Xu, "Catch It! Learning to Catch in Flight with Mobile Dexterous Hands," arXiv:2409.10319 (2024). Venue UNVERIFIED. | [arXiv](https://arxiv.org/abs/2409.10319), [HTML](https://arxiv.org/html/2409.10319) |
| Modular catching 2023 | W. Hu, F. Acero, E. Triantafyllidis, Z. Liu, Z. Li, "Modular Neural Network Policies for Learning In-Flight Object Catching with a Robot Hand-Arm System," IROS 2023, DOI 10.1109/IROS55552.2023.10341463, arXiv:2312.13987 | [arXiv](https://arxiv.org/abs/2312.13987) |
| Dynamic Handover 2023 | B. Huang, Y. Chen, T. Wang, Y. Qin, Y. Yang, N. Atanasov, X. Wang, "Dynamic Handover: Throw and Catch with Bimanual Hands," CoRL 2023, arXiv:2309.05655 | [arXiv](https://arxiv.org/abs/2309.05655) |
| GraspARL 2022 | T. Wu, F. Zhong, Y. Geng, H. Wang, Y. Zhu, Y. Wang, H. Dong, "GraspARL: Dynamic Grasping via Adversarial Reinforcement Learning," arXiv:2203.02119 (2022). Venue UNVERIFIED. | [arXiv](https://arxiv.org/abs/2203.02119) |
| Badminton 2025 | Y. Ma, A. Cramariuc, F. Farshidian, M. Hutter, "Learning Coordinated Badminton Skills for Legged Manipulators," *Science Robotics* 10(102), 2025. DOI 10.1126/scirobotics.adu3922; arXiv:2505.22974 | [arXiv](https://arxiv.org/abs/2505.22974), [HTML](https://arxiv.org/html/2505.22974), [project page](https://sites.google.com/view/legged-racket-sport/home) |
| Goalkeeper 2022/23 | X. Huang et al., "Creating a Dynamic Quadrupedal Robotic Goalkeeper with Reinforcement Learning," arXiv:2210.04435. Full author list and the IROS 2023 venue are UNVERIFIED (seen only in search snippets). | [arXiv](https://arxiv.org/abs/2210.04435v1), [project page](https://xbpeng.github.io/projects/A1Goalkeeping/index.html) |
| Racing OC vs RL 2023 | Y. Song, A. Romero, M. Mueller, V. Koltun, D. Scaramuzza, "Reaching the Limit in Autonomous Racing: Optimal Control versus Reinforcement Learning," *Science Robotics* 2023, DOI 10.1126/scirobotics.adg1462, arXiv:2310.10943 | [arXiv](https://arxiv.org/abs/2310.10943) |
| Perception-aware flight 2023 | Y. Song, K. Shi, R. Penicka, D. Scaramuzza, "Learning Perception-Aware Agile Flight in Cluttered Environments," ICRA 2023 (venue per secondary listings, otherwise UNVERIFIED), arXiv:2210.01841 | [arXiv PDF](https://arxiv.org/pdf/2210.01841) |

---

## Q1. Robot table tennis with learning: methods, sensing, baselines, results

### Takeaway
Table tennis research has moved from imitation-learned movement primitives with a model-based ball predictor (Mülling 2013) to fully sim-trained neural low-level controllers (Google DeepMind, 2020–2024). All of these systems act on a low-dimensional ball state (position, velocity, history) estimated by a separate fast stereo-vision pipeline at 60–125 Hz. None maps pixels to actions. Head-to-head comparisons against a strong model-based controller on the same hardware are rare. Most papers compare learned variants (for example S2R+FT against i-S2R) or compare against humans.

### Cited Findings
**Mülling, Kober, Kroemer & Peters 2013 (IJRR): Mixture of Motor Primitives (MoMP)**
- Robot: Barrett WAM, 7 DoF. Vision: "four Prosilica Gigabit GE640C cameras." The ball is "visually tracked with a sampling rate of 60 Hz" and filtered with an EKF whose model includes gravity and air drag but "neglect[s] the spin." — [Mülling 2013 PDF](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Muelling_IJRR_2013.pdf)
- Method: hitting movements are learned by kinesthetic teach-in as dynamical-system motor primitives (DMPs) with a modified "hitting primitive" formulation. A gating network chooses among primitives and generalizes them, using an augmented state (hitting position and velocity), and the gating weights are improved online by reinforcement. Where, when and how to hit is computed by a model-based predictor. — [Mülling 2013 PDF](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Muelling_IJRR_2013.pdf)
- Simulation: individual primitives had "a success rate of 23 % to 89 %." The untrained mixture reached "67 %," and after learning the weights "over 1000 trials improved the performance to a 94 % success rate." About 27% of primitives were pruned. — [Mülling 2013 PDF](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Muelling_IJRR_2013.pdf)
- Real robot: 25 kinesthetic demonstrations. With an oscillating launcher, return rate was "69%" before training and "97%" after 150 trials, and mean bounce error to the target fell from 1.9 m to 0.6 m. In a fixed hard region the initial policy returned no balls, and after 60 trials it returned "79%." Against a human it returned "74.4 %" at first and "88 %" after one hour of play, with "up to 9 balls in a row." — [Mülling 2013 PDF](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Muelling_IJRR_2013.pdf)
- Baseline note: the demonstrations used in simulation were "sampled from successful strokes of an analytical robot table tennis player (Muelling et al., 2011)," so the learned system was initialized from a model-based player. — [Mülling 2013 PDF](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Muelling_IJRR_2013.pdf)

**Gao et al. 2020: model-free RL table tennis (Google)**
- Evolution-strategies (ES) training of "CNN-based policy architectures for non-visual inputs" with convolution across time. Policies "control robot joints at 100Hz." A curriculum yields both forehand and backhand strokes. Reported "80% return rate on a wide range of ball throws." — [arXiv:2003.14398](https://arxiv.org/abs/2003.14398)

**Büchler et al.: RL from scratch on pneumatic-muscle arms**
- A stochastic model-free policy acts directly on the real system's low-level controls for "thousands of trials." Soft pneumatic muscles provide safety, so no safety constraints are added to the learning algorithm. HYSR (hybrid sim and real) replays recorded ball trajectories in simulation while the real robot moves. The robot returns and smashes balls at "5 m\s and 12m\s on average to a desired landing point." The abstract notes "imprecise state estimation of the flying ball." — [arXiv:2006.05935](https://arxiv.org/abs/2006.05935)

**i-Sim2Real 2022: cooperative human rallies**
- The method alternates between training in simulation and deploying on the real robot, refining a human-behavior model and the policy each iteration. All training uses "Blackbox Gradient Sensing (BGS)," an evolutionary search algorithm. Results: "22 successive hits on average and 150 at best." Against the sim-to-real plus fine-tuning baseline (S2R+FT), for 80% of players, rallies were "70% to 175% longer." — [arXiv:2207.06572](https://arxiv.org/abs/2207.06572)

**GoalsEye 2022: goal-conditioned imitation learning on the real robot**
- Iterative goal-conditioned behavior cloning from undirected demonstrations plus self-supervised goal-reaching practice, with "no reward design or sim-to-real transfer." The policy performs "on par or better than amateur humans" at returning to targets and trains in "just a few hours." The abstract names no algorithmic baseline. — [arXiv:2210.03662](https://arxiv.org/abs/2210.03662)

**RSS 2023 system paper**
- The paper studies "how various sources of latency affect performance," uses "a high-speed, low-latency robot controller" and an optimized perception subsystem, and reports zero-shot sim-to-real policies. It also examines distribution shift, sensitivity to policy hyperparameters, and the choice of action space. — [arXiv:2309.03315](https://arxiv.org/abs/2309.03315)

**Achieving Human Level Competitive Robot Table Tennis (Google DeepMind 2024)**
- Hardware: "a 6 DoF ABB IRB 1100 arm mounted on top of two Festo linear gantries" (4 m by 2 m). — [HTML](https://arxiv.org/html/2408.03906)
- Sensing: "Two Ximea MQ013CG-ON cameras" capture the ball at 125 Hz, and a neural perception system outputs ball positions at that rate. A 20-camera PhaseSpace motion-capture system tracks the opponent's paddle via LEDs. — [HTML](https://arxiv.org/html/2408.03906)
- Architecture: a hierarchy with low-level skill controllers (LLCs), each with a skill descriptor, and an event-driven high-level controller (HLC) that picks a skill once per opponent hit. LLCs output joint-velocity commands at 50 Hz with about 3 ms CPU inference, and the HLC runs within 20 ms. — [HTML](https://arxiv.org/html/2408.03906)
- LLC training: MuJoCo plus BGS (ES). The authors chose ES over PPO/SAC because those produced "noticeably jerkier" actions. Each policy is a dilated-gated CNN of about 10k parameters. The observation is 8 timesteps × (ball position and velocity, joint positions, style), described as "0.14 seconds of history." Initial ball states are sampled from real data. Observation noise and latency are domain-randomized following prior work. — [HTML](https://arxiv.org/html/2408.03906)
- Results: the robot won "45% (13/29)" of matches overall. It won "100%" against beginners and "55%" against intermediates, and lost every match against advanced players. Points won were 72%, 50% and 34% respectively. Return rate drops to about 50% with some underspin and approaches zero at high underspin. — [arXiv:2408.03906](https://arxiv.org/abs/2408.03906), [HTML](https://arxiv.org/html/2408.03906)
- Baselines: "No direct experimental comparison to non-learned controllers." The authors compare qualitatively with their prior iterative work, which needed about 6 h of real fine-tuning, and with the model-based Omron Forpheus system. Simulated return rates were "typically above 80%," and real-world rates were lower. — [HTML](https://arxiv.org/html/2408.03906)

### Inferences
- The proven recipe for fast interception is a factored pipeline: a dedicated high-rate tracker (60–125 Hz), a short history of estimated target state as the policy input, then a small neural policy that outputs joint velocities. This is the same structure as the planned privileged-state teacher, which supports using a teacher that sees the partner palm's position and velocity history.
- Google's finding that ES gave smoother joint-velocity actions than PPO/SAC is worth noting. If PPO's dap trajectories turn out jerky, consider action filtering, an action-rate penalty, or a smaller policy.
- Table tennis has no published learned-vs-analytic success rates on the same hardware. A dap project with paired seeds against a diff-IK servo would be more controlled than most of this literature.

### Gaps
- GoalsEye precision numbers and the RSS 2023 latency-ablation numbers were not extracted (abstracts only).
- Tebbe et al., "Sample-efficient reinforcement learning in robotic table tennis" (ICRA 2021, arXiv:2011.03275 from memory) is UNVERIFIED and was not checked.
- Kober's ball-in-a-cup work (Kober & Peters, policy search for motor primitives, NIPS 2008 / Machine Learning 2011, from memory) is UNVERIFIED. It is a dynamic skill, not interception of an independently moving target.

---

## Q2. Learned catching: observations (state vs pixels), privileged info, teacher-student

### Takeaway
Every learned catching system I checked feeds the policy an estimated object state (3D position, or a predicted trajectory) from motion capture, a stereo tracker, or an RGB-D color-blob detector, not raw pixels. Privileged information in this area mostly takes the form of an asymmetric critic or a separate trajectory predictor, not DAgger-style teacher-student distillation. The clearest head-to-head comparison of learned and model-based catching (Agile Catching, 2023) found that the learned policy matches or beats MPC in-distribution but is much less robust out of distribution.

### Cited Findings
**Kim, Shukla & Billard 2014 (T-RO): learning from demonstration for catching**
- Robot: KUKA LWR 4+ (7 DoF) with a 16-DoF Allegro hand, "controlled in joint positions at 500 Hz," plus iCub in simulation. — [Kim 2014 PDF](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)
- Sensing: objects tracked by "the Optitrack motion capture system" at "240 Hz," and the trajectory is filtered with a 25 Hz Butterworth filter. — [Kim 2014 PDF](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)
- Learned components: SVR-RBF models of object flight dynamics, compared against a rigid-body dynamics model, which predicted the trajectories of non-ballistic objects (such as a partially filled bottle) less accurately. GMMs model the graspable space and the reachable space, and the reachable-space model had "22" Gaussians for the LWR. Arm motion comes from coupled dynamical systems learned from kinesthetic demonstrations, which "enables a rapid and reactive adaptation of the arm motion in the presence of sensor uncertainty." — [Kim 2014 PDF](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)
- Results: about 100% catch success in iCub simulation. On the real LWR, the object was caught "52 times out of 71 total trials, yielding a 73.2% success rate." The human comparison was "38%" overall (38 of 100), ranging from 10% to 70% per subject. Catch timing must be fast; the paper says the planner must work "of the order of 10 ms." — [Kim 2014 PDF](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)

**Agile Catching with Whole-Body MPC and Blackbox Policy Learning (Abeyruwan et al., L4DC 2023)**
- Hardware: ABB IRB 120T 6-DoF arm on a 1-D Festo linear actuator (7 DoF in all) with a lacrosse head. Ball tracking uses a stereo pair of Ximea MQ013CG-ON cameras with a trained recurrent tracker. The thrower is about 3.9 m away and the ball reaches about 5.5 m/s at catch. Agents run at 75 Hz, and vision plus joint processing takes about 5 ms. — [HTML v2](https://arxiv.org/html/2306.08205v2)
- Abstract: the system has "access only to visual observations of the object and the proprioceptive state of the robot." — [arXiv:2306.08205](https://arxiv.org/abs/2306.08205)
- Model-based arm: free-end-time optimal control (second-order-integrator joints, hard constraint on pose alignment with the predicted ball), solved with a shooting-based SQP and re-solved asynchronously, followed by an open-loop cradling primitive. — [HTML v2](https://arxiv.org/html/2306.08205v2)
- Learned arm: a two-tower CNN with about 3,255 parameters. Inputs are joint-position history and the predicted ball trajectory, a structured state rather than pixels, and the outputs are joint velocities. Trained with BGS in PyBullet, then fine-tuned on hardware with 30 throws per iteration. About 40 iterations were needed to match SQP on the real robot. — [HTML v2](https://arxiv.org/html/2306.08205v2)
- Results (as extracted from the HTML v2 text):
  - Both agents reach about 80–85% on mechanical throws.
  - On hand throws, SQP reached 68.9% (196 throws) and the blackbox policy 2.0% (150 throws).
  - Inference takes about 7.25 ms per action for the policy, against about 43.0 ms (std 21.3 ms) for the asynchronous SQP solve, whose synchronous part is about 2.1 ms.
  - The blackbox policy degrades on slower throws, while SQP stays robust.
  - The blackbox policy catches only on the right, mirroring its 60/40 right-biased training data, while SQP is multimodal.
  - Conclusion: the fine-tuned blackbox agent "achieves the highest success on the training distribution" but is "less robust to distribution shifts." The authors suggest hybrids: learning SQP parameters, or handing off from SQP to a learned policy just before the intercept.
  - Source: [HTML v2](https://arxiv.org/html/2306.08205v2)

**Catch It! (Zhang et al. 2024): RL on a mobile dexterous manipulator**
- Platform: mobile base, 6-DoF arm, 12-DoF dexterous hand. Trained with PPO in two stages. Stage 1 (tracking) counts success when "the palm touches the object in flight." Stage 2 learns to grasp and fine-tunes the base and arm. — [arXiv:2409.10319](https://arxiv.org/abs/2409.10319), [HTML](https://arxiv.org/html/2409.10319)
- Observation: two consecutive 3D positions of the object and end-effector (relative to the arm base), base planar velocity, and hand joint positions. Object positions come from the tracking pipeline, not privileged simulator state. No pixel input. Real sensing: an overhead RealSense D455 RGB-D camera extracts the object's pixel coordinates, then a perspective transform and eye-on-base calibration produce 3D position. No motion capture is used. — [HTML](https://arxiv.org/html/2409.10319)
- Rates: 25 Hz policy in simulation and on the real robot, 30 Hz proprioception, 50 Hz camera. There is no explicit latency compensation beyond synchronization and a low-pass filter (LPF) on velocity commands. — [HTML](https://arxiv.org/html/2409.10319)
- Simulation, unseen objects:
  - One-stage training catches only 6–22% (for example "22±2" on bowls).
  - Two-stage training (ours) catches 65–84% and tracks (palm touch) 88–95%.
  - Source: [HTML](https://arxiv.org/html/2409.10319)
- Real robot:
  - Tracking (palm touches the object) succeeds 65–75% with the LPF and 5–15% without it.
  - Catching succeeds 15–25% with the LPF and 0–5% without it.
  - The authors say "the catching success rate did not exceed 25%," and attribute this mainly to object elasticity.
  - Source: [HTML](https://arxiv.org/html/2409.10319)

**Modular in-flight catching (Hu et al., IROS 2023)**
- Five modules: (i) a supervised object-state estimator for trajectory prediction, (ii) a supervised catching-pose quality network, (iii) an RL reaching policy, (iv) an RL grasping policy, and (v) an RL gating network that blends the two. Trained on cylinders and spheres and generalizes to household objects. The abstract gives no numbers. — [arXiv:2312.13987](https://arxiv.org/abs/2312.13987)

**Dynamic Handover (Huang et al., CoRL 2023)**
- Multi-agent RL in simulation with sim-to-real transfer, on two arms with multi-finger hands, plus a learned object trajectory-prediction model for the catcher. The abstract reports "significant improvements over multiple baselines" without numbers. — [arXiv:2309.05655](https://arxiv.org/abs/2309.05655)

**Legged and mobile analogues whose methods transfer directly**
- Badminton (ANYmal-D plus arm, Science Robotics 2025):
  - Observation and privileged info: the actor sees proprioception, the racket interception command and time-to-interception, and the "shuttlecock position in the camera frame with a detection flag." Only the critic gets privileged noiseless state. This is an asymmetric actor-critic, "not a teacher-student setup."
  - Perception noise model: detection probability and noise are regressed from real camera data as functions of distance and angular velocity. The same EKF and trajectory predictor run in training and deployment.
  - Timing and training: a ZED X stereo camera at 60 fps with a "60-160 ms delay" from shutter to computed positions. The policy runs at 100 Hz and is trained with N-P3O (constrained PPO) in IsaacGym.
  - Results: up to 10 consecutive shots in a rally, and a peak swing of 12.06 m/s.
  - Source: [HTML](https://arxiv.org/html/2505.22974), [project page](https://sites.google.com/view/legged-racket-sport/home)
- Quadruped goalkeeper (Mini Cheetah): hierarchical RL in which locomotion skill policies (jump, dive, sidestep) are chosen by a high-level planner that picks a skill and an end-effector trajectory. Real-world interception rate was "87.5%." — [arXiv:2210.04435](https://arxiv.org/abs/2210.04435v1), [project page](https://xbpeng.github.io/projects/A1Goalkeeping/index.html)

### Inferences
- Catch It!'s Stage 1 success criterion, a palm touching a moving object in flight, is close to the dap's palm-to-palm contact. Its real-robot palm-touch rate of 65–75% with RGB-D color tracking at 25 Hz is a useful external reference, although Catch It! does not require antiparallel orientation.
- Catch It! and Agile Catching both found that action smoothing (an LPF) or real-robot fine-tuning were decisive in practice. For a kinematic sim this matters less, but an action-rate penalty or filter in the teacher will affect distillation quality.
- Agile Catching suggests the dap evaluation should report out-of-distribution splits, for example partner speeds near the 0.15 and 0.9 m/s extremes and mid-course target switches. That is where learned policies have been shown to lose to model-based controllers, so in-distribution success alone could overstate RL.
- A privileged teacher distilled by DAgger into a depth student is not standard in arm-catching papers. Its closest precedent is perception-aware drone flight (Q3), and the asymmetric critic in the badminton work is the nearest arm-relevant precedent.

### Gaps
- Hu et al. (IROS 2023) and Dynamic Handover numbers and sensing details were not extracted (abstracts only).
- The real-world success rate of Catch It! compared against a non-learned baseline was not reported. Its baselines are ablations (one-stage training, no arm roll, no LPF).
- Learned catching for drones and quadrupeds beyond these two examples was not surveyed, because of the arm focus.

---

## Q3. End-to-end vision (RGB/depth) policies for dynamic interception; latency and occlusion handling

### Takeaway
I found no peer-reviewed arm interception or catching work where a policy maps raw RGB or depth images straight to actions and succeeds on fast targets. Fielded systems use a separate perception module that outputs a low-dimensional target state. Latency and occlusion are handled by randomizing latency and noise in simulation, by fitting a noise and dropout model to real camera data, by feeding the policy a short state history or a predicted trajectory, and in some cases by learning active perception. The best-documented privileged-teacher to depth-student distillation for agile tasks comes from drones.

### Cited Findings
- Google table tennis 2024: the policy input is estimated ball state from a 125 Hz stereo neural tracker, with an 8-step history ("0.14 seconds"). The paper says it domain-randomizes "observation noise and latency" following prior work. — [HTML](https://arxiv.org/html/2408.03906)
- RSS 2023: the system paper explicitly studies "how various sources of latency affect performance" and evaluates the robustness of its perception subsystem. — [arXiv:2309.03315](https://arxiv.org/abs/2309.03315)
- Badminton 2025:
  - Latency, occlusion and field of view: a measured "60-160 ms delay" from shutter to computed shuttle positions, and a narrow FOV. Occlusion and FOV loss enter the actor as a detection flag plus a fitted detection-probability and noise model.
  - Active perception: the learned behavior gives "0.10 s of additional sight of the shuttle flight." Ablations compared against a "No Perception Behavior" policy and an "FOV Reward" policy, and the text says the proposed method "notably outperformed" the no-perception policy in one court region.
  - Source: [HTML](https://arxiv.org/html/2505.22974)
- Catch It! 2024: RGB-D (D455) is used only to localize the object, which goes into a state-based PPO policy. Without the LPF on actions, real-robot catching was 0–5%. — [HTML](https://arxiv.org/html/2409.10319)
- Agile Catching 2023: a trained recurrent stereo tracker feeds a predicted ball trajectory to the policy, not pixels. — [HTML v2](https://arxiv.org/html/2306.08205v2)
- Perception-aware agile flight (drone; the method transfers directly): "using RL, the authors first train a perception-aware teacher policy with full-state information … then … use imitation learning to distill its knowledge into a vision-based student policy" that maps depth images to commands. The authors report better success rate, 10x faster computation than planning baselines, and real flights at up to 50 km/h. — [arXiv:2210.01841](https://arxiv.org/pdf/2210.01841) (abstract and figure caption as summarized in search results; full text not read)
- General end-to-end visuomotor precedent (static tasks): Levine et al., "End-to-End Training of Deep Visuomotor Policies," JMLR 2016, arXiv:1504.00702, maps monocular RGB and joint state to torques at 20 Hz on a 7-DoF arm for placement tasks, not moving targets. — [arXiv PDF](https://arxiv.org/pdf/1504.00702)
- GraspARL 2022 trains a grasping policy with RL against an adversarial "mover," formulated as a min-max "move-and-grasp" game, for objects on a moving platform. The abstract does not state the observation modality or give numbers. — [arXiv:2203.02119](https://arxiv.org/abs/2203.02119)

### Inferences
- The dap's 30 fps depth camera with 1-frame latency, about 33 ms (simple arithmetic, not taken from a source), is gentler than badminton's 60–160 ms and on par with the stereo trackers in table tennis. Its target speeds of 0.15–0.9 m/s are also slow next to balls at 5.5 m/s or more. So latency is a manageable rather than dominant issue. A student given a short frame stack or a recurrent state should be able to extrapolate the partner palm's motion.
- Because no arm interception paper does pixels-to-actions on fast targets, a depth-image student distilled with DAgger would be somewhat novel for arms. It needs ablations that separate perception error from control error, for example a student fed ground-truth palm state against one fed depth.
- Badminton's perception noise model is a cheap, well-cited way to make the teacher robust to what the student will see: train the teacher on noisy or dropped observations with a detection flag, as in an asymmetric critic. This also makes DAgger easier.
- Head-camera occlusion of the partner palm by the robot's own arm near contact is the dap analogue of badminton's FOV loss. Holding the last estimate with a detection flag, or using a recurrent student, is the standard remedy.

### Gaps
- I found no verified paper that does end-to-end RGB or depth-to-joint-command interception of a moving target with an arm. One more targeted search on event cameras, pixels or "visuomotor catching" might find workshop papers, but none were confirmed here.
- The Google DeepMind latency values used for randomization are in cited prior work and were not extracted.
- Numerical results of the perception-aware drone student against its teacher were not read from the full text.

---

## Q4. Baselines used, and where learning beat or lost to model-based control

### Takeaway
Most learned-interception papers compare against ablations of their own method or against humans, not against tuned model-based controllers. The few direct comparisons point the same way. Learned policies win in-distribution and on inference speed, and when the task objective can be optimized directly instead of through an intermediate trajectory. Model-based planners and MPC win on robustness to distribution shift, on multimodality, and on needing little or no real-world data.

### Cited Findings
- Agile Catching (the most direct arm comparison):
  - In-distribution the two are about equal, around 80–85% on mechanical throws. Out of distribution SQP wins: hand throws 68.9% against 2.0%, plus slower throws, out-of-distribution yaw, and left-side catches.
  - The learned policy is faster per action (about 7.25 ms against about 43 ms for the asynchronous solve) but needed about 40 real fine-tuning iterations of 30 throws each to match SQP.
  - The authors recommend hybrids.
  - Source: [HTML v2](https://arxiv.org/html/2306.08205v2)
- Human-level TT 2024: no experimental comparison to non-learned controllers. Baselines are humans of graded skill and prior learned systems. The model-based Omron Forpheus is discussed only qualitatively. — [HTML](https://arxiv.org/html/2408.03906)
- i-Sim2Real: the baseline is a learned variant (S2R+FT), and i-S2R gave rallies "70% to 175% longer" for 80% of players. — [arXiv:2207.06572](https://arxiv.org/abs/2207.06572)
- Mülling 2013: initialized from an analytical player's strokes in simulation. Learning raised success from "67 %" (untrained mixture) to "94 %" in simulation and from "69%" to "97%" on the real launcher test. — [Mülling 2013 PDF](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Muelling_IJRR_2013.pdf)
- Kim 2014: the learned SVR-RBF dynamics model was compared against a rigid-body dynamics model, which predicted less accurately for non-ballistic objects. The human catching baseline (38%) was below the robot (73.2%). — [Kim 2014 PDF](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)
- Catch It!: the baselines are training ablations (one-stage, no arm roll, no LPF), not an analytic planner. — [HTML](https://arxiv.org/html/2409.10319)
- Goalkeeper: compared against a human goalkeeper save rate (about 69% per a search summary). Search results noted this comparison is not like-for-like. — [project page](https://xbpeng.github.io/projects/A1Goalkeeping/index.html) (the human figure is UNVERIFIED in primary text)
- Why RL can beat optimal control (drone racing, Science Robotics 2023): RL "optimizes a better objective." Optimal control splits planning from control through an intermediate trajectory, which "limits the range of behaviors that can be expressed by the controller." RL plus domain randomization gives "more robust control responses" to unmodeled effects. — [arXiv:2310.10943](https://arxiv.org/abs/2310.10943)
- Perception-aware flight: the distilled vision student was reported 10x faster to compute than model-based planning and more successful. — [arXiv:2210.01841](https://arxiv.org/pdf/2210.01841)

### Inferences
- For the dap, a reactive diff-IK servo plus a blind baseline already exceeds what most learned-interception papers compare against. The literature suggests also reporting:
  - An oracle model-based variant with ground-truth state, for example diff-IK aimed at a predicted intercept point. Agile Catching shows a planner on predicted trajectories is the strong, fair baseline.
  - Out-of-distribution splits (speed extremes, target switches), where the literature predicts learned policies are weakest.
  - Teacher (privileged) against student (depth), to price the perception gap.
- The racing paper's "better objective" argument fits the dap: the success criterion (antiparallel palm contact at the right time) is awkward to express as a servo error, but direct to reward in RL. This is the main hypothesis for why PPO could beat the reactive servo.
- In a purely kinematic sim, the usual advantages of model-based control (no real-world data, robustness to dynamics mismatch) shrink. The remaining model-based edge is robustness to distribution shift, such as mid-course target switches, so it is worth testing specifically.

### Gaps
- Head-to-head optimal control against RL numbers for drone racing were not extracted (abstract only).
- I found no paper comparing learned interception directly with a reactive visual-servo or diff-IK baseline on an arm. This looks like a real gap in the literature, though the search was not exhaustive.
- Air-hockey RL benchmarks (for example the Robot Air Hockey Challenge) were not covered and may hold more learned-vs-model-based arm comparisons.
