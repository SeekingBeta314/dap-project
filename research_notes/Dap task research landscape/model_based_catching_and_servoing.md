# Model-based catching/interception, visual servoing, and differential IK for moving targets

Scope: classical (model-based) methods relevant to our "dap" task. In the task, a Drake-simulated, purely kinematic KUKA iiwa7 must meet a partner palm antiparallel. The partner moves on a min-jerk reach at 0.15–0.9 m/s and is seen by a head-mounted depth camera. Our baseline is a 100 Hz DLS resolved-rate servo with a constant-acceleration KF, 0.1 s look-ahead, a 5 cm standoff followed by closing, and partner-velocity feedforward.

How citations were checked: unless marked otherwise, every bibliographic record (authors, title, venue, year, vol/no/pages, DOI) was pulled from the Crossref API (api.crossref.org), which is the DOI registry's metadata. Content claims were read from the full text of author or open-access PDFs (EPFL Infoscience, DLR elib, INRIA HAL, arXiv) where possible. Anything I could not check against a primary listing is marked **UNVERIFIED**.

---

## Q1. Landmark robot catching papers: pipeline, sensing, success rates, speeds

### Takeaway
Every classical catching system follows the same four-stage pipeline: (1) a model-based trajectory predictor (parabola/ballistic + LSQ or EKF, or learned dynamics), (2) explicit selection of the intercept point *and* time subject to a reachability model, (3) trajectory generation to that point, re-planned at sensor rate (10–20 ms), and (4) low-level joint control, often through damped-least-squares IK. Catch rates for thrown objects are about 70–80+% at 5–7 m/s and ~0.3–1 s flight times. Those speeds are an order of magnitude faster than our 0.15–0.9 m/s partner palm.

### Cited Findings

**Bäuml, Wimböck & Hirzinger 2010.** "Kinematically optimal catching a flying ball with a hand-arm-system," *2010 IEEE/RSJ IROS*, pp. 2592–2599, DOI 10.1109/IROS.2010.5651175. Metadata from Crossref; full text at [DLR elib PDF](https://elib.dlr.de/67146/1/B%C3%A4uml2010.pdf).
- **Hardware:** 7-DOF DLR-LWR-III arm plus a 12-DOF DLR-Hand-II. — [DLR elib](https://elib.dlr.de/67146/1/B%C3%A4uml2010.pdf)
- **Unified optimization:** "where, when and how to catch" is one nonlinear optimization with nonlinear constraints, under joint, speed and work-cell limits. Three objective functions give clearly different behaviors (e.g., "soft", "latest" modes). The decision variables are catch configuration and catch time, (q_c, t_c) = argmin H(q,t), subject to equality and inequality constraints. — [DLR elib](https://elib.dlr.de/67146/1/B%C3%A4uml2010.pdf)
- **The paper frames the "general setup" of all prior ball catchers this way:** "a stereo vision system tracks the ball and predicts the balls trajectory, then the point and time … the robot should intercept the ball" is determined, then the catch configuration, then a path. Bäuml et al. merge these separate steps into one problem. — [DLR elib](https://elib.dlr.de/67146/1/B%C3%A4uml2010.pdf)
- **Sensing and prediction:**
  - Stereo PAL cameras at 50 Hz field rate.
  - EKF-based estimation and prediction, giving a new prediction every 20 ms.
  - Required catch precision is <2 cm and <5 ms, so the planner re-plans every 20 ms.
  - SQP (DONLP2) is run as a parallel multi-start on a cluster of up to 32 cores. The search is cut off at 60 ms.
  - Arm and hand controllers run at 1 kHz.
  - Total camera-to-motion delay is 90 ms, and it is explicitly compensated in the prediction.
  
  — [DLR elib](https://elib.dlr.de/67146/1/B%C3%A4uml2010.pdf)
- **Speeds:** the ball is thrown from ~5 m at a typical 7 m/s, giving ~1 s flight. — [DLR elib](https://elib.dlr.de/67146/1/B%C3%A4uml2010.pdf)
- **Success:** >80% catch rate in "soft" and "latest" modes, counting balls thrown into the feasible catch space. The main failure source is vision (prediction errors, lost tracks), not the arm or hand. — [DLR elib](https://elib.dlr.de/67146/1/B%C3%A4uml2010.pdf)
- **Predecessor:** Frese, Bäuml, Haidacher, Schreiber, Schaefer, Hähnle & Hirzinger, "Off-the-shelf vision for a robotic ball catcher," *Proc. 2001 IEEE/RSJ IROS*, vol. 3, pp. 1623–1629, DOI 10.1109/IROS.2001.977211. It used a DLR-LWR-II with a basket and "off-the-shelf" 50 Hz PAL stereo. — metadata from Crossref; description from [Bäuml 2010](https://elib.dlr.de/67146/1/B%C3%A4uml2010.pdf). Kim et al. say Frese et al. use a ballistic model with air drag in an EKF, and choose the intercept point closest to the end-effector's initial position, with reachability approximated as an infinite polyhedron. — [Kim et al. 2014 PDF](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)
- **Follow-ups on the humanoid Rollin' Justin:**
  - Birbach, Frese & Bäuml, "Realtime perception for catching a flying ball with a mobile humanoid," *ICRA 2011*, pp. 5955–5962, DOI 10.1109/ICRA.2011.5980138.
  - Bäuml et al., "Catching flying balls with a mobile humanoid: System overview and design considerations," *Humanoids 2011*, pp. 513–520, DOI 10.1109/Humanoids.2011.6100837.
  
  Both are metadata only (Crossref); I did not read their content.

**Kim, Shukla & Billard 2014.** "Catching Objects in Flight," *IEEE Transactions on Robotics* 30(5):1049–1065, DOI 10.1109/TRO.2014.2316022. Metadata from Crossref; full text at [EPFL Infoscience](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1).
- **Problem decomposition:** "accurate prediction of the trajectory … predicting the feasible catching configuration, and planning the arm motion, and all within milliseconds". Objects are non-spherical (hammer, racket, bottles, box) and the grasp point is not at the COM. — [Kim 2014](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)
- **Prediction:** object dynamics are learned from ~20 demonstration throws per object, using SVR with an RBF kernel. Acceleration is predicted from the current state, combined with an EKF, and integrated forward for a full 6-D trajectory. This builds on Kim & Billard, "Estimating the non-linear dynamics of free-flying objects," *Robotics and Autonomous Systems* 60(9):1108–1122, 2012, DOI 10.1016/j.robot.2012.05.022 (Crossref). — [Kim 2014](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)
- **Catch point selection:**
  - Probabilistic. A GMM models the *reachable* space (22 Gaussians for the LWR 4+) and the *graspable* space (e.g., 13–15 Gaussians), each learned from demonstration.
  - The catch configuration is found where the predicted object trajectory passes through high-likelihood reachable-and-graspable regions.
  - Computation takes about 0.2 ms on a 2.7 GHz quad-core.
  
  — [Kim 2014](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)
- **Motion generation:**
  - A coupled dynamical system (CDS) learned from demonstrations drives the hand. A scalar "boost factor" makes the end-effector arrive at the desired time.
  - "The output of the position and orientation of the DS is converted into the 7-DOF joint state using the damped least squares IK", with conservative joint limits and a critically damped filter.
  - The robot is controlled in joint position at 500 Hz.
  - The catching configuration is "updated at every control cycle".
  - Prediction stops once the predicted time-to-contact is <0.09 s, because occlusion degrades estimates near the robot.
  
  — [Kim 2014](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)
- **Sensing:** OptiTrack motion capture at 240 Hz, three markers per object. Tracking starts ~3.5 m from the robot. — [Kim 2014](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)
- **Results:**
  - Real robot (KUKA LWR 4+ with Allegro hand): 52 of 71 valid throws caught, a 73.2% success rate. 9 of 80 throws never entered the reachable space and were excluded.
  - iCub simulation: 100% (3 of 50 throws excluded as unreachable).
  - Untrained humans scored 38% on the same task.
  - The method can catch objects whose total flight time is ≤0.7 s.
  
  — [Kim 2014](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)
- **Failure analysis (useful for our diff-IK baseline):**
  - 12 of 19 failures came from the IK catch configuration being too far from the start configuration. The Cartesian DS "does not take into account the limits of robot in joint space (e.g., joint velocity or torque)", so the target was not reached in time.
  - 4 of 19 were fingers hitting the object.
  - 3 of 19 were torque-limit violations.
  
  — [Kim 2014](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)
- **Note on an inconsistency in the source:** the PDF text reports the average flying time as "4.97 ± 0.46 s". This conflicts with the paper's own statement that objects are caught when flight "does not exceed 0.7 s", and with the 3.5 m throwing distance. It is probably a typo for ~0.5 s. Do not quote the 4.97 s figure. — [Kim 2014](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)
- **Kim et al.'s summary of earlier catchers:**
  - Hong & Slotine and Riley & Atkeson model the ball as a parabola fit by recursive least squares.
  - Earlier works take the intersection of the predicted trajectory with a coarse reachable-space boundary (spheres, planes, or a polyhedron), then choose the point closest to the robot base or to the initial end-effector position.
  - Polynomial trajectories satisfying boundary conditions are common. They are "explicitly time dependent", so perturbations after motion onset are poorly handled.
  - Re-planning must run on the order of 10 ms.
  
  — [Kim 2014](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)

**Salehian (Mirrazavi Salehian), Khoramshahi & Billard 2016.** "A Dynamical System Approach for Softly Catching a Flying Object: Theory and Experiment," *IEEE Transactions on Robotics* 32(2):462–471, DOI 10.1109/TRO.2016.2536749. Metadata from Crossref; full text at [EPFL Infoscience](https://infoscience.epfl.ch/record/217007/files/TRO.pdf). The preprint title word order is "Catching Softly"; the Crossref title is "Softly Catching".
- **Idea:** "soft catching". The hand meets the object at the intercept point, then keeps moving and matches the object's velocity before slowing to zero. "Hard" catching, which stops the object at once, makes stiff objects bounce out. — [Salehian 2016](https://infoscience.epfl.ch/record/217007/files/TRO.pdf)
- **Controller:**
  - A DS written as a Linear Parameter Varying (LPV) system, with parameters fitted by GMM to kinematically feasible demonstrations from an offline optimal-control solver.
  - Lyapunov analysis proves the end-effector intercepts the object at the intercept point, at the right time, with the desired velocity direction.
  - The intercept point comes from the Kim 2014 prediction algorithm. The palm orientation is constrained so that it "face[s] opposite the trajectory of the object". This is directly analogous to our antiparallel palm constraint.
  
  — [Salehian 2016](https://infoscience.epfl.ch/record/217007/files/TRO.pdf)
- **Implementation:**
  - KUKA LBR iiwa (7-DOF) with a 16-DOF Allegro hand, joint position control at 500 Hz.
  - DS output goes to joints through "velocity based control without joint velocity integration", with a critically damped filter.
  - OptiTrack at 240 Hz. Predicted object positions are used in the faster control loop.
  - Post-interception, velocity is reduced linearly over 0.3 s.
  
  — [Salehian 2016](https://infoscience.epfl.ch/record/217007/files/TRO.pdf)
- **Results:** 20 throws per object, with flight time from the first tracked point to intercept of about 0.32–0.33 s.

  | Object | Soft catching | Hard catching (Kim 2014 method) |
  |---|---|---|
  | Small stiff ball | 70% | 0% |
  | Stiff brick | 70% | 0% |
  | Semi-stiff toy | 75% | 5% |

  The first 0.4 m of each throw is used to initialize prediction. — [Salehian 2016](https://infoscience.epfl.ch/record/217007/files/TRO.pdf)

**Hove & Slotine 1991.** "Experiments in Robotic Catching," *1991 American Control Conference*, pp. 380–386 per Crossref (380–385 per the MIT NSL page), DOI 10.23919/ACC.1991.4791395.
- **Method:** "All path-planning for the catch occurs in real-time during the half-second that the targeted object is airborne". The approach is a "trajectory-matching algorithm that combines an observer with a varying-strength filter, an error estimator, and an initial motion algorithm". — [MIT NSL abstract](https://web.mit.edu/nsl/www/abstracts/robotcatch-abst.html)
- **Hardware:** a real-time vision system and a 4-DOF cable-driven arm with a 4.2 m³ workspace and speeds up to 2.0 m/s. — [MIT NSL abstract](https://web.mit.edu/nsl/www/abstracts/robotcatch-abst.html)
- **Related:** Hong & Slotine, "Experiments in hand-eye coordination using active vision," *Experimental Robotics IV* (Lecture Notes in Control and Information Sciences), 1997, pp. 130–139, DOI 10.1007/BFb0035204 (Crossref). It is cited for parabola + recursive least-squares prediction and a closest-to-base intercept choice. — [Kim 2014](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)

**Riley & Atkeson 2002.** "Robot Catching: Towards Engaging Human-Humanoid Interaction," *Autonomous Robots* 12(1):119–128, DOI 10.1023/A:1013223328496 (Crossref).
- Kim et al. characterize it as using a parabolic ball model with recursive LSQ, plus movement primitives for humanoid catching. — [Kim 2014](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)
- I did not read the full text, so the hardware and success rates are UNVERIFIED here.

**Ishikawa lab (high-speed vision).**
- Namiki & Ishikawa, "Robotic catching using a direct mapping from visual information to motor command," *ICRA 2003*, pp. 2400–2405, DOI 10.1109/ROBOT.2003.1241952 (Crossref).
- Senoo, Namiki & Ishikawa, "High-speed batting using a multi-jointed manipulator," *ICRA 2004*, pp. 1191–1196, DOI 10.1109/ROBOT.2004.1307986 (Crossref).
- Senoo, Namiki & Ishikawa, "Ball control in high-speed batting motion using hybrid trajectory generator," *ICRA 2006*, pp. 1762–1767, DOI 10.1109/ROBOT.2006.1641961 (Crossref).
- Kim et al. say Namiki & Ishikawa generate polynomial trajectories that minimize the sum of torques and angular velocities subject to boundary constraints. — [Kim 2014](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)
- Specific numbers (e.g., 1 kHz vision, catch rates) are UNVERIFIED; I did not read these papers.

**Other optimal-catching references (metadata only, Crossref):**
- Lampariello, Nguyen-Tuong, Castellini, Hirzinger & Peters, "Trajectory planning for optimal robot catching in real-time," *ICRA 2011*, pp. 3719–3726, DOI 10.1109/ICRA.2011.5980114.
- Kober, Glisson & Mistry, "Playing catch and juggling with a humanoid robot," *Humanoids 2012*, pp. 875–881, DOI 10.1109/HUMANOIDS.2012.6651623.

### Inferences
- **Our pipeline matches the classical one in prediction and low-level control:** KF prediction, then a target pose, then DLS joint velocities. It is weaker on stage 2, the explicit intercept point and time selection. Our servo chases a fixed 0.1 s-ahead prediction rather than choosing a reachable rendezvous on the partner's predicted path. Every landmark catcher (Hove & Slotine, Frese, Bäuml, Kim, Salehian) has an explicit intercept-selection step with a reachability model.
- **Kim et al.'s dominant failure mode applies directly to our servo:** a Cartesian-space target generator ignored joint velocity limits, so the arm could not arrive in time. Our QP/DLS with velocity limits shares this risk whenever the commanded palm target moves faster than the arm can follow near joint limits or singularities.
- **The regime is much easier:** our partner moves at 0.15–0.9 m/s on a smooth min-jerk profile, while catchers face 5–7 m/s with ≤1 s flight. Reactive servoing is much more plausible for us than for ball catching, which is why catching-style global replanning is not strictly needed.
- **Salehian's "soft catching" is the closest classical analogue to an antiparallel palm-to-palm touch:** palm faces opposite the object's velocity, and the hand velocity is matched at contact. It is a natural design reference for a "velocity-matched touch" metric.

### Gaps
- No full text read for Riley & Atkeson 2002, Hove & Slotine 1991, or the Ishikawa papers. Their success rates and hardware are not verified beyond the abstracts and secondary characterizations above.
- I found no catching paper whose target is a slowly moving (<1 m/s) human-like palm. The closest regimes are handover and tracking papers (Q2/Q3).

---

## Q2. Visual servoing landmarks; PBVS vs IBVS; handling moving targets

### Takeaway
The canonical tutorials define PBVS and IBVS as resolved-rate laws v = -λ L̂⁺ e. For a moving target they add a feedforward term, v = -λ L̂⁺ e − L̂⁺ ∂ê/∂t. Without an accurate estimate of target motion this leaves a steady tracking error, so filtering/prediction (Allen et al. 1993) or integral/predictive control is recommended. Our "partner-velocity feedforward + KF prediction" is exactly this textbook moving-target PBVS structure.

### Cited Findings
- **Bibliographic records (all Crossref):**
  - Hutchinson, Hager & Corke, "A tutorial on visual servo control," *IEEE Transactions on Robotics and Automation* 12(5):651–670, 1996, DOI 10.1109/70.538972.
  - Corke & Good, "Dynamic effects in visual closed-loop systems," *IEEE Transactions on Robotics and Automation* 12(5):671–683, 1996, DOI 10.1109/70.538973. This covers visual-loop latency and dynamics in the same special issue.
  - Chaumette & Hutchinson, "Visual servo control. I. Basic approaches," *IEEE Robotics & Automation Magazine* 13(4):82–90, Dec. 2006, DOI 10.1109/MRA.2006.250573.
  - Chaumette & Hutchinson, "Visual servo control. II. Advanced approaches [Tutorial]," *IEEE Robotics & Automation Magazine* 14(1):109–118, Mar. 2007, DOI 10.1109/MRA.2007.339609.
- **Part I is explicitly limited** to "a fixed goal pose and a motionless target, i.e., s* is constant". More general cases are deferred to Part II. — [Chaumette & Hutchinson Part I, HAL](https://inria.hal.science/inria-00350283/document)
- **Part II, "Target Tracking" section:**
  - For a moving target, ė = L_e v_c + ∂e/∂t, which gives the control law v_c = −λ L̂e⁺ e − L̂e⁺ (∂ê/∂t). The second term "must be introduced in the control law to compensate for the target motion".
  - The error converges to zero only if the estimate of ∂e/∂t is accurate enough. "Otherwise, tracking errors will be observed."
  - Raising λ reduces tracking error, but too high a gain destabilizes the loop.
  - Remedies listed: an integral term (cancels error only for constant-velocity targets), filters that "smooth or predict the motion" (citing Allen et al. 1993 among others), and predictive controllers.
  - For a time-varying desired feature, v_c = −λ L̂e⁺ e + L̂e⁺ ṡ*, where the second term "anticipates the variation of s*".
  
  — [Chaumette & Hutchinson Part II, HAL](https://inria.hal.science/inria-00350638/document)
- **Part II also gives a joint-space form,** ṡ = J_s q̇ + ∂s/∂t. Computing q̇ directly with the feature Jacobian is "generally not equivalent" to computing v_c first and then inverting the robot Jacobian, because J(q) may be singular when J_s is not. — [Chaumette & Hutchinson Part II, HAL](https://inria.hal.science/inria-00350638/document)
- **Part II cites Papanikolopoulos, Khosla & Kanade (1993)** for LQG-optimal visual tracking of a moving target: "Visual tracking of a moving target by a camera mounted on a robot: A combination of vision and control," *IEEE Transactions on Robotics and Automation* 9(1):14–35, Feb. 1993. The reference string is taken from the tutorial's bibliography; I did not separately confirm it in Crossref. — [Chaumette & Hutchinson Part II, HAL](https://inria.hal.science/inria-00350638/document)
- **Allen, Timcenko, Yoshimi & Michelman 1993:** "Automated tracking and grasping of a moving object with a robotic hand-eye system," *IEEE Transactions on Robotics and Automation* 9(2):152–165, DOI 10.1109/70.238279 (Crossref). Companion paper: "Trajectory filtering and prediction for automated tracking and grasping of a moving object," *ICRA 1992*, pp. 1850–1856, DOI 10.1109/ROBOT.1992.219958 (Crossref).
  - **Content (from search-result summaries of the abstract; full text not read):**
    - A PUMA-560 with a parallel-jaw gripper tracks, grasps and picks up a moving model train.
    - Two fixed cameras compute stereo optic flow.
    - There are three sub-problems: 3-D motion estimation, predictive arm control to track the object, and interception/grasping.
    - The system works at "approximately human arm movement rates".
    - Their earlier work used an α-β-γ filter for prediction, with vision at ~5 Hz. The 1992 paper says a local-coordinate filter was needed because the previous tracking was "smooth but not accurate enough to allow actual grasping".
    
    These are UNVERIFIED against full text. — [Columbia Academic Commons / search summary](https://academiccommons.columbia.edu/doi/10.7916/D8RB7DV4); [bib record for ICRA 1992](https://bvbvf1.bib-bvb.de/vufind/PrimoRecord/cdi_ieee_primary_219958)

### Inferences
- **Our baseline is a position-based visual servo (PBVS) on a moving target** with an estimated ∂e/∂t feedforward and model-based prediction. That is the textbook remedy in Chaumette & Hutchinson Part II, so it is defensible as a "standard classical baseline". The depth camera gives 3-D palm pose, which is the PBVS setting.
- **Chaumette & Hutchinson's analysis predicts our servo's characteristic failure:** residual lag/tracking error whenever the feedforward estimate (KF velocity) is wrong. Min-jerk profiles have non-constant acceleration, so a constant-acceleration KF will be biased during the jerk phases. That suggests reporting tracking-error-vs-time as a diagnostic.
- **Allen et al. 1993 is the closest classical precedent in speed regime and structure:** tracking and grasping a slowly moving object at "human arm movement rates" with predictive filtering. It is the natural historical citation for "predict, then servo, then grasp a moving object".

### Gaps
- I did not read the Hutchinson/Hager/Corke 1996 full text. Its specific treatment of moving targets is not quoted here (the 2006/2007 tutorials supersede it on this point).
- Allen et al. 1993 numerical results (speeds, success counts) were not verified.

---

## Q3. Differential IK / resolved-rate control; is a DLS diff-IK servo with prediction a standard baseline?

### Takeaway
Resolved-rate control (Whitney 1969) with damped-least-squares / singularity-robust inverses (Nakamura & Hanafusa 1986; Wampler 1986; Chiaverini et al. 1994; Buss's survey) is the standard low-level layer. It sits inside catching systems (Kim 2014 uses DLS IK), PBVS handover systems (Rosenberger 2020), and Drake's DifferentialInverseKinematics QP. As a whole-task baseline for *learned* methods, the literature more often uses PBVS/IBVS servoing, reactive/RMP controllers, motion planners, or MPC rather than "pure diff-IK". So our baseline is best described and cited as a "PBVS + DLS resolved-rate servo with predictive feedforward". The strongest classical comparison in recent work is MPC.

### Cited Findings
- **Foundational records (Crossref):**
  - Whitney, "Resolved Motion Rate Control of Manipulators and Human Prostheses," *IEEE Transactions on Man-Machine Systems* 10(2):47–53, June 1969, DOI 10.1109/TMMS.1969.299896.
  - Nakamura & Hanafusa, "Inverse Kinematic Solutions With Singularity Robustness for Robot Manipulator Control," *ASME Journal of Dynamic Systems, Measurement, and Control* 108(3):163–171, 1986, DOI 10.1115/1.3143764.
  - Wampler, "Manipulator Inverse Kinematic Solutions Based on Vector Formulations and Damped Least-Squares Methods," *IEEE Transactions on Systems, Man, and Cybernetics* 16(1):93–101, 1986, DOI 10.1109/TSMC.1986.289285.
  - Chiaverini, Siciliano & Egeland, "Review of the damped least-squares inverse kinematics with experiments on an industrial robot manipulator," *IEEE Transactions on Control Systems Technology* 2(2):123–134, 1994, DOI 10.1109/87.294335.
- **Nakamura & Hanafusa's abstract:** the "singularity robust inverse (SR-inverse)" is introduced "as an alternative to the pseudoinverse of the Jacobian matrix". It gives "an approximating motion close to the desired Cartesian trajectory … even when the inverse kinematic solution by the inverse or the pseudoinverse … is not feasible at or in the neighborhood of singular points". — [Crossref abstract, DOI 10.1115/1.3143764](https://doi.org/10.1115/1.3143764)
- **Buss survey.** Samuel R. Buss, "Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods". The author's page lists it as an "Unpublished survey" whose analysis is "based on the singular value decomposition". There is no venue, DOI or year; cite it as an unpublished manuscript, commonly dated 2004 (year UNVERIFIED). Buss's companion work is published: Buss & Kim, "Selectively Damped Least Squares for Inverse Kinematics," *Journal of Graphics Tools* 10(3):37–49, 2005, DOI 10.1080/2151237X.2005.10129202 (Crossref). It adjusts damping per singular vector, while the plain pseudoinverse "oscillates badly and converges very poorly" in their demos. — [Buss IK methods page](https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/index.html)
- **Tedrake, *Robotic Manipulation* course notes, "Basic Pick and Place" chapter:**
  - The pseudo-inverse gives minimum-norm joint velocities for the 6×7 iiwa Jacobian, and it blows up as the smallest singular value goes to zero.
  - The notes reformulate diff-IK as a QP: least-squares on the spatial velocity error, with joint velocity bounds and position/acceleration limits via a one-step Euler approximation.
  - They add a null-space joint-centering objective and a pose-tracking mode in which the pose error divided by h becomes a commanded spatial velocity.
  - They add a "direction-preserving" variant with scalar α∈[0,1], in which the robot slows rather than deviating from the commanded direction.
  - Drake's DifferentialInverseKinematics is the implementation used in later chapters.
  - The notes do not use the term "damped least squares".
  
  — [manipulation.csail.mit.edu/pick.html](https://manipulation.csail.mit.edu/pick.html)
- **Drake `DoDifferentialInverseKinematics` documentation:**
  - It solves min −100·α + |P·(v_next − N⁺(q)·K·(q_nominal − q_current))|², subject to J·v_next = α·V and 0 ≤ α ≤ 1, plus joint position, velocity and acceleration limits and optional extra linear velocity constraints.
  - v_next = 0 is always feasible. Status becomes kStuck if α is very small.
  - The docs point to the newer `DifferentialInverseKinematicsSystem`, described as "more effective for real-world robots".
  
  — [Drake planning/kinematics docs](https://drake.mit.edu/doxygen_cxx/group__planning__kinematics.html)
- **DLS IK inside a landmark catching system:** in Kim et al. 2014, DS outputs are "converted into the 7-DOF joint state using the damped least squares IK", at 500 Hz joint-position control. — [Kim 2014](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)
- **PBVS as the controller in a handover system:** Rosenberger, Cosgun, Newbury, Kwan, Ortenzi, Corke & Grafinger, "Object-Independent Human-to-Robot Handovers using Real Time Robotic Vision," IEEE RA-L, accepted Sept. 2020 (arXiv 2006.01797; RA-L volume/pages UNVERIFIED).
  - "We use Position-Based Visual Servoing [Hutchinson et al. 1996] to drive the robot to the grasp pose and the velocity commands are sent to the low-level Panda robot controller."
  - Overall handover success was 81.9% over 40 handovers per object.
  - The authors say "a closed-loop approach will significantly reduce the safety stop and grasping failures".
  
  — [arXiv 2006.01797](https://arxiv.org/abs/2006.01797)
- **Reactive standoff-then-close handover (structurally like our 5 cm standoff then close):** Yang, Paxton, Mousavian, Chao, Cakmak & Fox, "Reactive Human-to-Robot Handovers of Arbitrary Objects," arXiv 2011.08961 (venue ICRA 2021, UNVERIFIED).
  - Riemannian Motion Policies drive the robot to a grasp standoff pose updated "at roughly 10Hz".
  - On arrival it runs "an open-loop grasp action".
  - Success was 100% for 0–45° object rotations and 69% average for 0–90° rotations (Table I).
  
  — [arXiv 2011.08961](https://arxiv.org/abs/2011.08961)
- **MPC vs reactive baseline in handover:** Yang, Sundaralingam, Paxton, Akinola, Chao, Cakmak & Fox, "Model Predictive Control for Fluid Human-to-Robot Handovers," arXiv 2204.00134 (venue UNVERIFIED).
  - Prior handover systems drive the end-effector with "local (non-global) policies such as Riemannian Motion Policies … or visual servoing".
  - Stochastic (sampling-based) MPC with a 2 s horizon runs at 50–100 Hz and approaches a standoff 15 cm from the grasp.
  - Against the Yang et al. baseline, approach time drops from 11.5 s to 7.1 s and jerk is lower.
  
  — [arXiv 2204.00134](https://arxiv.org/abs/2204.00134)
- **Benchmark baselines for learned handover policies:** Chao et al., "HandoverSim: A Simulation Framework and Benchmark for Human-to-Robot Object Handovers," arXiv 2205.09747 (venue UNVERIFIED).
  - Baselines are the OMG motion/grasp planner (62.50% success in the default setup), the Yang et al. reactive system, and GA-DDPG RL (36.81% without hold, failing because it "was never trained to adapt to moving objects").
  - The simulator uses "a simple PD controller".
  
  — [arXiv 2205.09747](https://arxiv.org/abs/2205.09747)
- **Learned point-cloud handover policy evaluated against those baselines:** Christen, Yang, Pérez-D'Arpino, Hilliges, Fox & Chao, "Learning Human-to-Robot Handovers from Point Clouds," arXiv 2303.17592. They note "prior work typically relies on open-loop motion planners" for demonstrations. — [arXiv 2303.17592](https://arxiv.org/abs/2303.17592)
- **Visual servoing as an explicit baseline for a learned RGB policy:** Wu, Pang, Cavallaro & Oh, "Learning human-to-robot handovers through 3D scene reconstruction," arXiv 2507.08726 (July 2025). A search-engine summary of the PDF says they implemented an IBVS baseline using object masks ("IBVS-MASK", centroid-to-image-center servoing) because no direct baseline existed. My WebFetch of the arXiv abstract page did not show this, so the IBVS-baseline detail is UNVERIFIED. — [arXiv 2507.08726](https://arxiv.org/abs/2507.08726)

### Inferences
- **A "DLS/QP resolved-rate servo + KF prediction + velocity feedforward" is a defensible classical baseline.** Each component has a landmark citation:
  - Whitney 1969 for resolved rate.
  - Nakamura & Hanafusa 1986 and Wampler 1986 for DLS, plus Chiaverini et al. 1994 for the review.
  - Chaumette & Hutchinson 2007 for moving-target feedforward.
  - Allen et al. 1993 for predictive filtering when grasping moving objects.
  - Tedrake notes and Drake for the QP diff-IK.
  
  Near-identical architectures appear in published handover systems (Rosenberger 2020 PBVS; Yang 2021 standoff tracking then close).
- **Learned-policy papers rarely compare against pure diff-IK.** They compare against motion planners (OMG), reactive RMP systems, IBVS, or MPC. To pre-empt a "straw-man baseline" critique, we should frame our baseline as PBVS-style and ideally add (or at least discuss) an intercept-aware variant: choose a reachable rendezvous point/time on the predicted partner path, as Bäuml/Kim/Croft do, or a short-horizon MPC.
- **Drake's direction-preserving formulation** (J v = αV, α∈[0,1]) can silently slow the arm when V is infeasible. With a moving target, this produces lag that velocity feedforward cannot recover. Log α (or the DLS residual) as a diagnostic.

### Gaps
- Exact RA-L volume/pages for Rosenberger 2020 and conference venues for arXiv 2011.08961, 2204.00134 and 2205.09747 were not verified against publisher listings (DBLP returned HTTP 202/no data during this session).
- I found no paper that uses exactly "DLS diff-IK + KF look-ahead" as *the* named baseline against an end-to-end pixel RL policy. Closest are the IBVS baseline (Wu et al. 2025, unverified detail) and MPC-vs-RL (Abeyruwan et al. 2023, Q4).

---

## Q4. Interception planning theory: time-optimal interception, rendezvous selection, reachability/timing, MPC

### Takeaway
Classical interception theory separates rendezvous-point/time selection, constrained by reachability and arm dynamics (Croft, Fenton & Benhabib), from trajectory generation. Modern work folds prediction, catch-pose selection and motion into a single (time-optimal or nonlinear) MPC (Gold et al. 2022; NMPC 2025). Abeyruwan et al. 2023 is the clearest direct comparison of whole-body MPC against blackbox RL on a catching task. It is a good model for how to compare our classical baseline to a learned policy.

### Cited Findings
- **Croft, Fenton & Benhabib (Crossref records):**
  - "Time-optimal interception of objects moving along predictable paths," *Proc. IEEE International Symposium on Assembly and Task Planning*, 1995, pp. 419–425, DOI 10.1109/ISATP.1995.518804.
  - "Time-optimal interception of objects moving along topologically varying paths," *1995 IEEE SMC*, vol. 5, pp. 4089–4094, DOI 10.1109/ICSMC.1995.538431.
  - "Optimal rendezvous-point selection for robotic interception of moving objects," *IEEE Transactions on Systems, Man, and Cybernetics, Part B* 28(2):192–204, Apr. 1998, DOI 10.1109/3477.662759.
  
  Content not read; the titles indicate time-optimal intercept and rendezvous-point selection.
- **Bäuml et al. 2010** solve where/when/how jointly as an NLP over (q_c, t_c) under joint position/velocity and work-cell constraints. In a ~1 s flight, only ~0.1 s goes to accelerating/decelerating and ~80% of the time is at maximum joint velocity. So timing is dominated by velocity limits. — [DLR elib](https://elib.dlr.de/67146/1/B%C3%A4uml2010.pdf)
- **Kim et al. 2014** choose the catch point probabilistically (reachable × graspable GMM likelihood along the predicted trajectory). Earlier works used intersections with a crude reachable boundary, closest to the base or initial end-effector position. — [Kim 2014](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)
- **Gold, Römer, Völz & Graichen**, "Catching Objects with a Robot Arm using Model Predictive Control," *2022 American Control Conference (ACC)*, pp. 1915–1920, DOI 10.23919/ACC53348.2022.9867380 (Crossref). Per a search-engine summary of the IEEE Xplore page: a 7-DOF arm, with trajectory prediction, catch-pose selection and motion planning folded into one time-optimal MPC solved with GRAMPC, validated in simulation and on hardware. Content is UNVERIFIED against full text. — [IEEE Xplore 9867380](https://ieeexplore.ieee.org/document/9867380)
- **Abeyruwan, Bewley, Boffi, Choromanski, D'Ambrosio, Jain, Sanketi, Shankar, Sindhwani, Singh, Slotine & Tu**, "Agile Catching with Whole-Body MPC and Blackbox Policy Learning," arXiv 2306.08205 (comment: L4DC 2023).
  - It compares accelerated constrained trajectory-optimization MPC against zeroth-order (blackbox) RL for catching thrown objects, using visual object observations plus proprioception.
  - Comparison axes are sample efficiency, sim-to-real transfer, robustness to distribution shift, and whole-body multimodality.
  - It proposes combining the two.
  - The abstract gives no success numbers.
  
  — [arXiv 2306.08205](https://arxiv.org/abs/2306.08205)
- **Cai, Kohli & Liu**, "NMPC-based Motion Planning with Adaptive Weighting for Dynamic Object Interception," arXiv 2511.15532 (Nov. 2025). It is a cooperative dual-arm NMPC that couples interception planning with real-time joint-space control. Per search summary it was submitted to the IFAC World Congress (under review). Details UNVERIFIED. — [arXiv 2511.15532](https://arxiv.org/abs/2511.15532)
- **Rationale for the 0.1 s look-ahead:** Bäuml et al. explicitly compensate a measured 90 ms camera-to-motion latency in the prediction. Kim et al. stop updating the catch posture when time-to-contact is <0.09 s. — [DLR elib](https://elib.dlr.de/67146/1/B%C3%A4uml2010.pdf); [Kim 2014](https://infoscience.epfl.ch/record/198748/files/TRO2316022.pdf?version=1)

### Inferences
- **A principled upgrade of our baseline** is to replace the fixed 0.1 s look-ahead with an intercept time t* chosen to minimize arrival time, or to maximize a feasibility margin, subject to the iiwa's joint velocity limits. A Croft/Bäuml-style rendezvous search over the KF-predicted min-jerk path, or a small MPC, would do this. Our 0.1 s look-ahead is best justified as latency compensation (cf. Bäuml's 90 ms), not as intercept planning.
- **Because the partner follows a min-jerk reach,** its future path is highly predictable once the endpoint is estimated. A min-jerk-model predictor could beat a constant-acceleration KF over horizons >0.1 s. That would make rendezvous planning more effective than in ballistic catching. This is an inference; I found no paper testing it in this exact setting.
- **For the RL-vs-classical comparison,** Abeyruwan et al. 2023 is the closest methodological template: same task, MPC vs RL, with robustness and distribution-shift analyses.

### Gaps
- Full text of the Croft et al. papers, Gold et al. 2022 and arXiv 2511.15532 was not read. Their algorithms and results are not verified beyond titles and abstract-level summaries.
- I found no landmark work on *antiparallel palm-to-palm* interception specifically. The nearest are soft catching (palm opposite object velocity; Salehian 2016) and handover standoff-then-approach (Yang 2021/2022).
- DBLP API calls returned HTTP 202 with no data in this session, so DBLP cross-checks were not possible. Crossref (the DOI registry) was used instead for all bibliographic metadata.
