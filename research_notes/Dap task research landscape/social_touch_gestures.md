# Robots performing social physical touch gestures (clapping, high-fives, fist bumps, handshakes, hugs)

Scope note: notes for the dap project (KUKA iiwa7 in Drake, kinematic; meet a moving min-jerk palm palm-to-palm and antiparallel; head-mounted depth camera; end-to-end RL from depth vs. a reactive diff-IK servo baseline). Each citation was checked against a primary or bibliographic listing (publisher page, Crossref, DBLP, arXiv, or the authors' institute page) unless marked UNVERIFIED or PARTIAL. "PARTIAL" means the bibliographic record was confirmed but technical details came from a secondary summary, not the paper body.

## Q1. Hand-clapping robots (Fitter & Kuchenbecker): sensing, timing, controller, evaluation

### Takeaway
Fitter and Kuchenbecker's Baxter line (2016 to 2020) is the landmark work on robot hand-clapping. The robot played rhythmic games on keyframed position-controlled arm motions at a fixed or adaptive tempo. It sensed the human through wrist-worn IMUs and hand contact (no vision). Evaluation combined user ratings (pleasantness, energeticness, dominance, safety), recognition accuracy, and tempo-synchrony behaviour. Tempo and synchrony mattered most to users, and stiffer arms felt safer.

### Cited Findings
**Bibliography (verified)**
- Fitter, N. T., & Kuchenbecker, K. J. "Equipping the Baxter robot with human-inspired hand-clapping skills." 2016 25th IEEE Int. Symp. on Robot and Human Interactive Communication (RO-MAN), pp. 105–112, 2016. DOI 10.1109/ROMAN.2016.7745097. — [Crossref record](https://api.crossref.org/works/10.1109/ROMAN.2016.7745097); [MPI-IS listing](https://is.mpg.de/publications/fitter16-roman-baxter)
  - The abstract and methods of this RO-MAN paper could not be retrieved (the MPI page served a bot-check, and Crossref has no abstract). Its technical details are listed under Gaps.
- Fitter, N. T., Hawkes, D. T., & Kuchenbecker, K. J. "Rhythmic Timing in Playful Human-Robot Social Motor Coordination." ICSR 2016, LNCS vol. 9979, pp. 296–305. — [MPI-IS listing](https://hi.is.mpg.de/publications/fitter16-icsr-timing); [DBLP](https://dblp.dagstuhl.de/rec/conf/socrob/FitterHK16.html)
- Fitter, N. T., & Kuchenbecker, K. J. "Qualitative User Reactions to a Hand-Clapping Humanoid Robot." ICSR 2016, pp. 317–327. — [MPI-IS listing](https://is.mpg.de/de/publications/fitter16-icsr-clapping) (PARTIAL: only the title and pages were seen)
- Fitter, N. T., & Kuchenbecker, K. J. "Using IMU Data to Demonstrate Hand-Clapping Games to a Robot." IROS 2016. — [MPI-IS listing](https://hi.is.tuebingen.mpg.de/publications/fitter16-iros-imu) (PARTIAL: pages and DOI not seen)
- Fitter, N. T., & Kuchenbecker, K. J. "Teaching a Robot Bimanual Hand-Clapping Games via Wrist-Worn IMUs." Frontiers in Robotics and AI 5:85, 2018. DOI 10.3389/frobt.2018.00085. — [Frontiers](https://www.frontiersin.org/articles/10.3389/frobt.2018.00085)
- Fitter, N. T., & Kuchenbecker, K. J. "How Does It Feel to Clap Hands with a Robot?" International Journal of Social Robotics 12(1):113–127. DOI 10.1007/s12369-019-00542-x. Published online 2019; the print issue is dated 2020. — [DBLP](https://dblp.org/rec/journals/ijsr/FitterK20.html); [MPI-IS](https://is.mpg.de/publications/fitter-ijsr-clap)
- An ISRR 2017 paper, later published as a chapter in *Robotics Research* (Springer, 2020; DOI prefix 10.1007/978-3-030-28619-4_24), reports Baxter playing one-handed tempo-matching clapping games: "Twenty adults tested two clapping tempos in each of these four interaction modes." — [MPI-IS listing](https://is.mpg.de/hi/publications/fitter17-isrr-synchronicity) (PARTIAL: the exact title is believed to be "Synchronicity Trumps Mischief in Rhythmic Human-Robot Social-Physical Interaction" but was not confirmed on a primary page; mark the title UNVERIFIED)

**Sensing, control and timing**
- Robot: Rethink Baxter with non-articulated 3D-printed hands covered in silicone rubber. Arm motions were taught kinesthetically as key poses (preparatory and action), then played back with the Baxter SDK raw position controller and cubic interpolation between key poses. — [Frontiers 2018](https://www.frontiersin.org/articles/10.3389/frobt.2018.00085)
- Human sensing (2018): two 9-axis MPU9150 IMUs strapped to the wrists. Accelerometer and gyroscope data streamed at 200 Hz through a Teensy. The magnetometer was dropped as unreliable indoors. — [Frontiers 2018](https://www.frontiersin.org/articles/10.3389/frobt.2018.00085)
- Timing (2018): demonstrations followed a 75 BPM metronome. Gyroscope thresholds segmented each demonstration, and local maxima of a high-pass-filtered RMS acceleration located each motion. A linear SVM classified 10 motion primitives. — [Frontiers 2018](https://www.frontiersin.org/articles/10.3389/frobt.2018.00085)
- Adaptive tempo (ICSR 2016): the authors observed human-human entrainment, modelled synchronization behaviour, and piloted three adaptive-tempo behaviours on Baxter. A "fading memory difference learning timing model" looked best. — [MPI-IS listing](https://hi.is.mpg.de/publications/fitter16-icsr-timing)
- Fitter's own description: no computer vision and no Baxter cameras were used, yet participants praised how well the robot "tracked" their hand. Physical reactivity meant updating "its plan of where to move next after sensing hand contact." Stiffness was a variable that controlled how quickly the motors pulled the arm to the desired position. — [IEEE Spectrum, Fitter, 11 Jul 2019](https://spectrum.ieee.org/how-high-fives-help-us-get-in-touch-with-robots)
- Fitter called the interaction "somewhere between a high five and a hand-clapping game." — [IEEE Spectrum](https://spectrum.ieee.org/how-high-fives-help-us-get-in-touch-with-robots)

**Evaluation**
- IJSR study: 20 participants, 24 conditions crossing facial reactivity, physical reactivity, arm stiffness and clapping tempo. Every participant completed the study, and each mentioned enjoying at least one game without being asked.
  - Tempo was highly salient to users.
  - Human-like robot errors were accepted more readily than mechanical errors.
  - Facial reactivity raised perceived pleasantness and energeticness. Physical reactivity lowered pleasantness, energeticness and dominance.
  - Source: [IJSR listing / summaries](https://is.mpg.de/publications/fitter-ijsr-clap); [DBLP](https://dblp.org/rec/journals/ijsr/FitterK20.html)
- A stiffer robot was rated safer and less dominant, which ran against the author's expectation. Safety ratings were generally high. — [IEEE Spectrum](https://spectrum.ieee.org/how-high-fives-help-us-get-in-touch-with-robots)
- Frontiers 2018 results:
  - Offline classifier: 97.3% training accuracy and 97.0% test accuracy on unseen participants, using leave-one-subject-out cross-validation.
  - Live study: 85.9% accuracy. 24 participants were recruited and 20 analysed. The authors blame the drop mainly on timing variation.
  - Safety rating: M = 79.71, SD = 21.59.
  - Robot-led blocks received higher performance ratings (p < 0.001).
  - Source: [Frontiers 2018](https://www.frontiersin.org/articles/10.3389/frobt.2018.00085)

### Inferences
- Classical clapping systems did not solve the dap's perception problem. Baxter never visually tracked the human hand. It ran keyframed motions on a metronome or adaptive beat and relied on the rhythm, so the human largely came to the robot. Your dap (meeting a visually observed, non-rhythmic min-jerk reach) is a harder sensing and timing problem than the clapping literature tackled.
- "Tempo/synchrony is salient" and "human-like errors are tolerated" suggest timing error at contact is a meaningful metric to report next to spatial and orientation error.
- The Spectrum quote (stiffer equals safer) concerns compliance. That is out of scope for a kinematic simulation, but worth naming as a limitation.

### Gaps
- RO-MAN 2016 methods (contact sensing in the hand, the controller, tempo source, and whether it used Baxter's wrist force or accelerometer) are unverified. The full text could not be retrieved.
- The IJSR paper's exact stiffness values and tempos (BPM) were not retrieved.

## Q2. High-five / fist-bump robots and interaction primitives (learned coupling of human and robot motion)

### Takeaway
The dominant learned approach for "respond to the partner's reaching gesture" is the interaction-primitive family: learn a joint distribution over human and robot trajectories from demonstrations, then condition on the partially observed human motion. The family runs from Interaction Primitives (ICRA 2014), through ProMP mixtures (AuRo 2017) and Bayesian IPs (CoRL 2017), to deep latent variants (Bütepage et al. 2019; MILD 2022). High-fives and fist bumps appear as benchmark tasks. These methods predict trajectories and are evaluated by trajectory error (MSE). Contact success is not their headline metric.

### Cited Findings
- Ben Amor, H., Neumann, G., Kamthe, S., Kroemer, O., & Peters, J. "Interaction primitives for human-robot cooperation tasks." 2014 IEEE ICRA, pp. 2831–2837. DOI 10.1109/ICRA.2014.6907265. — [Crossref](https://api.crossref.org/works/10.1109/ICRA.2014.6907265); [CMU RI listing](https://publications.ri.cmu.edu/interaction-primitives-for-human-robot-cooperation-tasks)
  - Method: keep a distribution over DMP parameters learned from two humans performing the task. This captures correlations so the robot can "infer the behavior of the human partner and participate in the cooperation." The paper also gives algorithms to synchronize and adapt human and robot behaviour. — [CMU RI listing](https://publications.ri.cmu.edu/interaction-primitives-for-human-robot-cooperation-tasks)
  - High-five: later material ties the IP approach to a humanoid high-fiving with a human partner, and Ben Amor's talks use the high-five as the running example. — search summaries of [ASU / TUM workshop slides](https://www.ce.cit.tum.de/fileadmin/w00cgn/ics/pics/IROS2017/IROS2018-SPAR_Amor.pdf) (PARTIAL: whether the ICRA 2014 paper itself contains a high-five experiment, and its sensing, phase estimation and metrics, could not be confirmed; the PDF exceeded the fetch limit)
- Maeda, G. J., Neumann, G., Ewerton, M., Lioutikov, R., Kroemer, O., & Peters, J. "Probabilistic movement primitives for coordination of multiple human–robot collaborative tasks." Autonomous Robots 41(3):593–612, 2017. DOI 10.1007/s10514-016-9556-2. — [MPI-IS](https://is.mpg.de/publications/maedaetal17); [TU Darmstadt manuscript](https://www.ias.tu-darmstadt.de/uploads/Team/PubGJMaeda/gjm_2016_AURO_c.pdf)
  - Method: a mixture of interaction ProMPs learned from demonstrations does both action recognition and inference of the robot trajectory from observed human motion.
  - Evaluation: a lightweight arm doing a bottle handover and collaborative toolbox assembly. These are not touch gestures. — [search summary of CMU/TUD/MPI listings](https://www.ri.cmu.edu/publications/probabilistic-movement-primitives-for-coordination-of-multiple-human-robot-collaborative-tasks)
- Campbell, J., & Ben Amor, H. "Bayesian Interaction Primitives: A SLAM Approach to Human-Robot Interaction." CoRL 2017, PMLR vol. 78, pp. 379–387. — [PMLR](https://proceedings.mlr.press/v78/campbell17a.html)
  - Treats phase (timing) as the "location" and the primitive weights as the "map" in an EKF-SLAM formulation. This estimates timing and trajectory jointly, where earlier IPs handled them separately. — [authors' slides](https://interactive-robotics.engineering.asu.edu/wp-content/uploads/sites/54/2017/11/campbell_corl17_presentation.pdf)
  - Open-source library: IntPrim (ASU). — [ASU page](https://interactive-robotics.engineering.asu.edu/interaction-primitives/)
- Vogt, D., Stepputtis, S., Grehl, S., Jung, B., & Ben Amor, H. "A system for learning continuous human-robot interactions from human-human demonstrations." ICRA 2017. Imitation learning from motion-captured human-human pairs. — [ASU lab page](https://interactive-robotics.engineering.asu.edu/2017/02/01/a-system-for-learning-continuous-human-robot-interactions-from-human-human-demonstrations); [researchr](https://researchr.org/publication/VogtSGJA17) (PARTIAL: the DOI and the task list, including whether high-five is among them, were not verified)
- Bütepage, J., Ghadirzadeh, A., Öztimur Karadağ, Ö., Björkman, M., & Kragic, D. "Imitating by generating: deep generative models for imitation of interactive tasks." arXiv:1910.06031, 2019. — [arXiv](https://arxiv.org/abs/1910.06031)
  - Tasks: hand-shake, hand-wave, parachute fist-bump and rocket fist-bump.
  - Pipeline: embed human and robot motion, predict the human, then generate robot joint trajectories.
  - Publication: the arXiv page lists no journal; a later venue (possibly Frontiers in Robotics and AI 2020) is UNVERIFIED.
- Prasad, V., Koert, D., Stock-Homburg, R., Peters, J., & Chalvatzaki, G. "MILD: Multimodal Interactive Latent Dynamics for Learning Human-Robot Interaction." IEEE-RAS Humanoids 2022. arXiv:2210.12418; DOI 10.1109/Humanoids53995.2022.10000239. — [arXiv](https://arxiv.org/abs/2210.12418)
  - Model: HSMMs in the latent space of a VAE, learned from human-human demonstrations. It "can learn directly from camera-based pose estimations." — [arXiv](https://arxiv.org/abs/2210.12418)
  - Benchmarks — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2210.12418):
    - Bütepage et al. data: wave, handshake, rocket and parachute fist-bumps; ABB YuMi for the HRI portion.
    - NuiSI data, collected by the authors with a RealSense D435 and Nuitrack skeletons: handshake, wave, high five, fist bump, clap-fist and rocket fist-bump.
    - Demo robot: Pepper.
  - Metric: MSE of the predicted controlled-agent motion over a 1 s window (40 observations). Example: YuMi handshake 0.038 ± 0.035 rad for MILD vs. 0.371 ± 0.224 rad for Bütepage et al. NuiSI high five: 0.293 ± 0.191 cm. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2210.12418)
  - A follow-up journal-style version exists: "Learning Multimodal Latent Dynamics for Human-Robot Interaction," arXiv:2311.16380. — [arXiv PDF](https://arxiv.org/pdf/2311.16380) (venue UNVERIFIED)

**High-five-specific user studies**
- I found no dedicated, verified high-five user study on Pepper, NAO or Baxter that measures contact timing. The closest are:
  - Fitter's Baxter clapping study (Q1), described as between a high five and a clapping game. — [IEEE Spectrum](https://spectrum.ieee.org/how-high-fives-help-us-get-in-touch-with-robots)
  - "Hey, robot! An investigation of getting robot's attention through touch" (NAO with 26 participants, Pepper with 28). For both robots, the preferred response to a touched hand was "moving back." — [arXiv 2209.00414](https://arxiv.org/pdf/2209.00414) (authors and venue UNVERIFIED)

### Inferences
- IP/ProMP methods answer "what trajectory should the robot run given a partial view of the partner's reach." That is exactly the dap's prediction subproblem, and these are natural learned-prediction baselines sitting between a reactive diff-IK servo and end-to-end RL.
- These methods typically rely on motion capture or skeleton tracking, not raw depth pixels, and their reported metric is trajectory MSE, not contact or pose success at meeting time. A dap evaluation built on ground-truth contact, antiparallel-orientation error and timing error would fill a real gap in this literature.
- Bayesian IPs' joint phase estimation is the closest classical analogue to inferring min-jerk progress, or time-to-contact, from a partial reach.

### Gaps
- Whether the ICRA 2014 IP paper contains a high-five experiment, and its sensing and metrics, are not verified from the full text.
- No verified Okamura high-five work was found. The "Okamura" lead from the brief may be a misremembering, and I would treat it as UNVERIFIED.

## Q3. Robot handshakes: models, control, timing/synchrony

### Takeaway
Handshake work splits into two streams:
- Approach-phase timing models. Jindai and Watanabe's handshake robots shape the robot's reach as a second-order lag plus dead time driven by the human hand's trajectory, and evaluate with subjective sensory evaluation.
- Contact-phase haptic or impedance models. These are judged by human-likeness tests, such as Avraham et al.'s Turing-like handshake test.

Prasad et al.'s IJSR review is the standard survey. It stresses synchronization across handshake phases and multimodal cues such as gaze, voice and face.

### Cited Findings
- Jindai, M., Watanabe, T., Shibata, S., & Yamamoto, T. "Development of a Handshake Robot System Based on a Handshake Approaching Motion Model." J. Robotics and Mechatronics 20(4):650–659, 2008. DOI 10.20965/jrm.2008.p0650. — [Fuji Press](https://www.fujipress.jp/jrm/rb/robot002000040650/)
  - The model generates an approaching motion acceptable to the human using "second-order lag and dead time elements" applied to the human hand's trajectory, derived from analysis of human-human handshakes with and without voice greetings. Effectiveness was shown by sensory evaluation. — [Fuji Press](https://www.fujipress.jp/jrm/rb/robot002000040650/)
- Ota, S., Jindai, M., Fukuta, T., & Watanabe, T. "Small-sized handshake robot system for generation of handshake behavior with active approach to human." J. Advanced Mechanical Design, Systems, and Manufacturing 13(2), JAMDSM0026, 2019. DOI 10.1299/jamdsm.2019jamdsm0026. — [J-STAGE](https://www.jstage.jst.go.jp/article/jamdsm/13/2/13_2019jamdsm0026/_article/-char/en)
  - Adds a response-motion model so the robot can answer a handshake the human requests mid-approach, plus switching control between request and response motions based on the human's behaviour. Validated by sensory evaluation. — [J-STAGE](https://www.jstage.jst.go.jp/article/jamdsm/13/2/13_2019jamdsm0026/_article/-char/en)
- Avraham, G., Nisky, I., Fernandes, H. L., Acuna, D. E., Kording, K. P., Loeb, G. E., & Karniel, A. "Toward Perceiving Robots as Humans: Three Handshake Models Face the Turing-Like Handshake Test." IEEE Transactions on Haptics 5(3):196–207, 2012. DOI 10.1109/TOH.2012.16. — [HUJI CRIS](https://cris.huji.ac.il/en/publications/toward-perceiving-robots-as-humans-three-handshake-models-face-th/); [BGU PDF](https://in.bgu.ac.il:443/en/engn/biomed/CMCL/Site%20Assets/Pages/Publications/Avraham2012.pdf) (author list as I recall it; the listings confirmed Avraham et al., the title, venue and DOI, but the full author order was not re-read and should be checked)
  - Models compared: Tit-for-Tat, a λ model, and a machine-learning model (iML-shake), each played through a haptic device. Human-likeness was measured as how much noise had to be mixed into a real human handshake to make it as distinguishable as the model's. Tit-for-Tat and iML-shake were judged most human-like. — [search summary of the TOH abstract](https://cris.iucc.ac.il/en/publications/toward-perceiving-robots-as-humans-three-handshake-models-face-th/)
- Vanello, N., Bonino, D., Ricciardi, E., et al. "Neural correlates of human-robot handshaking." RO-MAN 2010. DOI 10.1109/ROMAN.2010.5598624. — [DBLP](https://dblp.org/rec/conf/ro-man/VanelloBRTSHTZCGPRL10)
  - An fMRI paradigm studying the visual and haptic components of human-human vs. human-robot handshakes, with a sensing glove; results described as preliminary. The page range is inconsistent across listings (520–526 vs. 555–561). — [dissem.in / UniPi listings](https://arpi.unipi.it/handle/11568/208021)
- Prasad, V., Stock-Homburg, R., & Peters, J. "Human-Robot Handshaking: A Review." International Journal of Social Robotics, 2021 (online). DOI 10.1007/s12369-021-00763-z; arXiv:2102.07193. — [arXiv](https://arxiv.org/abs/2102.07193)
  - Finds that synchronization occurs across handshake phases, that gaze, voice and facial expression change how a robotic handshake is perceived, and that personality and mood affect handshakes. — [arXiv abstract](https://arxiv.org/abs/2102.07193)
  - Note: the brief's "Prasad et al. survey" is this paper. There is no Kumar or Kuchenbecker co-author.

### Inferences
- Jindai's second-order-lag-plus-dead-time approach is in effect a hand-designed reactive servo on the human hand trajectory, the classical analogue of the dap's diff-IK servo baseline. It is a citable precedent that reactive tracking with a delay is the "human-acceptable" baseline for approach timing.
- Handshake evaluation is mostly perceptual (sensory evaluation, Turing-like tests). Kinematic success metrics are rare, so a simulation-only dap study should be explicit that it measures geometric and timing success, not perceived naturalness.

### Gaps
- Sensing modality in the Jindai systems was not confirmed from the full text.
- Impedance-control handshake papers were not individually verified in this pass. Examples are Wang, Peer & Buss's HMM-based haptic handshake (World Haptics 2009) and Kasuga & Hashimoto. Treat them as UNVERIFIED.
- Prasad et al.'s ICRA 2021 "Learning human-like hand reaching for human-robot handshaking" is UNVERIFIED; it was not checked in this pass.

## Q4. Hugging robots (HuggieBot) as reactive touch

### Takeaway
Block and Kuchenbecker's HuggieBot series is the reference for closed-loop social touch. Version 1 (PR2) showed softness, warmth and responsive release matter. HuggieBot 2.0 added visual hug initiation and haptic sizing and release. HuggieBot 3.0 adds intra-hug gesture perception (88% accuracy) and a probabilistic response policy. Evaluation was within-subject user ratings (n = 32, n = 16).

### Cited Findings
- Block, A. E., & Kuchenbecker, K. J. "Softness, Warmth, and Responsiveness Improve Robot Hugs." International Journal of Social Robotics 11(1):49–64. DOI 10.1007/s12369-018-0495-2. Online 2018, issue 2019. — [DBLP](https://dblp.uni-trier.de/rec/journals/ijsr/BlockK19.html); [ETH Research Collection](https://research-collection.ethz.ch/handle/20.500.11850/306705?show=full)
  - A PR2 covered in foam, fabric and heating pads. Users preferred soft and warm hugs, and hugs that squeeze and "release immediately when they are ready for the hug to end." — [search summary of the IJSR abstract](https://is.mpg.de/publications/block18-ijsr-improve)
- Block, A. E., Christen, S., Gassert, R., Hilliges, O., & Kuchenbecker, K. J. "The Six Hug Commandments: Design and Evaluation of a Human-Sized Hugging Robot with Visual and Haptic Perception." HRI '21, pp. 380–388. DOI 10.1145/3434073.3444656; arXiv:2101.07679. — [arXiv](https://arxiv.org/abs/2101.07679)
  - HuggieBot 2.0: soft, warm, inflated "HuggieChest."
  - Visual hug initiation and haptic sizing and release give closed-loop hugging.
  - 32 users × 8 hugs. Haptic reactivity "definitively improves user perception."
  - Tenets: soft, warm, human-sized, visually perceives the user, adjusts the embrace to user size and position, reliably releases.
  - Source: [arXiv](https://arxiv.org/abs/2101.07679)
- Block, A. E., Seifi, H., Hilliges, O., Gassert, R., & Kuchenbecker, K. J. "In the Arms of a Robot: Designing Autonomous Hugging Robots with Intra-Hug Gestures." ACM Transactions on Human-Robot Interaction 12(2), 2023 (online Feb 2022). DOI 10.1145/3526110. — [Crossref](https://api.crossref.org/works/10.1145%2F3526110); [DeepAI abstract](https://deepai.org/publication/in-the-arms-of-a-robot-designing-autonomous-hugging-robots-with-intra-hug-gestures)
  - 32 users × 16 hugs (experimenter-controlled), covering hold, rub, pat and squeeze.
  - The perception algorithm classifies user gestures with 88% accuracy.
  - A probabilistic behaviour algorithm picks responses from average user ratings.
  - HuggieBot 3.0 was validated with 16 users.
  - Source: [DeepAI abstract](https://deepai.org/publication/in-the-arms-of-a-robot-designing-autonomous-hugging-robots-with-intra-hug-gestures) (the issue number "2" is UNVERIFIED; Crossref confirmed volume 12 and the 2023 date)

### Inferences
- The HuggieBot 2.0 design, with vision to initiate and to size the embrace to the user's pose, is the closest social-touch precedent for the dap's "see the human with a camera, then shape the robot's pose to meet them." The paper's "visually perceive its user" and "adjust to user position" tenets are a citable rationale for vision-driven pose adaptation.

### Gaps
- HuggieBot 2.0's visual pipeline (camera type, pose estimator) was not retrieved in detail.

## Q5. Learned approaches (RL, imitation, sim benchmarks) for these gestures

### Takeaway
The closest prior work to "deep RL for a hand-meeting social gesture in simulation" is Christen, Stevšić & Hilliges (ICRA 2019). They trained a DDPG policy for a Shadow Hand on a 4-DoF arm to handshake, hand-clap and fingertip-touch in simulation. Rewards were parameterized from human-human mocap, and evaluation used success curves plus a 116-person video-rating study. It used a static (or simply moving) target hand state as input, not pixels. No real-robot transfer.

Other learned work, covered in Q2, is imitation or probabilistic prediction (IPs, ProMPs, Bütepage et al., MILD). I found no standard simulation benchmark specifically for high-five or dap gestures.

### Cited Findings
- Christen, S., Stevšić, S., & Hilliges, O. "Demonstration-Guided Deep Reinforcement Learning of Control Policies for Dexterous Human-Robot Interaction." ICRA 2019. arXiv:1906.11695. — [arXiv](https://arxiv.org/abs/1906.11695v1); [DBLP (CoRR)](https://dblp.uni-trier.de/rec/journals/corr/abs-1906-11695.html) (the ICRA DOI was not retrieved)
  - Setup: 4-DoF arm plus Shadow Dexterous Hand (24 DoF, 20 actuators) in simulation, with robot models from OpenAI Gym; the physics engine is not named in the text. The target hand is an unactuated copy posed from human mocap. — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/1906.11695)
  - Observations: agent joint angles, velocities and contact sensors, plus target-hand link positions. Actions: joint-angle setpoints in [-1, 1]. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1906.11695)
  - Algorithm: DDPG with 3 × 256 ReLU layers. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1906.11695)
  - Reward: final-state reward plus optional imitation reward.
    - Final-state terms: fingertip-position error relative to the target hand, joint-angle error, a contact bonus over 15 sensors matching the desired contact profile, and an action penalty.
    - Weights are set from OptiTrack mocap of human-human interactions (5 demonstrations per interaction) at the closest-approach frame.
    - Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/1906.11695)
  - Timing: a goal-centric formulation. The target pose is randomized and held fixed per episode, and the robot wrist is initialized from an earlier time step of the demonstration plus noise. Robustness was also tested with moving targets (constant speed, direction changes every 0.5 s). — [ar5iv](https://ar5iv.labs.arxiv.org/html/1906.11695)
  - Evaluation — [ar5iv](https://ar5iv.labs.arxiv.org/html/1906.11695):
    - Success-rate curves over 5 seeds. The full reward beats variants that use palm position only, or that drop the contact reward. The imitation term did not change success.
    - Video user study with N = 116 on a −2..+2 preference scale favouring imitation: clap +1.44, handshake +0.26, E.T. touch +0.07.
    - Stated limitations: simulation only, static starting target, no force modelling.
- MILD, Bütepage et al., Maeda et al., and Campbell & Ben Amor (Q2) are learned-from-demonstration predictors, not RL controllers. — see Q2 sources.
- A December 2025 arXiv paper, "ContactRL" (arXiv:2512.03707), uses RL with contact force for safe human-robot contact. It is not gesture-specific. — [arXiv](https://arxiv.org/pdf/2512.03707v1) (authors and venue UNVERIFIED; seen only in a search snippet)

### Inferences
- Christen et al. 2019 is the most direct precedent for the dap's RL arm and should be cited as such. The dap differs in four ways:
  - Arm-level (iiwa7) rather than hand-dexterity.
  - A partner that moves with a min-jerk profile throughout the episode, rather than a mostly static goal.
  - Depth-pixel observations rather than privileged target-hand state.
  - Ground-truth geometric and timing success rather than video ratings.
- Christen et al. found that removing the contact reward hurt success, which is evidence that an explicit contact or proximity term in the reward matters for touch-meeting tasks.

### Gaps
- No verified end-to-end RL from pixels (depth or RGB) for social touch gestures was found. This appears to be an open niche (absence of evidence from roughly 30 searches, not proof).
- No standard simulation benchmark or dataset for high-five or dap meeting was found. The motion datasets are Bütepage et al.'s and NuiSI (Q2), which are skeleton trajectories, not simulation environments.

## Q6. Evaluation practices: timing/synchrony metrics, user ratings, safety

### Takeaway
The field evaluates mainly with:
- Subjective ratings: pleasantness, energeticness, dominance, safety, naturalness, preference.
- Recognition and classification accuracy.
- Trajectory-prediction MSE.

Timing and synchrony appear as manipulated variables (tempo, reactivity, release timing), more often than as measured contact-time error. Safety is mostly a self-reported rating, not a measured force or velocity bound.

### Cited Findings
- Subjective ratings:
  - Fitter & Kuchenbecker (IJSR): pleasantness, energeticness, dominance and apparent safety under 24 conditions. — [IEEE Spectrum](https://spectrum.ieee.org/how-high-fives-help-us-get-in-touch-with-robots); [MPI-IS](https://is.mpg.de/publications/fitter-ijsr-clap)
  - Frontiers 2018: pre/post survey items (feeling understood p = 0.023; willingness to follow p = 0.031) and a safety rating of M = 79.71. — [Frontiers](https://www.frontiersin.org/articles/10.3389/frobt.2018.00085)
- Synchrony as a variable: Fitter et al. ICSR 2016 modelled human-human entrainment and compared three adaptive-tempo controllers. — [MPI-IS](https://hi.is.mpg.de/publications/fitter16-icsr-timing)
- Human-likeness: Avraham et al.'s Turing-like test measures how much noise must be mixed into a real handshake. — [HUJI CRIS](https://cris.huji.ac.il/en/publications/toward-perceiving-robots-as-humans-three-handshake-models-face-th/)
- Sensory evaluation for handshake approach timing: Jindai et al. 2008; Ota et al. 2019. — [Fuji Press](https://www.fujipress.jp/jrm/rb/robot002000040650/); [J-STAGE](https://www.jstage.jst.go.jp/article/jamdsm/13/2/13_2019jamdsm0026/_article/-char/en)
- Trajectory metric: MSE over a 1 s window of the predicted robot or human motion (MILD). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2210.12418)
- RL success rate plus a video preference study (N = 116): Christen et al. 2019. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1906.11695)
- Release timing as a safety and comfort factor: users dislike release that is too early or too late (HuggieBot line). — [IJSR 2019 summary](https://is.mpg.de/publications/block18-ijsr-improve)
- Reaction to touch: the NAO/Pepper study found users prefer the touched hand to "move back." — [arXiv 2209.00414](https://arxiv.org/pdf/2209.00414)

### Inferences
- A dap evaluation can borrow the field's practice of reporting both contact success and timing, while adding objective metrics the literature largely lacks:
  - Contact-time error relative to the partner's min-jerk arrival.
  - Palm-normal antiparallel angle error.
  - Positional offset at contact.
  - Peak relative velocity at contact, as a safety proxy since force is not modelled kinematically.
- Paired comparisons across sampled partner speeds echo the factorial within-subject designs in the user studies (tempo × reactivity), but with ground truth in place of ratings.

### Gaps
- No paper found reports a standardized "contact timing error" metric for high-fives or claps. The Frontiers 2018 accuracy drop is attributed to timing variation, but no timing-error statistic was retrieved.
- No safety metric based on measured contact force or velocity was found for social touch gestures in this pass. ISO/TS 15066-style limits belong to adjacent pHRI safety literature, out of scope here.
