"""PPO teacher on the privileged observation (docs/teacher_design.md, section 2).

Training goes through `ParallelPPO`, Stable-Baselines3's PPO with its rollout
collection replaced: each worker process runs its own env and its own copy of
the actor for a whole rollout. SB3's lockstep VecEnvs stall every env while
one rebuilds its Drake scene on reset (about 49 ms, against about 1 ms per
step), which capped training at about 600 steps/s (docs/teacher_design.md).
The update is SB3's own `PPO.train`, plus, optionally, kickstarting's
imitation term toward the servo expert (`kick_coef`; see `ParallelPPO.train`).

Observations: the privileged vector plus `teacher_features` (the palm-to-palm
vector and the rotation from our palm to the antiparallel target), normalized
by running statistics (as VecNormalize, clip 10) that are frozen within a
rollout and saved with every checkpoint. Rewards (dap.reward) are scaled by
the running std of the discounted return and clipped to 10 (as VecNormalize).
Time-limit truncations are bootstrapped with the critic, as SB3 does.

    teacher = load_teacher("results/teacher/<run>/ckpt_2000000")
    action = teacher(obs)            # a policy for dap.policy_eval (obs_mode "privileged")
    action = teacher.label(info["privileged"])   # DAgger labels from a pixel rollout
"""

import multiprocessing as mp
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from dap.envs.dap_env import PRIVILEGED_DIM, PRIVILEGED_LAYOUT, DapEnv
from dap.reward import DapReward, RewardConfig

ROT_Y_PI = np.diag([-1.0, 1.0, -1.0])
FEATURE_DIM = PRIVILEGED_DIM + 3 + 9
CLIP_OBS = 10.0
CLIP_REWARD = 10.0
EPS = 1e-8


def _slices(layout):
    out, i = {}, 0
    for name, n in layout:
        out[name] = slice(i, i + n)
        i += n
    return out


_S = _slices(PRIVILEGED_LAYOUT)


def _frame(x, z):
    """Rotation with columns (x, z cross x, z): a palm frame from its finger
    direction and normal."""
    return np.stack([x, np.cross(z, x), z], axis=-1)


def teacher_features(state):
    """(..., 43) privileged vectors -> (..., 55): the vector, the partner palm
    relative to ours (3), and R_err = R_partner RotY(pi) R_ours^T (9 entries;
    identity when antiparallel). Matrix entries are continuous where an
    axis-angle vector flips at 180 degrees. Ours."""
    s = np.asarray(state, np.float64)
    rel = s[..., _S["partner_p"]] - s[..., _S["palm_p"]]
    R_o = _frame(s[..., _S["palm_x"]], s[..., _S["palm_n"]])
    R_p = _frame(s[..., _S["partner_x"]], s[..., _S["partner_n"]])
    R_err = R_p @ ROT_Y_PI @ np.swapaxes(R_o, -1, -2)
    flat = R_err.reshape(*R_err.shape[:-2], 9)
    return np.concatenate([s, rel, flat], axis=-1).astype(np.float32)


class TeacherObs(gym.ObservationWrapper):
    """{"state": privileged (43)} -> {"state": teacher features (55)}."""

    def __init__(self, env):
        super().__init__(env)
        box = env.observation_space["state"]
        lo = np.concatenate([box.low, np.full(3, -2 * box.high.max()), -np.ones(9)])
        hi = np.concatenate([box.high, np.full(3, 2 * box.high.max()), np.ones(9)])
        self.observation_space = spaces.Dict(
            {"state": spaces.Box(lo.astype(np.float32), hi.astype(np.float32), dtype=np.float32)})

    def observation(self, obs):
        return {"state": teacher_features(obs["state"])}


def make_teacher_env(bank=None, reward=RewardConfig(), **env_kwargs):
    """The training env: privileged mode (latency-matched by default), teacher
    features, and the training reward."""
    return DapReward(TeacherObs(DapEnv(obs_mode="privileged", bank=bank, **env_kwargs)), reward)


def normalize(x, rms):
    return np.clip((x - rms.mean) / np.sqrt(rms.var + EPS), -CLIP_OBS, CLIP_OBS).astype(np.float32)


# -- workers ------------------------------------------------------------------

def _worker(remote, seed, env_kwargs, policy_class, policy_kwargs, obs_space, action_space,
            kick):
    th.set_num_threads(1)
    env = make_teacher_env(**env_kwargs)
    expert = ServoExpert() if kick else None  # labels the policy's own states
    policy = policy_class(obs_space, action_space, lambda _: 0.0, **policy_kwargs)
    policy.set_training_mode(False)
    th.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    ep_ret, ep_len, ep_terms = 0.0, 0, {}
    while True:
        cmd, data = remote.recv()
        if cmd == "close":
            remote.close()
            return
        state_dict, rms, n = data
        policy.load_state_dict(state_dict)

        def value(raw):
            with th.no_grad():
                t = policy.obs_to_tensor({"state": normalize(raw, rms)})[0]
                return float(policy.predict_values(t)[0])

        raw = np.zeros((n, FEATURE_DIM), np.float32)
        norm = np.zeros_like(raw)
        act = np.zeros((n, 7), np.float32)
        val, logp, rew, boot = (np.zeros(n, np.float32) for _ in range(4))
        expert_act = np.zeros((n, 7), np.float32) if expert else None
        start, done = np.zeros(n, bool), np.zeros(n, bool)
        episodes = []
        for t in range(n):
            start[t] = ep_len == 0
            raw[t] = obs["state"]
            norm[t] = normalize(raw[t], rms)
            if expert:
                if start[t]:
                    expert.reset()
                expert_act[t] = expert({"state": raw[t][:PRIVILEGED_DIM]})
            with th.no_grad():
                a, v, lp = policy(policy.obs_to_tensor({"state": norm[t]})[0])
            act[t], val[t], logp[t] = a.numpy()[0], float(v), float(lp)
            obs, r, term, trunc, info = env.step(np.clip(act[t], -1.0, 1.0))
            rew[t] = r
            ep_ret += r
            ep_len += 1
            for k, x in info["reward_terms"].items():
                ep_terms[f"r_{k}"] = ep_terms.get(f"r_{k}", 0.0) + x
            if term or trunc:
                done[t] = True
                if trunc and not term:
                    boot[t] = value(obs["state"])  # in normalized-reward units
                episodes.append(dict(r=ep_ret, l=ep_len, is_success=bool(info["success"]),
                                     outcome=info["event"] or "timeout", **ep_terms))
                ep_ret, ep_len, ep_terms = 0.0, 0, {}
                obs, _ = env.reset()
        last_value = value(obs["state"])
        remote.send(dict(raw=raw, norm=norm, act=act, val=val, logp=logp, rew=rew, boot=boot,
                         start=start, done=done, last_value=last_value, expert=expert_act,
                         last_start=ep_len == 0, episodes=episodes))


class _SpacesOnly(VecEnv):
    """Carries the spaces and the number of workers for SB3; never steps."""

    def __init__(self, n, observation_space, action_space):
        super().__init__(n, observation_space, action_space)

    def reset(self):
        return {"state": np.zeros((self.num_envs, FEATURE_DIM), np.float32)}

    def step_async(self, actions):
        raise RuntimeError("ParallelPPO collects rollouts in its workers")

    def step_wait(self):
        raise RuntimeError("ParallelPPO collects rollouts in its workers")

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs


class ParallelPPO(PPO):
    """SB3 PPO whose rollouts come from `n_workers` processes, each stepping
    its own env with a copy of the current policy (see module docstring).
    `kick_coef` > 0 adds kickstarting's term (see `train`), annealed linearly
    to 0 over `kick_steps`. With `expert_labels` (implied by `kick_coef`), the
    workers run the ServoExpert alongside the policy (for `dagger`)."""

    def __init__(self, n_workers, env_kwargs, seed=0, kick_coef=0.0, kick_steps=1_000_000,
                 expert_labels=False, **ppo_kwargs):
        probe = make_teacher_env(**{k: v for k, v in env_kwargs.items() if k != "bank"})
        obs_space, action_space = probe.observation_space, probe.action_space
        reward = env_kwargs.get("reward", RewardConfig())
        if abs(reward.gamma - ppo_kwargs.get("gamma", 0.99)) > 1e-12:
            raise ValueError("the reward's gamma must equal PPO's (potential-based shaping)")
        super().__init__("MultiInputPolicy", _SpacesOnly(n_workers, obs_space, action_space),
                         seed=seed, device="cpu", **ppo_kwargs)
        self.obs_rms = RunningMeanStd(shape=(FEATURE_DIM,))
        self.ret_rms = RunningMeanStd(shape=())
        self._returns = np.zeros(n_workers)
        self.episode_log = []  # every finished training episode (for the run's log)
        self.kick_coef, self.kick_steps = kick_coef, kick_steps
        self._kick = None  # this rollout's (normalized obs, expert actions)
        ctx = mp.get_context("spawn")
        self._remotes, self._procs = [], []
        for i in range(n_workers):
            here, there = ctx.Pipe()
            p = ctx.Process(target=_worker, daemon=True,
                            args=(there, seed * 1000 + i, env_kwargs, type(self.policy),
                                  self.policy_kwargs, obs_space, action_space,
                                  expert_labels or kick_coef > 0))
            p.start()
            there.close()
            self._remotes.append(here)
            self._procs.append(p)

    def gather(self, n_steps):
        """`n_steps` steps from each worker with the current (stochastic)
        policy and frozen observation statistics: one dict per worker."""
        self.policy.set_training_mode(False)
        state = {k: v.detach().cpu() for k, v in self.policy.state_dict().items()}
        rms = RunningMeanStd(shape=(FEATURE_DIM,))
        rms.mean, rms.var = self.obs_rms.mean.copy(), self.obs_rms.var.copy()
        for r in self._remotes:
            r.send(("collect", (state, rms, n_steps)))
        return [r.recv() for r in self._remotes]

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        callback.on_rollout_start()
        outs = self.gather(n_rollout_steps)
        stack = {k: np.stack([o[k] for o in outs], axis=1) for k in
                 ("raw", "norm", "act", "val", "logp", "rew", "boot", "start", "done")}

        # Reward scaling (VecNormalize's rule), then the truncation bootstrap.
        rew = np.zeros_like(stack["rew"])
        for t in range(n_rollout_steps):
            self._returns = self._returns * self.gamma + stack["rew"][t]
            self.ret_rms.update(self._returns.copy())
            rew[t] = np.clip(stack["rew"][t] / np.sqrt(self.ret_rms.var + EPS),
                             -CLIP_REWARD, CLIP_REWARD)
            self._returns[stack["done"][t]] = 0.0
        rew = rew + self.gamma * stack["boot"]
        self.obs_rms.update(stack["raw"].reshape(-1, FEATURE_DIM))
        if self.kick_coef:
            self._kick = (th.as_tensor(stack["norm"].reshape(-1, FEATURE_DIM)),
                          th.as_tensor(np.concatenate([o["expert"] for o in outs])))

        rollout_buffer.reset()
        for t in range(n_rollout_steps):
            rollout_buffer.add({"state": stack["norm"][t]}, stack["act"][t], rew[t],
                               stack["start"][t], th.as_tensor(stack["val"][t]),
                               th.as_tensor(stack["logp"][t]))
        last_values = th.as_tensor(np.array([o["last_value"] for o in outs], np.float32))
        rollout_buffer.compute_returns_and_advantage(
            last_values=last_values, dones=np.array([o["last_start"] for o in outs]))

        episodes = [e for o in outs for e in o["episodes"]]
        self.episode_log += [dict(e, timesteps=self.num_timesteps) for e in episodes]
        self.ep_info_buffer.extend({"r": e["r"], "l": e["l"]} for e in episodes)
        self.ep_success_buffer.extend(e["is_success"] for e in episodes)
        if episodes:
            for o in ("contact", "collision", "timeout"):
                self.logger.record(f"rollout/frac_{o}", np.mean([e["outcome"] == o for e in episodes]))
            for k in ("outcome", "shaping", "action_rate"):
                self.logger.record(f"rollout/ep_r_{k}", np.mean([e.get(f"r_{k}", 0.0) for e in episodes]))
        self.num_timesteps += env.num_envs * n_rollout_steps
        callback.update_locals(locals())
        if not callback.on_step():
            return False
        callback.on_rollout_end()
        return True

    def train(self):
        """SB3 2.9.2's PPO.train, trimmed to what we use (Box actions, no value
        clipping), plus kickstarting's term (Schmitt et al. 2018): the
        cross-entropy from the expert to the policy on the policy's own states,
        which for a deterministic expert is -lambda_k mean log pi(a_expert | s).
        Each gradient step takes it on a random minibatch of this rollout's
        states, labelled by the ServoExpert running alongside in each worker.
        lambda_k = kick_coef (1 - steps / kick_steps), floored at 0."""
        lam = self.kick_coef * max(0.0, 1.0 - self.num_timesteps / self.kick_steps)
        if not lam:
            return super().train()
        policy = self.policy
        policy.set_training_mode(True)
        self._update_learning_rate(policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        kick_obs, kick_act = self._kick
        logs = {k: [] for k in ("pg", "vf", "kick_nll", "kick_mse", "clip", "kl")}
        continue_training = True
        for epoch in range(self.n_epochs):
            for data in self.rollout_buffer.get(self.batch_size):
                values, log_prob, _ = policy.evaluate_actions(data.observations, data.actions)
                adv = data.advantages
                if self.normalize_advantage and len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = th.exp(log_prob - data.old_log_prob)
                pg_loss = -th.min(adv * ratio,
                                  adv * th.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
                vf_loss = th.mean((data.returns - values.flatten()) ** 2)
                idx = th.randint(len(kick_obs), (self.batch_size,))
                dist = policy.get_distribution({"state": kick_obs[idx]})
                kick_nll = -dist.log_prob(kick_act[idx]).mean()
                loss = pg_loss + self.vf_coef * vf_loss + lam * kick_nll
                with th.no_grad():
                    log_ratio = log_prob - data.old_log_prob
                    kl = float(th.mean(th.exp(log_ratio) - 1 - log_ratio))
                    logs["kick_mse"].append(float(th.mean((dist.distribution.mean
                                                           - kick_act[idx]) ** 2)))
                logs["pg"].append(pg_loss.item())
                logs["vf"].append(vf_loss.item())
                logs["kick_nll"].append(kick_nll.item())
                logs["clip"].append(float(th.mean((th.abs(ratio - 1) > clip_range).float())))
                logs["kl"].append(kl)
                if self.target_kl is not None and kl > 1.5 * self.target_kl:
                    continue_training = False
                    break
                policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break
        rec = self.logger.record
        rec("train/policy_gradient_loss", np.mean(logs["pg"]))
        rec("train/value_loss", np.mean(logs["vf"]))
        rec("train/approx_kl", np.mean(logs["kl"]))
        rec("train/clip_fraction", np.mean(logs["clip"]))
        rec("train/kick_coef", lam)
        rec("train/kick_nll", np.mean(logs["kick_nll"]))
        rec("train/kick_mse", np.mean(logs["kick_mse"]))
        rec("train/explained_variance", explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()))
        rec("train/std", th.exp(policy.log_std).mean().item())
        rec("train/n_updates", self._n_updates, exclude="tensorboard")
        rec("train/clip_range", clip_range)

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "_remotes", "_procs", "obs_rms", "ret_rms",
                "_returns", "episode_log", "_kick"]

    def save_checkpoint(self, path):
        """`path`.zip (the SB3 model) and `path`.norm.pkl (observation stats)."""
        path = Path(path)
        self.save(path.with_suffix(".zip"))
        with open(path.with_suffix(".norm.pkl"), "wb") as f:
            pickle.dump({"mean": self.obs_rms.mean, "var": self.obs_rms.var,
                         "count": self.obs_rms.count}, f)

    def close(self):
        for r in self._remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=5)


# -- the servo as an expert on the teacher's observation ----------------------

class ServoExpert:
    """The diff-IK servo (dap.baselines.DiffIKServo) acting at the policy rate
    on the privileged observation: the latency-matched partner palm pose and
    velocity, extrapolated LOOKAHEAD ahead as servo_gt does. It sees exactly
    what the teacher sees, so it can warm-start the teacher (DAPG, Rajeswaran
    et al. 2018; kickstarting, Schmitt et al. 2018) and label its states
    (DAgger, Ross et al. 2011) without an information gap. Its closing rule
    is memoryless (`latch=False`): the latched rule depends on history the
    teacher's MLP does not have, and its labels on the policy's own states
    conflicted (docs/teacher_design.md, section 2b). A reference, not a
    method."""

    obs_mode = "privileged"

    def __init__(self, dt=0.05):
        from dap.planner import controller_model
        from dap.scene import Scene

        # Our arm's Jacobian does not depend on where the partner stands.
        self.model = controller_model(Scene(static=False), dt)
        self.v_max = self.model.v_max
        self.reset()

    def reset(self):
        from dap.baselines import DiffIKServo

        self.servo = DiffIKServo(self._partner, latch=False)
        self.servo.reset(self.model, None)

    def _partner(self, t, _):
        return self._target

    def __call__(self, obs):
        from dap.baselines import LOOKAHEAD

        s = np.asarray(obs["state"], np.float64)
        v = s[_S["partner_v"]]
        self._target = (s[_S["partner_p"]] + LOOKAHEAD * v, v,
                        _frame(s[_S["partner_x"]], s[_S["partner_n"]]))
        q = s[_S["q"]]
        qd = (self.servo.step(float(s[_S["t"]][0]), q, None) - q) / self.model.dt
        return np.clip(qd / self.v_max, -1.0, 1.0)


# -- warm start from the expert ---------------------------------------------

def _expert_episodes(args):
    seed, n, env_kwargs = args
    th.set_num_threads(1)
    env, expert = make_teacher_env(**env_kwargs), ServoExpert()
    obs, _ = env.reset(seed=seed)
    out = []
    for _ in range(n):
        expert.reset()
        feats, acts, rews, done = [], [], [], False
        while not done:
            a = expert({"state": obs["state"][:PRIVILEGED_DIM]})  # the privileged vector
            feats.append(obs["state"])
            acts.append(a)
            obs, r, term, trunc, info = env.step(a)
            rews.append(r)
            done = term or trunc
        out.append(dict(obs=np.array(feats, np.float32), act=np.array(acts, np.float32),
                        rew=np.array(rews, np.float64), success=bool(info["success"]),
                        outcome=info["event"] or "timeout"))
        obs, _ = env.reset()
    return out


def collect_expert(n_episodes, env_kwargs, workers=8, seed=0):
    """`n_episodes` ServoExpert episodes on training scenarios (the env's bank
    or fresh training samples), with the training reward."""
    per = [n_episodes // workers + (i < n_episodes % workers) for i in range(workers)]
    jobs = [(seed * 1000 + 500 + i, k, env_kwargs) for i, k in enumerate(per) if k]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        return [e for part in pool.map(_expert_episodes, jobs) for e in part]


def warm_start(model, episodes, epochs=30, batch_size=512, lr=1e-3, log_std=float(np.log(0.2)),
               seed=0):
    """Fit `model` (a ParallelPPO) to expert episodes before PPO.
      - observation and return statistics from the expert's data;
      - actor: the Gaussian's mean regressed on the expert's actions (MSE);
      - critic: regressed on the expert's discounted returns in the scaled
        reward units PPO will use (episodes that time out are not
        bootstrapped, a small bias: the expert rarely times out);
      - exploration std reset to exp(`log_std`) (Ours).
    Returns the final actor and critic losses."""
    gamma = model.gamma
    model.obs_rms.update(np.concatenate([e["obs"] for e in episodes]))
    running = []
    for e in episodes:  # the running discounted return, as VecNormalize tracks it
        ret = 0.0
        for r in e["rew"]:
            ret = ret * gamma + r
            running.append(ret)
    model.ret_rms.update(np.array(running))
    scale = np.sqrt(model.ret_rms.var + EPS)
    obs, act, target = [], [], []
    for e in episodes:
        r = np.clip(e["rew"] / scale, -CLIP_REWARD, CLIP_REWARD)
        g, G = 0.0, np.zeros(len(r))
        for t in range(len(r) - 1, -1, -1):
            g = r[t] + gamma * g
            G[t] = g
        obs.append(normalize(e["obs"], model.obs_rms))
        act.append(e["act"])
        target.append(G)
    obs = th.as_tensor(np.concatenate(obs))
    act = th.as_tensor(np.concatenate(act))
    target = th.as_tensor(np.concatenate(target), dtype=th.float32)

    pi_loss, vf_loss = _regress(model.policy, obs, act, target, epochs, batch_size, lr, seed)
    with th.no_grad():
        model.policy.log_std.fill_(log_std)
    return dict(pi_mse=pi_loss, vf_mse=vf_loss, target_var=float(target.var()), n=len(obs))


def _regress(policy, obs, act, target, epochs, batch_size, lr, seed):
    """Adam on the actor mean's MSE to `act`, plus the critic's MSE to
    `target` unless it is None. Returns both final losses on all the data."""
    policy.set_training_mode(True)
    opt = th.optim.Adam(policy.parameters(), lr=lr)
    gen = th.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in th.randperm(len(obs), generator=gen).split(batch_size):
            o = {"state": obs[idx]}
            loss = th.mean((policy.get_distribution(o).distribution.mean - act[idx]) ** 2)
            if target is not None:
                loss = loss + th.mean((policy.predict_values(o).flatten() - target[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    policy.set_training_mode(False)
    with th.no_grad():
        o = {"state": obs}
        pi = float(th.mean((policy.get_distribution(o).distribution.mean - act) ** 2))
        vf = (float(th.mean((policy.predict_values(o).flatten() - target) ** 2))
              if target is not None else float("nan"))
    return pi, vf


def dagger(model, episodes, steps, epochs=10, batch_size=512, lr=1e-3, seed=0):
    """One DAgger round (Ross et al. 2011) toward the ServoExpert. Rolls out
    the current policy (its Gaussian, at the std it has) for `steps` steps in
    the workers, which must run with `expert_labels`; the expert labels every
    state visited; those states join `episodes` (the aggregated dataset, which
    starts as the expert's own episodes and is extended in place); the actor's
    mean is refitted on all of it (MSE, continuing from its weights).
    Observation statistics stay as warm_start fitted them, so labels from every
    round are normalized alike. Returns the round's statistics."""
    outs = model.gather(-(-steps // len(model._remotes)))
    for o in outs:
        episodes.append(dict(obs=o["raw"], act=o["expert"]))
    obs = th.as_tensor(np.concatenate([normalize(e["obs"], model.obs_rms) for e in episodes]))
    act = th.as_tensor(np.concatenate([e["act"] for e in episodes]))
    rolled = [e for o in outs for e in o["episodes"]]
    new = np.concatenate([o["raw"] for o in outs])
    with th.no_grad():  # how far the policy was from the expert on its own states
        mean = model.policy.get_distribution(
            {"state": th.as_tensor(normalize(new, model.obs_rms))}).distribution.mean
        gap = float(th.mean((mean - th.as_tensor(np.concatenate([o["expert"] for o in outs])))
                            ** 2))
    pi_mse, _ = _regress(model.policy, obs, act, None, epochs, batch_size, lr, seed)
    return dict(n=len(obs), gap_mse=gap, pi_mse=pi_mse,
                rollout_success=float(np.mean([e["is_success"] for e in rolled])) if rolled
                else float("nan"),
                rollout_collision=float(np.mean([e["outcome"] == "collision" for e in rolled]))
                if rolled else float("nan"))


# -- a trained teacher ---------------------------------------------------------

class Teacher:
    """A frozen teacher acting deterministically (its Gaussian's mean, clipped
    to [-1, 1]). A reference, not a method: it sees ground truth."""

    obs_mode = "privileged"

    def __init__(self, policy, mean, var):
        self.policy, self.mean, self.var = policy, mean, var
        self.policy.set_training_mode(False)

    def label(self, privileged):
        """Actions for (..., 43) privileged vectors (e.g. info["privileged"])."""
        x = teacher_features(privileged)
        x = np.clip((x - self.mean) / np.sqrt(self.var + EPS), -CLIP_OBS, CLIP_OBS)
        a, _ = self.policy.predict({"state": x.astype(np.float32)}, deterministic=True)
        return np.clip(a, -1.0, 1.0)

    def __call__(self, obs):
        return self.label(obs["state"])


def load_teacher(path):
    """A Teacher from `path`.zip and `path`.norm.pkl (ParallelPPO.save_checkpoint)."""
    path = Path(path)
    model = PPO.load(path.with_suffix(".zip"), device="cpu")
    with open(path.with_suffix(".norm.pkl"), "rb") as f:
        stats = pickle.load(f)
    return Teacher(model.policy, stats["mean"], stats["var"])
