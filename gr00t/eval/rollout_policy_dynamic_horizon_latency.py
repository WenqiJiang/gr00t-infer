"""Rollout policy evaluation with dynamic horizon and latency switching.

Extends rollout_policy_with_latency.py by supporting two (n_action_steps, latency)
configurations that switch at a trigger point during each episode. The trigger is
when the gripper first closes (grabs something). The current action chunk is always
finished before the switch takes effect.

Phase 1 (pre-grasp):  (n_action_steps_1, latency_1, staleness_1)
Phase 2 (post-grasp): (n_action_steps_2, latency_2, staleness_2)

Implementation: MultiStepWrapper.n_action_steps is set to the maximum of the two
phases' total steps (padding + n_action_steps). For the shorter phase, tail hold
actions are appended so the env always steps a fixed number of raw steps per macro
step. This wastes a few sim steps but keeps the vectorized env interface intact.

Additional per-episode metrics recorded:
    - time_before_switch / time_after_switch: wall-clock seconds in each phase.
    - inferences_before_switch / inferences_after_switch: policy query counts.
    - switch_macro_step: which macro step the switch happened (None if never).

Usage:
    # Terminal 1 - Server (same as standard eval):
    uv run python gr00t/eval/run_gr00t_server.py \\
        --model-path nvidia/GR00T-N1.6-3B \\
        --embodiment-tag ROBOCASA_PANDA_OMRON \\
        --use-sim-policy-wrapper

    # Terminal 2 - Client with dynamic horizon/latency:
    uv run python gr00t/eval/rollout_policy_dynamic_horizon_latency.py \\
        --policy_client_host 127.0.0.1 \\
        --policy_client_port 5555 \\
        --env_name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \\
        --n_episodes 10 --n_envs 5 \\
        --latency_1 10 --n_action_steps_1 8 \\
        --latency_2 4 --n_action_steps_2 4

NOTE on reproducibility: even with fixed seeds (--seed), results may vary across
runs due to GPU non-determinism (CUDA, cuDNN), async vectorized env scheduling,
and floating-point non-associativity in MuJoCo physics. The best way to report
reliable accuracy is to run a large number of episodes (50+) and report mean +/- stderr.
"""

import argparse
import collections
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import time
from typing import Any
import uuid

from gr00t.eval.rollout_policy import (
    MultiStepConfig,
    VideoConfig,
    WrapperConfigs,
    create_eval_env,
    create_gr00t_sim_policy,
)
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.policy import BasePolicy
import gymnasium as gym
import numpy as np
from tqdm import tqdm


# Action keys that should retain their last value during hold (e.g., gripper state,
# control mode). All other keys are assumed to be delta/velocity commands and are
# zeroed out during hold so the robot stays stationary.
HOLD_KEEP_LAST_KEYS = {"action.gripper_close", "action.control_mode"}

# Key used to detect gripper close events for phase switching.
GRIPPER_CLOSE_KEY = "action.gripper_close"


@dataclass
class DynamicHorizonLatencyConfig:
    """Configuration for two-phase dynamic horizon and latency.

    Phase 1 (pre-grasp): active until the gripper first closes.
    Phase 2 (post-grasp): active after the gripper close chunk finishes.

    Attributes:
        latency_1/2: Raw env steps of simulated inference delay per phase.
        staleness_1/2: Observation staleness per phase.
        n_action_steps_1/2: Model actions to execute per replan per phase.
        extend_episode_budget: Extend max steps to compensate for hold actions.
    """

    # Phase 1: pre-grasp
    latency_1: int = 0
    staleness_1: int = 0
    n_action_steps_1: int = 8
    # Phase 2: post-grasp
    latency_2: int = 0
    staleness_2: int = 0
    n_action_steps_2: int = 8
    # Common
    extend_episode_budget: bool = True


def run_rollout_with_dynamic_horizon_latency(
    env_name: str,
    policy: BasePolicy,
    wrapper_configs: WrapperConfigs,
    dynamic_config: DynamicHorizonLatencyConfig,
    n_episodes: int = 10,
    n_envs: int = 1,
    seed: int | None = None,
) -> Any:
    """Run policy rollouts with two-phase dynamic horizon and latency.

    The rollout starts in phase 1. When the model predicts a gripper close action
    above the threshold, the current action chunk finishes executing, then phase 2
    settings take effect from the next macro step onward.

    MultiStepWrapper.n_action_steps is set to max(total_1, total_2) so the env
    always steps a fixed number of raw steps. Shorter phases get tail hold padding.

    Args:
        env_name: Name of the gymnasium environment to use.
        policy: Policy instance (local or PolicyClient).
        wrapper_configs: Configuration for environment wrappers.
        dynamic_config: Two-phase latency/horizon configuration.
        n_episodes: Number of episodes to run.
        n_envs: Number of parallel environments.
        seed: Random seed.

    Returns:
        Tuple of (env_name, episode_successes, episode_infos).
    """
    # Phase 1 parameters
    latency_1 = dynamic_config.latency_1
    staleness_1 = dynamic_config.staleness_1
    nact_1 = dynamic_config.n_action_steps_1
    padding_1 = latency_1 - staleness_1
    assert 0 <= staleness_1 <= latency_1, (
        f"staleness_1 ({staleness_1}) must be in [0, latency_1={latency_1}]"
    )
    total_1 = padding_1 + nact_1

    # Phase 2 parameters
    latency_2 = dynamic_config.latency_2
    staleness_2 = dynamic_config.staleness_2
    nact_2 = dynamic_config.n_action_steps_2
    padding_2 = latency_2 - staleness_2
    assert 0 <= staleness_2 <= latency_2, (
        f"staleness_2 ({staleness_2}) must be in [0, latency_2={latency_2}]"
    )
    total_2 = padding_2 + nact_2

    max_total = max(total_1, total_2)
    max_staleness = max(staleness_1, staleness_2)

    # Episode budget extension: use worst-case factor.
    original_max_steps = wrapper_configs.multistep.max_episode_steps
    max_padding = max(padding_1, padding_2)
    if dynamic_config.extend_episode_budget and max_padding > 0:
        min_nact = min(nact_1, nact_2)
        factor = max_total / min_nact
        extended_max_steps = int(original_max_steps * factor)
        wrapper_configs.multistep.max_episode_steps = extended_max_steps
        wrapper_configs.video.max_episode_steps = extended_max_steps
        print(
            f"Extended episode budget: {original_max_steps} -> {extended_max_steps} "
            f"raw steps (factor={factor:.2f})"
        )
    else:
        extended_max_steps = original_max_steps

    # Set MultiStepWrapper to max of both phases.
    wrapper_configs.multistep.n_action_steps = max_total

    # Enable intermediate observation caching if either phase uses staleness.
    if max_staleness > 0:
        wrapper_configs.multistep.cache_intermediate_obs = True

    start_time = time.time()
    n_envs = min(n_envs, n_episodes)
    print(
        f"Running collecting {n_episodes} episodes for {env_name} with {n_envs} vec envs"
        f" (phase1: lat={latency_1}, stale={staleness_1}, nact={nact_1}"
        f" | phase2: lat={latency_2}, stale={staleness_2}, nact={nact_2})"
    )

    env_fns = [
        partial(
            create_eval_env,
            env_idx=idx,
            env_name=env_name,
            total_n_envs=n_envs,
            wrapper_configs=wrapper_configs,
            seed=seed,
        )
        for idx in range(n_envs)
    ]

    if n_envs == 1:
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=False,
            context="spawn",
        )

    # Storage for results
    episode_lengths = []
    current_rewards = [0] * n_envs
    current_lengths = [0] * n_envs
    completed_episodes = 0
    current_successes = [False] * n_envs
    episode_successes = []
    episode_infos = defaultdict(list)

    # Track success at original budget (before extension) for fair comparison.
    current_successes_original = [False] * n_envs
    episode_successes_original = []

    # -- Per-env phase state --
    env_phase = [1] * n_envs  # 1 = pre-grasp, 2 = post-grasp
    switch_pending = [False] * n_envs  # gripper close detected, finish chunk first

    # Per-env timing and inference tracking
    env_episode_start_time = [time.time()] * n_envs
    env_switch_time = [None] * n_envs  # wall-clock time of phase switch
    env_inferences_phase1 = [0] * n_envs
    env_inferences_phase2 = [0] * n_envs
    env_switch_macro_step = [None] * n_envs  # macro step index of switch

    # -- Latency state --
    # Last model-predicted actions for hold-action generation: {key: (B, nact, D)}
    last_model_actions = None

    # -- Staleness state: per-env raw observation cache --
    if max_staleness > 0:
        raw_obs_caches = [
            collections.deque(maxlen=max_staleness + 1) for _ in range(n_envs)
        ]

    # Initial reset
    observations, _ = env.reset(seed=seed)
    policy.reset()

    # Seed staleness caches with initial observation.
    if max_staleness > 0:
        for env_idx in range(n_envs):
            per_env_obs = {}
            for k, v in observations.items():
                val = v[env_idx]
                if (k.startswith("video") or k.startswith("state")) and isinstance(
                    val, np.ndarray
                ):
                    val = val[-1]
                per_env_obs[k] = val
            raw_obs_caches[env_idx].append(per_env_obs)

    i = 0

    pbar = tqdm(total=n_episodes, desc="Episodes")
    while completed_episodes < n_episodes:
        # -- Apply pending phase switches (from previous macro step) --
        for env_idx in range(n_envs):
            if switch_pending[env_idx]:
                env_phase[env_idx] = 2
                switch_pending[env_idx] = False
                env_switch_time[env_idx] = time.time()
                env_switch_macro_step[env_idx] = current_lengths[env_idx]

        # -- Select observation for policy (fresh or stale, per-env) --
        if max_staleness > 0 and last_model_actions is not None:
            stale_obs_parts = {k: [] for k in observations}
            for env_idx in range(n_envs):
                phase = env_phase[env_idx]
                stale = staleness_1 if phase == 1 else staleness_2
                cache = raw_obs_caches[env_idx]
                idx = max(0, len(cache) - 1 - stale) if stale > 0 else len(cache) - 1
                for k in observations:
                    stale_obs_parts[k].append(cache[idx][k])
            policy_obs = {}
            for k, parts in stale_obs_parts.items():
                if isinstance(parts[0], str):
                    policy_obs[k] = tuple(parts)
                else:
                    if k.startswith("video") or k.startswith("state"):
                        parts = [np.expand_dims(p, axis=0) for p in parts]
                    policy_obs[k] = np.stack(parts, axis=0)
        else:
            policy_obs = observations

        # -- Query policy --
        actions, _ = policy.get_action(policy_obs)

        # -- Track inference counts per phase per env --
        for env_idx in range(n_envs):
            if env_phase[env_idx] == 1:
                env_inferences_phase1[env_idx] += 1
            else:
                env_inferences_phase2[env_idx] += 1

        # -- Detect gripper close for phase switch --
        # Switch as soon as the model commands any positive gripper close action.
        if GRIPPER_CLOSE_KEY in actions:
            gripper_vals = actions[GRIPPER_CLOSE_KEY]  # (B, T_model, D)
            for env_idx in range(n_envs):
                if env_phase[env_idx] == 1 and not switch_pending[env_idx]:
                    if np.any(gripper_vals[env_idx] > 0.5):
                        switch_pending[env_idx] = True

        # -- Construct per-env padded actions based on phase --
        padded_actions = {}
        for key, value in actions.items():
            # value shape: (B, T_model, D)
            B = value.shape[0]
            D = value.shape[-1]
            per_env_list = []

            for env_idx in range(B):
                phase = env_phase[env_idx]
                pad = padding_1 if phase == 1 else padding_2
                nact = nact_1 if phase == 1 else nact_2
                total = pad + nact
                tail_len = max_total - total

                # Head hold padding (latency simulation)
                if pad > 0:
                    if key in HOLD_KEEP_LAST_KEYS and last_model_actions is not None:
                        hold = np.repeat(
                            last_model_actions[key][env_idx : env_idx + 1, -1:, :],
                            pad,
                            axis=1,
                        )
                    else:
                        hold = np.zeros((1, pad, D), dtype=value.dtype)
                else:
                    hold = np.zeros((1, 0, D), dtype=value.dtype)

                # Model actions (take first nact from model output)
                model_acts = value[env_idx : env_idx + 1, :nact, :]

                # Tail hold padding (to fill up to max_total)
                if tail_len > 0:
                    if key in HOLD_KEEP_LAST_KEYS:
                        tail = np.repeat(model_acts[:, -1:, :], tail_len, axis=1)
                    else:
                        tail = np.zeros((1, tail_len, D), dtype=value.dtype)
                else:
                    tail = np.zeros((1, 0, D), dtype=value.dtype)

                env_action = np.concatenate([hold, model_acts, tail], axis=1)
                per_env_list.append(env_action)

            padded_actions[key] = np.concatenate(per_env_list, axis=0)

        # Save model actions for next hold generation (per-env, using max nact)
        max_nact = max(nact_1, nact_2)
        last_model_actions = {k: v[:, :max_nact, :] for k, v in actions.items()}
        actions = padded_actions

        # -- Step environment --
        next_obs, rewards, terminations, truncations, env_infos = env.step(actions)
        i += 1

        # Update episode tracking
        for env_idx in range(n_envs):
            if "success" in env_infos:
                env_success = env_infos["success"][env_idx]
                if isinstance(env_success, list):
                    env_success = np.any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            else:
                current_successes[env_idx] = False

            if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                env_success = env_infos["final_info"][env_idx]["success"]
                if isinstance(env_success, list):
                    env_success = any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)

            # Track success within original (non-extended) budget.
            raw_steps_so_far = current_lengths[env_idx] * max_total
            if raw_steps_so_far <= original_max_steps:
                current_successes_original[env_idx] = current_successes[env_idx]

            current_rewards[env_idx] += rewards[env_idx]
            current_lengths[env_idx] += 1

            # If episode ended, store results
            if terminations[env_idx] or truncations[env_idx]:
                if "final_info" in env_infos:
                    current_successes[env_idx] |= any(
                        env_infos["final_info"][env_idx]["success"]
                    )
                if "task_progress" in env_infos:
                    episode_infos["task_progress"].append(
                        env_infos["task_progress"][env_idx][-1]
                    )
                if "q_score" in env_infos:
                    episode_infos["q_score"].append(np.max(env_infos["q_score"][env_idx]))
                if "valid" in env_infos:
                    episode_infos["valid"].append(all(env_infos["valid"][env_idx]))

                # -- Record phase-switch metrics --
                ep_end_time = time.time()
                if env_switch_time[env_idx] is not None:
                    t_before = env_switch_time[env_idx] - env_episode_start_time[env_idx]
                    t_after = ep_end_time - env_switch_time[env_idx]
                else:
                    t_before = ep_end_time - env_episode_start_time[env_idx]
                    t_after = 0.0
                episode_infos["time_before_switch"].append(t_before)
                episode_infos["time_after_switch"].append(t_after)
                episode_infos["inferences_before_switch"].append(
                    env_inferences_phase1[env_idx]
                )
                episode_infos["inferences_after_switch"].append(
                    env_inferences_phase2[env_idx]
                )
                episode_infos["switch_macro_step"].append(env_switch_macro_step[env_idx])
                episode_infos["switched"].append(env_switch_time[env_idx] is not None)

                # Accumulate results
                episode_lengths.append(current_lengths[env_idx])
                episode_successes.append(current_successes[env_idx])
                episode_successes_original.append(current_successes_original[env_idx])
                raw_steps = current_lengths[env_idx] * max_total
                switched_str = ""
                if env_switch_macro_step[env_idx] is not None:
                    switched_str = (
                        f", switched@step={env_switch_macro_step[env_idx]}"
                        f" (inf1={env_inferences_phase1[env_idx]}"
                        f", inf2={env_inferences_phase2[env_idx]})"
                    )
                print(
                    f"[DEBUG] Episode {len(episode_successes)} done: "
                    f"env={env_idx}, "
                    f"steps={current_lengths[env_idx]} "
                    f"(~{raw_steps} raw), "
                    f"success={current_successes[env_idx]}"
                    f"{'' if current_successes[env_idx] == current_successes_original[env_idx] else f' (original_budget={current_successes_original[env_idx]})'}"
                    f", terminated={terminations[env_idx]}, "
                    f"truncated={truncations[env_idx]}"
                    f"{switched_str}"
                )

                # Reset trackers for this environment.
                current_successes[env_idx] = False
                current_successes_original[env_idx] = False
                if "valid" in episode_infos:
                    if episode_infos["valid"][-1]:
                        completed_episodes += 1
                        pbar.update(1)
                else:
                    completed_episodes += 1
                    pbar.update(1)
                current_rewards[env_idx] = 0
                current_lengths[env_idx] = 0

                # Reset phase state for new episode
                env_phase[env_idx] = 1
                switch_pending[env_idx] = False
                env_episode_start_time[env_idx] = time.time()
                env_switch_time[env_idx] = None
                env_inferences_phase1[env_idx] = 0
                env_inferences_phase2[env_idx] = 0
                env_switch_macro_step[env_idx] = None

                # Reset hold actions for this env
                if last_model_actions is not None:
                    for k in last_model_actions:
                        last_model_actions[k][env_idx] = 0.0

                # Reset staleness cache for this env
                if max_staleness > 0:
                    raw_obs_caches[env_idx].clear()
                    per_env_obs = {}
                    for k, v in next_obs.items():
                        val = v[env_idx]
                        if (k.startswith("video") or k.startswith("state")) and isinstance(
                            val, np.ndarray
                        ):
                            val = val[-1]
                        per_env_obs[k] = val
                    raw_obs_caches[env_idx].append(per_env_obs)

        observations = next_obs

        # Cache raw-step observations for staleness.
        if max_staleness > 0 and "_intermediate_obs" in env_infos:
            for env_idx in range(n_envs):
                intermediate = env_infos["_intermediate_obs"][env_idx]
                if intermediate is None:
                    continue
                for raw_obs in intermediate:
                    raw_obs_caches[env_idx].append(raw_obs)

    pbar.close()

    env.reset()
    env.close()
    print(f"Collecting {n_episodes} episodes took {time.time() - start_time} seconds")

    # Report average episode length and success rates
    raw_lengths = [l * max_total for l in episode_lengths]
    print(f"Avg steps (all episodes): {np.mean(raw_lengths):.1f} raw")
    success_raw = [l for l, s in zip(raw_lengths, episode_successes) if s]
    if success_raw:
        print(f"Avg steps (succeeded):    {np.mean(success_raw):.1f} raw")
    else:
        print("Avg steps (succeeded):    N/A (no successes)")

    if extended_max_steps != original_max_steps:
        print(
            f"Success rate (extended budget, {extended_max_steps} steps): "
            f"{np.mean(episode_successes):.3f}"
        )
        print(
            f"Success rate (original budget, {original_max_steps} steps): "
            f"{np.mean(episode_successes_original):.3f}"
        )

    # Phase switch statistics
    switched_eps = episode_infos.get("switched", [])
    n_switched = sum(switched_eps) if switched_eps else 0
    print(
        f"Phase switch: {n_switched}/{len(episode_successes)} episodes triggered gripper close"
    )
    if n_switched > 0:
        switch_steps = [
            s for s in episode_infos.get("switch_macro_step", []) if s is not None
        ]
        inf_before = [
            ib
            for ib, sw in zip(episode_infos["inferences_before_switch"], switched_eps)
            if sw
        ]
        inf_after = [
            ia
            for ia, sw in zip(episode_infos["inferences_after_switch"], switched_eps)
            if sw
        ]
        print(f"  Avg switch macro step: {np.mean(switch_steps):.1f}")
        print(f"  Avg inferences before: {np.mean(inf_before):.1f}")
        print(f"  Avg inferences after:  {np.mean(inf_after):.1f}")
        t_before = [
            t for t, sw in zip(episode_infos["time_before_switch"], switched_eps) if sw
        ]
        t_after = [
            t for t, sw in zip(episode_infos["time_after_switch"], switched_eps) if sw
        ]
        print(f"  Avg wall time before:  {np.mean(t_before):.2f}s")
        print(f"  Avg wall time after:   {np.mean(t_after):.2f}s")

    assert len(episode_successes) >= n_episodes, (
        f"Expected at least {n_episodes} episodes, got {len(episode_successes)}"
    )

    episode_infos = dict(episode_infos)  # Convert defaultdict to dict
    for key, value in episode_infos.items():
        if key in ("switched", "time_before_switch", "time_after_switch",
                    "inferences_before_switch", "inferences_after_switch",
                    "switch_macro_step"):
            assert len(value) == len(episode_successes), (
                f"Length of {key} is not equal to the number of episodes"
            )

    # process valid results
    if "valid" in episode_infos:
        valids = episode_infos["valid"]
        valid_idxs = np.where(valids)[0]
        episode_successes = [episode_successes[i] for i in valid_idxs]
        episode_successes_original = [episode_successes_original[i] for i in valid_idxs]
        episode_infos = {k: [v[i] for i in valid_idxs] for k, v in episode_infos.items()}

    # Store metadata
    episode_infos["episode_lengths_raw"] = [l * max_total for l in episode_lengths]
    episode_infos["episode_lengths_macro"] = episode_lengths
    episode_infos["max_total_action_steps"] = max_total
    episode_infos["total_action_steps_phase1"] = total_1
    episode_infos["total_action_steps_phase2"] = total_2
    episode_infos["episode_successes_original_budget"] = episode_successes_original
    episode_infos["original_max_steps"] = original_max_steps
    episode_infos["extended_max_steps"] = extended_max_steps
    return env_name, episode_successes, episode_infos


def run_gr00t_sim_policy_with_dynamic_horizon_latency(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    latency_1: int = 0,
    staleness_1: int = 0,
    n_action_steps_1: int = 8,
    latency_2: int = 0,
    staleness_2: int = 0,
    n_action_steps_2: int = 8,
    model_path: str = "",
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    n_envs: int = 8,
    video_dir: str = "",
    seed: int | None = None,
    extend_episode_budget: bool = True,
):
    """Run gr00t sim policy evaluation with dynamic horizon/latency switching."""
    embodiment_tag = get_embodiment_tag_from_env_name(env_name)

    if not video_dir:
        tag = (
            f"_dyn_lat{latency_1}-{latency_2}"
            f"_stale{staleness_1}-{staleness_2}"
            f"_ac{n_action_steps_1}-{n_action_steps_2}"
            f"_{uuid.uuid4()}"
        )
        if model_path:
            video_dir = f"data/sim_eval_videos/{model_path.split('/')[-1]}{tag}"
        else:
            video_dir = f"data/sim_eval_videos/{env_name}{tag}"
    if env_name.startswith("sim_behavior_r1_pro"):
        video_dir = None

    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=video_dir,
            max_episode_steps=max_episode_steps,
        ),
        multistep=MultiStepConfig(
            n_action_steps=max(n_action_steps_1, n_action_steps_2),  # overridden later
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )
    dynamic_config = DynamicHorizonLatencyConfig(
        latency_1=latency_1,
        staleness_1=staleness_1,
        n_action_steps_1=n_action_steps_1,
        latency_2=latency_2,
        staleness_2=staleness_2,
        n_action_steps_2=n_action_steps_2,
        extend_episode_budget=extend_episode_budget,
    )

    policy = create_gr00t_sim_policy(
        model_path, embodiment_tag, policy_client_host, policy_client_port
    )

    results = run_rollout_with_dynamic_horizon_latency(
        env_name=env_name,
        policy=policy,
        wrapper_configs=wrapper_configs,
        dynamic_config=dynamic_config,
        n_episodes=n_episodes,
        n_envs=n_envs,
        seed=seed,
    )
    print("Video saved to:", wrapper_configs.video.video_dir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run gr00t sim evaluation with dynamic horizon/latency switching."
    )
    parser.add_argument("--max_episode_steps", type=int, default=504)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--policy_client_host", type=str, default="127.0.0.1")
    parser.add_argument("--policy_client_port", type=int, default=5555)
    parser.add_argument(
        "--env_name",
        type=str,
        default="gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    )
    parser.add_argument("--n_envs", type=int, default=8)

    # Phase 1 (pre-grasp) parameters
    parser.add_argument(
        "--latency_1", type=int, default=0,
        help="Phase 1 (pre-grasp): raw env steps of simulated inference delay",
    )
    parser.add_argument(
        "--staleness_1", type=int, default=0,
        help="Phase 1: observation staleness in raw env steps",
    )
    parser.add_argument(
        "--n_action_steps_1", type=int, default=8,
        help="Phase 1: number of model actions to execute before replanning",
    )

    # Phase 2 (post-grasp) parameters
    parser.add_argument(
        "--latency_2", type=int, default=0,
        help="Phase 2 (post-grasp): raw env steps of simulated inference delay",
    )
    parser.add_argument(
        "--staleness_2", type=int, default=0,
        help="Phase 2: observation staleness in raw env steps",
    )
    parser.add_argument(
        "--n_action_steps_2", type=int, default=8,
        help="Phase 2: number of model actions to execute before replanning",
    )

    # Common parameters
    parser.add_argument(
        "--results_json_path", type=str, default="",
        help="Path to save results JSON (if empty, results are only printed)",
    )
    parser.add_argument(
        "--video_dir", type=str, default="",
        help="Directory to save episode videos (if empty, auto-generated under data/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--extend_episode_budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extend max episode steps to compensate for hold actions during latency",
    )

    args = parser.parse_args()

    # Validate policy configuration
    assert (args.model_path and not (args.policy_client_host or args.policy_client_port)) or (
        not args.model_path and args.policy_client_host and args.policy_client_port is not None
    ), (
        "Invalid policy configuration: You must provide EITHER model_path OR "
        "(policy_client_host & policy_client_port), not both.\n"
        "If all 3 arguments are provided, explicitly choose one:\n"
        "  - To use policy client: set --policy_client_host and --policy_client_port, "
        'and set --model_path ""\n'
        '  - To use model path: set --model_path, and set --policy_client_host "" '
        "(and leave --policy_client_port unset)"
    )

    results = run_gr00t_sim_policy_with_dynamic_horizon_latency(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        latency_1=args.latency_1,
        staleness_1=args.staleness_1,
        n_action_steps_1=args.n_action_steps_1,
        latency_2=args.latency_2,
        staleness_2=args.staleness_2,
        n_action_steps_2=args.n_action_steps_2,
        model_path=args.model_path,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
        n_envs=args.n_envs,
        video_dir=args.video_dir,
        seed=args.seed,
        extend_episode_budget=args.extend_episode_budget,
    )
    print("results:", results)
    print("success rate (extended budget):", np.mean(results[1]))
    if "episode_successes_original_budget" in results[2]:
        print(
            "success rate (original budget):",
            np.mean(results[2]["episode_successes_original_budget"]),
        )

    if args.results_json_path:
        import json
        import tempfile

        results_path = Path(args.results_json_path).resolve()

        skip_write = False
        if results_path.exists():
            try:
                with open(results_path) as f:
                    json.load(f)
                print(f"Results already exist (valid JSON), skipping write: {results_path}")
                skip_write = True
            except (json.JSONDecodeError, OSError):
                print(f"Existing results file is corrupt, will overwrite: {results_path}")

        if not skip_write:
            results_path.parent.mkdir(parents=True, exist_ok=True)
            orig_successes = results[2].get("episode_successes_original_budget", [])
            results_data = {
                "env_name": results[0],
                "task": args.env_name.split("/")[-1] if "/" in args.env_name else args.env_name,
                "latency_1": args.latency_1,
                "staleness_1": args.staleness_1,
                "n_action_steps_1": args.n_action_steps_1,
                "latency_2": args.latency_2,
                "staleness_2": args.staleness_2,
                "n_action_steps_2": args.n_action_steps_2,
                "n_episodes": len(results[1]),
                "max_episode_steps": results[2].get(
                    "original_max_steps", args.max_episode_steps
                ),
                "extended_max_episode_steps": results[2].get("extended_max_steps"),
                "success_rate_extended_budget": float(np.mean(results[1])),
                "success_rate_original_budget": float(np.mean(orig_successes))
                if orig_successes
                else float(np.mean(results[1])),
                "episode_successes": [bool(s) for s in results[1]],
                "episode_successes_original_budget": [bool(s) for s in orig_successes],
                "episode_lengths_raw": results[2].get("episode_lengths_raw", []),
                "episode_lengths_macro": results[2].get("episode_lengths_macro", []),
                "max_total_action_steps": results[2].get("max_total_action_steps"),
                "total_action_steps_phase1": results[2].get("total_action_steps_phase1"),
                "total_action_steps_phase2": results[2].get("total_action_steps_phase2"),
                "avg_steps_all": float(np.mean(results[2]["episode_lengths_raw"]))
                if "episode_lengths_raw" in results[2]
                else None,
                "avg_steps_succeeded": float(
                    np.mean([
                        l
                        for l, s in zip(
                            results[2].get("episode_lengths_raw", []), results[1]
                        )
                        if s
                    ])
                )
                if any(results[1])
                else None,
            }
            # Add phase-switch per-episode metrics
            for metric_key in (
                "time_before_switch", "time_after_switch",
                "inferences_before_switch", "inferences_after_switch",
                "switch_macro_step", "switched",
            ):
                if metric_key in results[2]:
                    val = results[2][metric_key]
                    serialized = []
                    for x in val:
                        if isinstance(x, (np.floating, np.integer)):
                            serialized.append(float(x))
                        elif isinstance(x, np.ndarray):
                            serialized.append(x.tolist())
                        elif x is None:
                            serialized.append(None)
                        else:
                            serialized.append(x)
                    results_data[metric_key] = serialized

            if results[2]:
                for k, v in results[2].items():
                    if k in results_data:
                        continue
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        results_data[k] = (
                            float(v) if isinstance(v, (np.floating, np.integer)) else v
                        )
                    elif isinstance(v, list):
                        serialized = []
                        for x in v:
                            if isinstance(x, (np.floating, np.integer)):
                                serialized.append(float(x))
                            elif isinstance(x, np.ndarray):
                                serialized.append(x.tolist())
                            else:
                                serialized.append(x)
                        results_data[k] = serialized
            # Atomic write
            fd, tmp_path = tempfile.mkstemp(
                dir=results_path.parent, suffix=".tmp", prefix=results_path.stem
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(results_data, f, indent=2)
                os.replace(tmp_path, results_path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            print(f"Results saved to: {results_path}")
