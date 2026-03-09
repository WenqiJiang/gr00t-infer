"""Rollout policy evaluation with latency simulation and async inference support.

This module extends the standard rollout evaluation to simulate real-world inference
latency and observation staleness. Unlike rollout_policy.py which uses MultiStepWrapper
to bundle N env steps per policy call, this module steps environments individually to
enable per-step latency control.

Key parameters:
    latency: Number of env steps of simulated inference delay.
    staleness: How old the observation is when inference is triggered.
        staleness=0 means sync (fresh obs, full freeze before actions).
        staleness=latency means fully async (stale obs, no freeze).
    n_action_steps: Number of model-predicted actions to execute before re-querying.

Usage:
    # Terminal 1 - Server (same as standard eval):
    uv run python gr00t/eval/run_gr00t_server.py \
        --model-path nvidia/GR00T-N1.6-3B \
        --embodiment-tag ROBOCASA_PANDA_OMRON \
        --use-sim-policy-wrapper

    # Terminal 2 - Client with latency simulation:
    uv run python gr00t/eval/rollout_policy_with_latency.py \
        --policy_client_host 127.0.0.1 \
        --policy_client_port 5555 \
        --env_name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
        --n_episodes 10 --n_envs 5 \
        --latency 10 --staleness 5 --n_action_steps 8
"""

import argparse
import collections
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import time
from typing import Any
import uuid

from gr00t.eval.rollout_policy import VideoConfig, create_gr00t_sim_policy, get_gym_env
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.policy import BasePolicy
import gymnasium as gym
import numpy as np
from tqdm import tqdm


@dataclass
class LatencyConfig:
    """Configuration for latency simulation.

    Attributes:
        latency: Number of env steps of simulated inference delay.
        staleness: How old the observation is when inference is triggered.
            staleness=0 → sync: fresh obs, full latency freeze before model actions.
            staleness=latency → fully async: stale obs, no freeze after inference.
            Must satisfy 0 <= staleness <= latency.
        n_action_steps: Number of model-predicted actions to execute before
            re-querying the model.
    """

    latency: int = 0
    staleness: int = 0
    n_action_steps: int = 8


def create_eval_env_no_multistep(
    env_name: str,
    env_idx: int,
    total_n_envs: int,
    video_config: VideoConfig,
) -> gym.Env:
    """Create an evaluation environment with video recording but no MultiStepWrapper."""
    env = get_gym_env(env_name, env_idx, total_n_envs)
    if video_config.video_dir is not None:
        from gr00t.eval.sim.wrapper.video_recording_wrapper import (
            VideoRecorder,
            VideoRecordingWrapper,
        )

        video_recorder = VideoRecorder.create_h264(
            fps=video_config.fps,
            codec=video_config.codec,
            input_pix_fmt=video_config.input_pix_fmt,
            crf=video_config.crf,
            thread_type=video_config.thread_type,
            thread_count=video_config.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(video_config.video_dir),
            steps_per_render=video_config.steps_per_render,
            max_episode_steps=video_config.max_episode_steps,
            overlay_text=video_config.overlay_text,
        )
    return env


def _batch_observations(per_env_obs: list[dict]) -> dict[str, Any]:
    """Stack per-env observations into a batched dict for the policy.

    For video/state keys: adds T=1 temporal dim, then stacks across envs → (B, 1, ...).
    For task/annotation/language keys: collects into a tuple of strings → (B,).
    """
    batched = {}
    keys = per_env_obs[0].keys()
    for key in keys:
        values = [obs[key] for obs in per_env_obs]
        if isinstance(values[0], str):
            batched[key] = tuple(values)
        elif isinstance(values[0], np.ndarray):
            # Add temporal dimension T=1, then stack for batch: (B, 1, ...)
            stacked = np.stack([np.expand_dims(v, axis=0) for v in values], axis=0)
            batched[key] = stacked
        else:
            # Fallback: just collect as-is
            batched[key] = values
    return batched


def _make_hold_action(last_action: dict[str, np.ndarray] | None, action_space: gym.spaces.Dict):
    """Create a hold/no-op action. Uses last_action if available, else zeros."""
    if last_action is not None:
        return {k: v.copy() for k, v in last_action.items()}
    noop = {}
    for key, space in action_space.items():
        noop[key] = np.zeros(space.shape, dtype=np.float32)
    return noop


def _decompose_action_chunk(
    action_chunk: dict[str, np.ndarray],
    env_idx: int,
    step_idx: int,
) -> dict[str, np.ndarray]:
    """Extract a single-step action for one env from a batched action chunk.

    Policy returns {key: (B, T, D)}. This extracts {key: (D,)} for one env and step.
    """
    return {key: value[env_idx, step_idx] for key, value in action_chunk.items()}


def run_rollout_with_latency(
    env_name: str,
    policy: BasePolicy,
    latency_config: LatencyConfig,
    video_config: VideoConfig,
    max_episode_steps: int = 720,
    n_episodes: int = 10,
    n_envs: int = 1,
    terminate_on_success: bool = True,
) -> Any:
    """Run policy rollouts with latency simulation in parallel environments.

    Args:
        env_name: Name of the gymnasium environment.
        policy: Policy instance (local or PolicyClient).
        latency_config: Latency, staleness, and replan configuration.
        video_config: Video recording configuration.
        max_episode_steps: Maximum env steps per episode.
        n_episodes: Number of episodes to run.
        n_envs: Number of parallel environments.
        terminate_on_success: Whether to end episode early on success.

    Returns:
        Tuple of (env_name, episode_successes, episode_infos).
    """
    latency = latency_config.latency
    staleness = latency_config.staleness
    n_action_steps = latency_config.n_action_steps
    assert 0 <= staleness <= latency, f"staleness ({staleness}) must be in [0, latency={latency}]"

    start_time = time.time()
    n_episodes = max(n_episodes, n_envs)
    print(
        f"Running {n_episodes} episodes for {env_name} with {n_envs} vec envs "
        f"(latency={latency}, staleness={staleness}, n_action_steps={n_action_steps})"
    )

    # Create vectorized envs WITHOUT MultiStepWrapper
    env_fns = [
        partial(
            create_eval_env_no_multistep,
            env_name=env_name,
            env_idx=idx,
            total_n_envs=n_envs,
            video_config=video_config,
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

    # Get action space from the vectorized env's single-env action space
    single_action_space = env.single_action_space

    # ---- Per-env state ----
    action_queues: list[collections.deque] = [collections.deque() for _ in range(n_envs)]
    obs_histories: list[list[dict]] = [[] for _ in range(n_envs)]
    last_actions: list[dict | None] = [None] * n_envs
    is_first_inference: list[bool] = [True] * n_envs
    step_counts: list[int] = [0] * n_envs

    # ---- Episode result tracking ----
    episode_successes: list[bool] = []
    episode_lengths: list[int] = []
    episode_infos: dict[str, list] = defaultdict(list)
    current_successes: list[bool] = [False] * n_envs
    completed_episodes = 0

    # ---- Initial reset ----
    observations, _ = env.reset()
    policy.reset()
    for env_idx in range(n_envs):
        per_env_obs = {k: v[env_idx] for k, v in observations.items()}
        obs_histories[env_idx] = [per_env_obs]

    pbar = tqdm(total=n_episodes, desc="Episodes")

    while completed_episodes < n_episodes:
        # --- Replan if any env's action queue is empty ---
        needs_replan = any(len(q) == 0 for q in action_queues)

        if needs_replan:
            # Check which envs need first-inference freeze vs real inference
            envs_needing_inference = []
            for env_idx in range(n_envs):
                if len(action_queues[env_idx]) > 0:
                    continue  # This env still has actions queued
                if is_first_inference[env_idx] and latency > 0:
                    # First inference with latency: freeze with hold actions
                    hold = _make_hold_action(last_actions[env_idx], single_action_space)
                    for _ in range(latency):
                        action_queues[env_idx].append(hold)
                    is_first_inference[env_idx] = False
                else:
                    envs_needing_inference.append(env_idx)
                    if is_first_inference[env_idx]:
                        # latency == 0, first inference
                        is_first_inference[env_idx] = False

            if envs_needing_inference:
                # Gather stale observations for envs that need inference
                stale_obs_list = []
                for env_idx in envs_needing_inference:
                    history = obs_histories[env_idx]
                    stale_idx = max(0, len(history) - 1 - staleness)
                    stale_obs_list.append(history[stale_idx])

                # Batch stale observations and query policy
                batched_obs = _batch_observations(stale_obs_list)
                action_chunk, _ = policy.get_action(batched_obs)

                # Determine action horizon from the chunk
                sample_key = next(iter(action_chunk))
                action_horizon = action_chunk[sample_key].shape[1]
                padding = latency - staleness

                for batch_idx, env_idx in enumerate(envs_needing_inference):
                    # Prepend hold-action padding for remaining inference wait
                    hold = _make_hold_action(last_actions[env_idx], single_action_space)
                    for _ in range(padding):
                        action_queues[env_idx].append(hold)

                    # Enqueue model-predicted actions
                    n_model_actions = min(n_action_steps, action_horizon)
                    for step_idx in range(n_model_actions):
                        act = _decompose_action_chunk(action_chunk, batch_idx, step_idx)
                        action_queues[env_idx].append(act)

        # --- Pop one action per env and step ---
        per_env_actions = []
        for env_idx in range(n_envs):
            act = action_queues[env_idx].popleft()
            last_actions[env_idx] = act
            per_env_actions.append(act)

        # Stack into vectorized action format: {key: (n_envs, D)}
        vec_action = {}
        for key in per_env_actions[0]:
            shapes = [a[key].shape for a in per_env_actions]
            if len(set(shapes)) > 1:
                print(
                    f"[DEBUG] Shape mismatch for action key '{key}': "
                    f"{[(i, s) for i, s in enumerate(shapes)]}"
                )
            vec_action[key] = np.stack([a[key] for a in per_env_actions], axis=0)

        next_obs, rewards, terminations, truncations, env_infos = env.step(vec_action)

        # --- Update per-env state ---
        for env_idx in range(n_envs):
            step_counts[env_idx] += 1

            # Track success
            if "success" in env_infos:
                env_success = env_infos["success"][env_idx]
                if isinstance(env_success, (list, np.ndarray)):
                    env_success = bool(np.any(env_success))
                else:
                    env_success = bool(env_success)
                current_successes[env_idx] |= env_success

            if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                final_success = env_infos["final_info"][env_idx].get("success", False)
                if isinstance(final_success, (list, np.ndarray)):
                    final_success = bool(np.any(final_success))
                else:
                    final_success = bool(final_success)
                current_successes[env_idx] |= final_success

            # NOTE: We do NOT use terminate_on_success to end episodes early here
            # because AsyncVectorEnv only auto-resets sub-envs when they return
            # done=True internally. Manually ending an episode without the env
            # resetting would cause the next "episode" to continue from the same
            # env state, producing phantom results. Instead, we let episodes run
            # to natural completion and track success via the sticky |= above.
            should_terminate = step_counts[env_idx] >= max_episode_steps

            episode_done = terminations[env_idx] or truncations[env_idx] or should_terminate

            if episode_done:
                # Collect final info
                if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                    fi = env_infos["final_info"][env_idx]
                    fi_success = fi.get("success", False)
                    if isinstance(fi_success, (list, np.ndarray)):
                        fi_success = bool(np.any(fi_success))
                    current_successes[env_idx] |= fi_success
                if "task_progress" in env_infos:
                    tp = env_infos["task_progress"][env_idx]
                    episode_infos["task_progress"].append(tp[-1] if isinstance(tp, list) else tp)
                if "q_score" in env_infos:
                    qs = env_infos["q_score"][env_idx]
                    episode_infos["q_score"].append(
                        np.max(qs) if isinstance(qs, (list, np.ndarray)) else qs
                    )
                if "valid" in env_infos:
                    vl = env_infos["valid"][env_idx]
                    episode_infos["valid"].append(all(vl) if isinstance(vl, list) else bool(vl))

                # Record episode results
                episode_successes.append(current_successes[env_idx])
                episode_lengths.append(step_counts[env_idx])
                print(
                    f"[DEBUG] Episode {len(episode_successes)} done: "
                    f"env={env_idx}, steps={step_counts[env_idx]}, "
                    f"success={current_successes[env_idx]}, "
                    f"terminated={terminations[env_idx]}, "
                    f"truncated={truncations[env_idx]}"
                )

                # Update completed count (respecting valid flag)
                if "valid" in episode_infos:
                    if episode_infos["valid"][-1]:
                        completed_episodes += 1
                        pbar.update(1)
                else:
                    completed_episodes += 1
                    pbar.update(1)

                # Reset per-env state for next episode
                current_successes[env_idx] = False
                step_counts[env_idx] = 0
                action_queues[env_idx].clear()
                is_first_inference[env_idx] = True
                # NOTE: intentionally keep last_actions[env_idx] (don't set to None)
                # so that _make_hold_action uses the model's output shape rather than
                # falling back to gym action_space.shape, which may differ.

                # The vectorized env auto-resets; next_obs already has the new initial obs
                new_obs = {k: v[env_idx] for k, v in next_obs.items()}
                obs_histories[env_idx] = [new_obs]
            else:
                # Store observation in history
                per_env_obs = {k: v[env_idx] for k, v in next_obs.items()}
                obs_histories[env_idx].append(per_env_obs)

    pbar.close()
    env.reset()
    env.close()
    print(f"Collecting {n_episodes} episodes took {time.time() - start_time:.1f} seconds")

    assert len(episode_successes) >= n_episodes, (
        f"Expected at least {n_episodes} episodes, got {len(episode_successes)}"
    )

    episode_infos = dict(episode_infos)
    for key, value in episode_infos.items():
        assert len(value) == len(episode_successes), (
            f"Length of {key} is not equal to the number of episodes"
        )

    # Process valid results
    if "valid" in episode_infos:
        valids = episode_infos["valid"]
        valid_idxs = np.where(valids)[0]
        episode_successes = [episode_successes[i] for i in valid_idxs]
        episode_infos = {k: [v[i] for i in valid_idxs] for k, v in episode_infos.items()}

    # Report average episode length
    avg_steps_all = np.mean(episode_lengths)
    print(f"Avg steps (all episodes): {avg_steps_all:.1f}")
    success_lengths = [l for l, s in zip(episode_lengths, episode_successes) if s]
    if success_lengths:
        print(f"Avg steps (succeeded):    {np.mean(success_lengths):.1f}")
    else:
        print("Avg steps (succeeded):    N/A (no successes)")

    episode_infos["episode_lengths"] = episode_lengths
    return env_name, episode_successes, episode_infos


def run_gr00t_sim_policy_with_latency(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    latency: int = 0,
    staleness: int = 0,
    n_action_steps: int = 8,
    model_path: str = "",
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    n_envs: int = 8,
    video_dir: str = "",
):
    """Run gr00t sim policy evaluation with latency simulation."""
    embodiment_tag = get_embodiment_tag_from_env_name(env_name)

    if not video_dir:
        tag = f"_lat{latency}_stale{staleness}_rp{n_action_steps}_{uuid.uuid4()}"
        if model_path:
            video_dir = f"data/sim_eval_videos/{model_path.split('/')[-1]}{tag}"
        else:
            video_dir = f"data/sim_eval_videos/{env_name}{tag}"
    if env_name.startswith("sim_behavior_r1_pro"):
        video_dir = None

    video_config = VideoConfig(
        video_dir=video_dir,
        max_episode_steps=max_episode_steps,
    )
    latency_config = LatencyConfig(
        latency=latency,
        staleness=staleness,
        n_action_steps=n_action_steps,
    )

    policy = create_gr00t_sim_policy(
        model_path, embodiment_tag, policy_client_host, policy_client_port
    )

    results = run_rollout_with_latency(
        env_name=env_name,
        policy=policy,
        latency_config=latency_config,
        video_config=video_config,
        max_episode_steps=max_episode_steps,
        n_episodes=n_episodes,
        n_envs=n_envs,
    )
    print("Video saved to:", video_config.video_dir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run gr00t sim evaluation with latency simulation."
    )
    parser.add_argument("--max_episode_steps", type=int, default=720)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--policy_client_host", type=str, default="127.0.0.1")
    parser.add_argument("--policy_client_port", type=int, default=5555)
    parser.add_argument(
        "--env_name",
        type=str,
        default="robocasa_panda_omron/OpenDrawer_PandaOmron_Env",
    )
    parser.add_argument("--n_envs", type=int, default=8)

    # Latency simulation parameters
    parser.add_argument(
        "--latency",
        type=int,
        default=0,
        help="Number of env steps of simulated inference delay",
    )
    parser.add_argument(
        "--staleness",
        type=int,
        default=0,
        help="Observation staleness in env steps (0=sync, =latency for fully async)",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=8,
        help="Number of model actions to execute before replanning",
    )
    parser.add_argument(
        "--results_json_path",
        type=str,
        default="",
        help="Path to save results JSON (if empty, results are only printed)",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="",
        help="Directory to save episode videos (if empty, auto-generated under data/)",
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

    results = run_gr00t_sim_policy_with_latency(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        latency=args.latency,
        staleness=args.staleness,
        n_action_steps=args.n_action_steps,
        model_path=args.model_path,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
        n_envs=args.n_envs,
        video_dir=args.video_dir,
    )
    print("results:", results)
    print("success rate:", np.mean(results[1]))

    if args.results_json_path:
        import json

        results_path = Path(args.results_json_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_data = {
            "env_name": results[0],
            "task": args.env_name.split("/")[-1] if "/" in args.env_name else args.env_name,
            "latency": args.latency,
            "staleness": args.staleness,
            "n_action_steps": args.n_action_steps,
            "n_episodes": len(results[1]),
            "success_rate": float(np.mean(results[1])),
            "episode_successes": [bool(s) for s in results[1]],
            "episode_lengths": results[2].get("episode_lengths", []),
            "avg_steps_all": float(np.mean(results[2]["episode_lengths"]))
            if "episode_lengths" in results[2]
            else None,
            "avg_steps_succeeded": float(
                np.mean([
                    l
                    for l, s in zip(results[2].get("episode_lengths", []), results[1])
                    if s
                ])
            )
            if any(results[1])
            else None,
        }
        if results[2]:
            for k, v in results[2].items():
                serialized = []
                for x in v:
                    if isinstance(x, (np.floating, np.integer)):
                        serialized.append(float(x))
                    elif isinstance(x, np.ndarray):
                        serialized.append(x.tolist())
                    else:
                        serialized.append(x)
                results_data[k] = serialized
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to: {results_path}")
