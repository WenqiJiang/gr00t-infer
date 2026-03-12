"""Rollout policy evaluation with latency simulation.

Extends rollout_policy.py by prepending hold (repeat-last) actions to simulate
real-world inference latency. Uses the same MultiStepWrapper infrastructure,
so behavior is identical to rollout_policy.py when latency=0 and staleness=0.

Key parameters:
    latency: Number of raw env steps of simulated inference delay.
    staleness: How old the observation is when inference is triggered (in raw steps).
        staleness=0 → sync: fresh obs, full latency freeze before model actions.
        staleness=latency → fully async: stale obs, no freeze after inference.
    n_action_steps: Number of model-predicted actions to execute before re-querying.

The latency is implemented by setting MultiStepWrapper.n_action_steps to
(latency - staleness + n_action_steps) and prepending hold actions to the
model's output. The observation staleness is approximated at the macro-step
level (each macro step = padding + n_action_steps raw env steps).

Usage:
    # Terminal 1 - Server (same as standard eval):
    uv run python gr00t/eval/run_gr00t_server.py \\
        --model-path nvidia/GR00T-N1.6-3B \\
        --embodiment-tag ROBOCASA_PANDA_OMRON \\
        --use-sim-policy-wrapper

    # Terminal 2 - Client with latency simulation:
    uv run python gr00t/eval/rollout_policy_with_latency.py \\
        --policy_client_host 127.0.0.1 \\
        --policy_client_port 5555 \\
        --env_name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \\
        --n_episodes 10 --n_envs 5 \\
        --latency 10 --staleness 5 --n_action_steps 8

NOTE on reproducibility: even with fixed seeds (--seed), results may vary across
runs due to GPU non-determinism (CUDA, cuDNN), async vectorized env scheduling,
and floating-point non-associativity in MuJoCo physics. The best way to report
reliable accuracy is to run a large number of episodes (50+) and report mean ± stderr.
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


@dataclass
class LatencyConfig:
    """Configuration for latency simulation.

    Attributes:
        latency: Number of raw env steps of simulated inference delay.
        staleness: Observation staleness in raw env steps.
            0 = sync (fresh obs, full latency freeze).
            = latency for fully async (stale obs, no freeze).
        n_action_steps: Number of model actions to execute per replan.
    """

    latency: int = 0
    staleness: int = 0
    n_action_steps: int = 8
    extend_episode_budget: bool = True


def run_rollout_with_latency(
    env_name: str,
    policy: BasePolicy,
    wrapper_configs: WrapperConfigs,
    latency_config: LatencyConfig,
    n_episodes: int = 10,
    n_envs: int = 1,
    seed: int | None = None,
) -> Any:
    """Run policy rollouts with latency simulation.

    Mirrors run_rollout_gymnasium_policy exactly, with latency simulated by
    prepending hold actions to the model's action chunk. When latency=0 and
    staleness=0, behavior is identical to the original.

    Args:
        env_name: Name of the gymnasium environment to use.
        policy: Policy instance (local or PolicyClient).
        wrapper_configs: Configuration for environment wrappers.
            NOTE: wrapper_configs.multistep.n_action_steps will be overridden
            to (padding + n_action_steps) to accommodate hold-action prepending.
        latency_config: Latency, staleness, and replan configuration.
        n_episodes: Number of episodes to run.
        n_envs: Number of parallel environments.

    Returns:
        Tuple of (env_name, episode_successes, episode_infos).
    """
    latency = latency_config.latency
    staleness = latency_config.staleness
    n_action_steps = latency_config.n_action_steps
    padding = latency - staleness
    assert 0 <= staleness <= latency, (
        f"staleness ({staleness}) must be in [0, latency={latency}]"
    )

    # Total raw env steps per macro step = hold padding + model actions.
    # When latency=0: total == n_action_steps (identical to original).
    total_action_steps = padding + n_action_steps

    # Extend episode budget so the robot gets the same number of effective
    # (non-hold) action steps as the baseline without latency.
    # factor = total_action_steps / n_action_steps (>= 1.0).
    original_max_steps = wrapper_configs.multistep.max_episode_steps
    if latency_config.extend_episode_budget and padding > 0:
        factor = total_action_steps / n_action_steps
        extended_max_steps = int(original_max_steps * factor)
        wrapper_configs.multistep.max_episode_steps = extended_max_steps
        wrapper_configs.video.max_episode_steps = extended_max_steps
        print(
            f"Extended episode budget: {original_max_steps} -> {extended_max_steps} "
            f"raw steps (factor={factor:.2f})"
        )
    else:
        extended_max_steps = original_max_steps

    # Override MultiStepWrapper's n_action_steps to include hold padding.
    wrapper_configs.multistep.n_action_steps = total_action_steps

    # Enable intermediate observation caching for sub-macro-step staleness.
    if staleness > 0:
        wrapper_configs.multistep.cache_intermediate_obs = True

    start_time = time.time()
    n_envs = min(n_envs, n_episodes)
    print(
        f"Running collecting {n_episodes} episodes for {env_name} with {n_envs} vec envs"
        + (f" (latency={latency}, staleness={staleness})" if latency > 0 else "")
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

    # Storage for results (identical to original)
    episode_lengths = []
    current_rewards = [0] * n_envs
    current_lengths = [0] * n_envs
    completed_episodes = 0
    current_successes = [False] * n_envs
    episode_successes = []
    episode_infos = defaultdict(list)

    # Track success at original budget (before extension) for fair comparison.
    # An episode "succeeds at original budget" if it succeeded within the
    # original_max_steps raw steps.
    current_successes_original = [False] * n_envs
    episode_successes_original = []

    # -- Latency state --
    # Last model-predicted actions for hold-action generation: {key: (B, n_action_steps, D)}
    last_model_actions = None

    # -- Staleness state: per-env raw observation cache --
    # Each cache stores individual raw-step observations so that staleness=N
    # means the policy sees an observation from exactly N raw env steps ago.
    if staleness > 0:
        raw_obs_caches = [
            collections.deque(maxlen=staleness + 1) for _ in range(n_envs)
        ]

    # Initial reset
    observations, _ = env.reset(seed=seed)
    policy.reset()

    # Seed staleness caches with initial observation (treated as 1 raw obs).
    if staleness > 0:
        for env_idx in range(n_envs):
            per_env_obs = {k: v[env_idx] for k, v in observations.items()}
            raw_obs_caches[env_idx].append(per_env_obs)

    i = 0

    pbar = tqdm(total=n_episodes, desc="Episodes")
    while completed_episodes < n_episodes:
        # -- Select observation for policy (fresh or stale) --
        if staleness > 0 and last_model_actions is not None:
            # Build batched stale observation from per-env raw-step caches.
            # Each cache has up to (staleness + 1) raw observations.
            # Index 0 = oldest, -1 = newest. We want the one that is
            # `staleness` raw steps behind the latest.
            stale_obs_parts = {k: [] for k in observations}
            for env_idx in range(n_envs):
                cache = raw_obs_caches[env_idx]
                # Pick observation `staleness` steps back, or oldest available.
                idx = max(0, len(cache) - 1 - staleness)
                for k in observations:
                    stale_obs_parts[k].append(cache[idx][k])
            policy_obs = {}
            for k, parts in stale_obs_parts.items():
                if isinstance(parts[0], str):
                    policy_obs[k] = tuple(parts)
                else:
                    policy_obs[k] = np.stack(parts, axis=0)
        else:
            policy_obs = observations

        # -- Query policy --
        actions, _ = policy.get_action(policy_obs)

        # -- Prepend hold actions for latency --
        # During hold, the robot should stay stationary: delta actions (position,
        # rotation, base motion) are zeroed, while discrete state actions (gripper,
        # control mode) retain their last value so the robot doesn't drop objects.
        if padding > 0:
            padded_actions = {}
            for key, value in actions.items():
                # value shape: (B, T_model, D)
                B = value.shape[0]
                D = value.shape[-1]

                if key in HOLD_KEEP_LAST_KEYS:
                    # Gripper / control mode: repeat last value to maintain state.
                    if last_model_actions is not None:
                        hold_single = last_model_actions[key][:, -1:, :]
                        hold = np.repeat(hold_single, padding, axis=1)
                    else:
                        hold = np.zeros((B, padding, D), dtype=value.dtype)
                else:
                    # Delta actions: zero out so the robot stays stationary.
                    hold = np.zeros((B, padding, D), dtype=value.dtype)

                # Take first n_action_steps from model output
                model_acts = value[:, :n_action_steps, :]  # (B, n_action_steps, D)
                padded_actions[key] = np.concatenate(
                    [hold, model_acts], axis=1
                )  # (B, total_action_steps, D)

            # Save model actions for next hold generation
            last_model_actions = {k: v[:, :n_action_steps, :] for k, v in actions.items()}
            actions = padded_actions
        else:
            # No padding; save model actions for potential future use
            last_model_actions = {k: v[:, :n_action_steps, :] for k, v in actions.items()}

        # -- Step environment (identical to original from here) --
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
            raw_steps_so_far = current_lengths[env_idx] * total_action_steps
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
                # Accumulate results
                episode_lengths.append(current_lengths[env_idx])
                episode_successes.append(current_successes[env_idx])
                episode_successes_original.append(current_successes_original[env_idx])
                n_act = total_action_steps
                raw_steps = current_lengths[env_idx] * n_act
                print(
                    f"[DEBUG] Episode {len(episode_successes)} done: "
                    f"env={env_idx}, "
                    f"steps={current_lengths[env_idx]} "
                    f"(~{raw_steps} raw), "
                    f"success={current_successes[env_idx]}"
                    f"{'' if current_successes[env_idx] == current_successes_original[env_idx] else f' (original_budget={current_successes_original[env_idx]})'}"
                    f", terminated={terminations[env_idx]}, "
                    f"truncated={truncations[env_idx]}"
                )
                # Reset trackers for this environment.
                current_successes[env_idx] = False
                current_successes_original[env_idx] = False
                # only update completed_episodes if valid
                if "valid" in episode_infos:
                    if episode_infos["valid"][-1]:
                        completed_episodes += 1
                        pbar.update(1)
                else:
                    # envs don't return valid
                    completed_episodes += 1
                    pbar.update(1)
                current_rewards[env_idx] = 0
                current_lengths[env_idx] = 0

                # Reset hold actions for this env so the new episode
                # starts with zero-hold instead of previous episode's actions.
                if last_model_actions is not None and padding > 0:
                    for k in last_model_actions:
                        last_model_actions[k][env_idx] = 0.0

                # Reset staleness cache for this env
                if staleness > 0:
                    raw_obs_caches[env_idx].clear()

        observations = next_obs

        # Cache raw-step observations for staleness.
        # _intermediate_obs contains one raw observation per inner env step,
        # giving us sub-macro-step resolution for staleness.
        if staleness > 0 and "_intermediate_obs" in env_infos:
            for env_idx in range(n_envs):
                intermediate = env_infos["_intermediate_obs"][env_idx]
                for raw_obs in intermediate:
                    raw_obs_caches[env_idx].append(raw_obs)

    pbar.close()

    env.reset()
    env.close()
    print(f"Collecting {n_episodes} episodes took {time.time() - start_time} seconds")

    # Report average episode length and success rates
    n_act = total_action_steps
    raw_lengths = [l * n_act for l in episode_lengths]
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

    assert len(episode_successes) >= n_episodes, (
        f"Expected at least {n_episodes} episodes, got {len(episode_successes)}"
    )

    episode_infos = dict(episode_infos)  # Convert defaultdict to dict
    for key, value in episode_infos.items():
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

    # Store raw env step counts (macro_steps * total_action_steps) so that
    # reported lengths include hold actions during latency.
    episode_infos["episode_lengths_raw"] = [l * total_action_steps for l in episode_lengths]
    episode_infos["episode_lengths_macro"] = episode_lengths
    episode_infos["total_action_steps_per_macro"] = total_action_steps
    episode_infos["episode_successes_original_budget"] = episode_successes_original
    episode_infos["original_max_steps"] = original_max_steps
    episode_infos["extended_max_steps"] = extended_max_steps
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
    seed: int | None = None,
    extend_episode_budget: bool = True,
):
    """Run gr00t sim policy evaluation with latency simulation."""
    embodiment_tag = get_embodiment_tag_from_env_name(env_name)

    if not video_dir:
        tag = f"_lat{latency}_stale{staleness}_ac{n_action_steps}_{uuid.uuid4()}"
        if model_path:
            video_dir = f"data/sim_eval_videos/{model_path.split('/')[-1]}{tag}"
        else:
            video_dir = f"data/sim_eval_videos/{env_name}{tag}"
    if env_name.startswith("sim_behavior_r1_pro"):
        # BEHAVIOR sim will crash if decord is imported in video_utils.py
        video_dir = None

    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=video_dir,
            max_episode_steps=max_episode_steps,
        ),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,  # Will be overridden in run_rollout_with_latency
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )
    latency_config = LatencyConfig(
        latency=latency,
        staleness=staleness,
        n_action_steps=n_action_steps,
        extend_episode_budget=extend_episode_budget,
    )

    policy = create_gr00t_sim_policy(
        model_path, embodiment_tag, policy_client_host, policy_client_port
    )

    results = run_rollout_with_latency(
        env_name=env_name,
        policy=policy,
        wrapper_configs=wrapper_configs,
        latency_config=latency_config,
        n_episodes=n_episodes,
        n_envs=n_envs,
        seed=seed,
    )
    print("Video saved to:", wrapper_configs.video.video_dir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run gr00t sim evaluation with latency simulation."
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

    # Latency simulation parameters
    parser.add_argument(
        "--latency",
        type=int,
        default=0,
        help="Number of raw env steps of simulated inference delay",
    )
    parser.add_argument(
        "--staleness",
        type=int,
        default=0,
        help="Observation staleness in raw env steps (0=sync, =latency for fully async)",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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

        results_path = Path(args.results_json_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        orig_successes = results[2].get("episode_successes_original_budget", [])
        results_data = {
            "env_name": results[0],
            "task": args.env_name.split("/")[-1] if "/" in args.env_name else args.env_name,
            "latency": args.latency,
            "staleness": args.staleness,
            "n_action_steps": args.n_action_steps,
            "n_episodes": len(results[1]),
            "max_episode_steps": results[2].get("original_max_steps", args.max_episode_steps),
            "extended_max_episode_steps": results[2].get("extended_max_steps"),
            "success_rate_extended_budget": float(np.mean(results[1])),
            "success_rate_original_budget": float(np.mean(orig_successes))
            if orig_successes
            else float(np.mean(results[1])),
            "episode_successes": [bool(s) for s in results[1]],
            "episode_successes_original_budget": [bool(s) for s in orig_successes],
            "episode_lengths_raw": results[2].get("episode_lengths_raw", []),
            "episode_lengths_macro": results[2].get("episode_lengths_macro", []),
            "total_action_steps_per_macro": results[2].get("total_action_steps_per_macro"),
            "avg_steps_all": float(np.mean(results[2]["episode_lengths_raw"]))
            if "episode_lengths_raw" in results[2]
            else None,
            "avg_steps_succeeded": float(
                np.mean([
                    l
                    for l, s in zip(results[2].get("episode_lengths_raw", []), results[1])
                    if s
                ])
            )
            if any(results[1])
            else None,
        }
        if results[2]:
            for k, v in results[2].items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    results_data[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
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
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to: {results_path}")
