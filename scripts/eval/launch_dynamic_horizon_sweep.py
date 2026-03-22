"""Launch a dynamic-horizon latency sweep across RoboCasa tasks.

Launches persistent policy servers (one per GPU) and schedules client experiments
in rounds using rollout_policy_dynamic_horizon_latency.py. Servers are reused
across experiments to avoid repeated model loading.

The sweep grid is: tasks x (latency_1, nact_1) x (latency_2, nact_2).

Usage:
    uv run python scripts/eval/launch_dynamic_horizon_sweep.py \\
        --tasks CoffeeSetupMug_PandaOmron_Env \\
        --latencies-1 0 5 10 --n-action-steps-1 4 8 \\
        --latencies-2 0 5 10 --n-action-steps-2 4 8
    uv run python scripts/eval/launch_dynamic_horizon_sweep.py \\
        --tasks CoffeeSetupMug_PandaOmron_Env OpenDrawer_PandaOmron_Env \\
        --latencies-1 10 --n-action-steps-1 8 \\
        --latencies-2 0 5 --n-action-steps-2 4 8

    # Auto-generate lat2 as powers of 2 up to each lat1:
    uv run python scripts/eval/launch_dynamic_horizon_sweep.py \\
        --tasks CoffeeSetupMug_PandaOmron_Env \\
        --latencies-1 0 4 8 16 --n-action-steps-1 8 \\
        --auto-latencies-2 pow2 --n-action-steps-2 4 8

    # Auto-generate lat2 as just [0, lat1] (minimal):
    uv run python scripts/eval/launch_dynamic_horizon_sweep.py \\
        --tasks CoffeeSetupMug_PandaOmron_Env \\
        --latencies-1 0 4 8 16 --n-action-steps-1 8 \\
        --auto-latencies-2 endpoints --n-action-steps-2 4 8

After completion, view results:
    uv run python scripts/eval/summarize_latency_sweep.py \\
        data/dynamic_horizon_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/

NOTE on reproducibility: even with fixed seeds (--seed), results may vary across
runs due to GPU non-determinism (CUDA, cuDNN), async vectorized env scheduling,
and floating-point non-associativity in MuJoCo physics. The best way to report
reliable accuracy is to run a large number of episodes (50+) and report mean +/- stderr.
"""

import dataclasses
import itertools
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import time

import tyro

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    tasks: list[str] = dataclasses.field(
        default_factory=lambda: ["CoffeeSetupMug_PandaOmron_Env"]
    )
    """RoboCasa task class names to sweep over."""

    latencies_1: list[int] = dataclasses.field(default_factory=lambda: [0, 5, 10, 20])
    """Phase 1 (pre-grasp): simulated inference latency in env steps."""

    n_action_steps_1: list[int] = dataclasses.field(default_factory=lambda: [1, 4, 8, 16])
    """Phase 1: number of model actions to execute before replanning."""

    staleness_1: list[int] = dataclasses.field(default_factory=lambda: [0])
    """Phase 1: observation staleness values."""

    latencies_2: list[int] = dataclasses.field(default_factory=lambda: [0, 5, 10, 20])
    """Phase 2 (post-grasp): simulated inference latency in env steps."""

    n_action_steps_2: list[int] = dataclasses.field(default_factory=lambda: [1, 4, 8, 16])
    """Phase 2: number of model actions to execute before replanning."""

    staleness_2: list[int] = dataclasses.field(default_factory=lambda: [0])
    """Phase 2: observation staleness values."""

    model_path: str = "nvidia/GR00T-N1.6-3B"
    """Path to the model checkpoint."""

    embodiment_tag: str = "ROBOCASA_PANDA_OMRON"
    """Embodiment tag for the policy server."""

    env_prefix: str = "robocasa_panda_omron"
    """Environment name prefix (env_name = {env_prefix}/{task})."""

    base_port: int = 5555
    """Base ZMQ port for policy servers (server on GPU i uses base_port + i)."""

    num_trials: int = 50
    """Number of episodes per experiment."""

    n_envs: int = 10
    """Number of parallel simulation environments per experiment."""

    max_episode_steps: int = 720
    """Maximum env steps per episode."""

    num_gpus: int = 0
    """Number of GPUs to use. 0 = auto-detect."""

    log_dir: str = ""
    """Directory for logs and results. Auto-generated if empty."""

    overwrite: bool = False
    """Re-run experiments even if results already exist."""

    server_wait: int = 300
    """Seconds to wait for each server to become ready."""

    seed: int | None = None
    """Random seed for reproducibility. When set, results are stored in a
    seed-specific subdirectory (e.g. .../seed42/) to support variance analysis.
    When not set, uses seed=42 with no subdirectory (default behavior)."""

    extend_episode_budget: bool = True
    """Extend max episode steps to compensate for hold actions during latency."""

    auto_latencies_2: str = ""
    """Auto-generate phase 2 latencies per lat1 instead of using --latencies-2.
    Options:
      ""           — disabled, use explicit --latencies-2 (default)
      "pow2"       — [0, 1, 2, 4, ..., lat1] (powers of 2 up to lat1)
      "endpoints"  — [0, lat1] (only extremes, minimal experiment set)"""

    client_python: str = ""
    """Python interpreter for client processes (e.g., a sim-specific venv).
    If empty, auto-detected from env_prefix (robocasa uses its own venv)."""


@dataclasses.dataclass
class Experiment:
    task: str
    latency_1: int
    n_action_steps_1: int
    staleness_1: int
    latency_2: int
    n_action_steps_2: int
    staleness_2: int


def detect_num_gpus() -> int:
    if shutil.which("nvidia-smi"):
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        return len([line for line in result.stdout.strip().splitlines() if line])
    return 1


def wait_and_kill(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        try:
            p.terminate()
        except OSError:
            pass
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def exp_tag(exp: Experiment) -> str:
    tag = (
        f"{exp.task}"
        f"_lat1-{exp.latency_1}_nact1-{exp.n_action_steps_1}"
        f"_lat2-{exp.latency_2}_nact2-{exp.n_action_steps_2}"
    )
    if exp.staleness_1 > 0:
        tag += f"_stale1-{exp.staleness_1}"
    if exp.staleness_2 > 0:
        tag += f"_stale2-{exp.staleness_2}"
    return tag


def wait_for_server(port: int, timeout: int = 300) -> bool:
    """Poll the server's ping endpoint until it responds."""
    import msgpack
    import zmq

    deadline = time.time() + timeout
    while time.time() < deadline:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 5000)
        sock.setsockopt(zmq.SNDTIMEO, 5000)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://127.0.0.1:{port}")
        try:
            sock.send(msgpack.packb({"endpoint": "ping"}))
            resp = msgpack.unpackb(sock.recv())
            sock.close()
            ctx.term()
            if isinstance(resp, dict) and resp.get("status") == "ok":
                return True
        except (zmq.error.Again, zmq.error.ZMQError):
            try:
                sock.close()
                ctx.term()
            except Exception:
                pass
        time.sleep(5)
    return False


def main(args: Args) -> None:
    import datetime

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    num_gpus = args.num_gpus or detect_num_gpus()

    # Build experiment list: tasks x (lat1, nact1, stale1) x (lat2, nact2, stale2)
    all_experiments = []
    for task, lat1, nact1, stale1 in itertools.product(
        args.tasks,
        args.latencies_1, args.n_action_steps_1, args.staleness_1,
    ):
        if stale1 > lat1:
            continue

        # Determine phase 2 latencies for this lat1.
        if args.auto_latencies_2 == "pow2":
            # Powers of 2 up to lat1, plus 0 and lat1 itself.
            lat2_candidates = {0}
            if lat1 > 0:
                lat2_candidates.add(lat1)
                exp = 0
                while 2**exp <= lat1:
                    lat2_candidates.add(2**exp)
                    exp += 1
            lat2_list = sorted(lat2_candidates)
        elif args.auto_latencies_2 == "endpoints":
            # Only the extremes: 0 and lat1.
            lat2_list = sorted({0, lat1})
        else:
            lat2_list = args.latencies_2

        for lat2, nact2, stale2 in itertools.product(
            lat2_list, args.n_action_steps_2, args.staleness_2,
        ):
            if stale2 > lat2:
                continue
            all_experiments.append(
                Experiment(task, lat1, nact1, stale1, lat2, nact2, stale2)
            )

    model_name = os.path.basename(args.model_path.rstrip("/"))
    effective_seed = args.seed if args.seed is not None else 42
    default_log_dir = (
        f"data/dynamic_horizon_sweep/{args.env_prefix}/{model_name}/trials{args.num_trials}"
    )
    if args.seed is not None:
        default_log_dir += f"/seed{args.seed}"
    log_dir = args.log_dir or default_log_dir
    os.makedirs(log_dir, exist_ok=True)

    # Skip experiments whose results already exist.
    if args.overwrite:
        experiments = all_experiments
    else:
        experiments = []
        for exp in all_experiments:
            results_json = os.path.join(log_dir, f"results_{exp_tag(exp)}.json")
            if os.path.exists(results_json):
                log.info(f"  Skipping {exp_tag(exp)} (results exist)")
            else:
                experiments.append(exp)

    num_total = len(all_experiments)
    num_skipped = num_total - len(experiments)
    num_experiments = len(experiments)

    # Server runs via "uv run python" (needs main project deps).
    # Client may need a different venv (e.g., robocasa).
    server_cmd_prefix = ["uv", "run", "python"]

    _DEFAULT_CLIENT_PYTHON = {
        "robocasa_panda_omron": "gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python",
    }
    if args.client_python:
        client_cmd_prefix = args.client_python.split()
    elif args.env_prefix in _DEFAULT_CLIENT_PYTHON:
        client_cmd_prefix = [_DEFAULT_CLIENT_PYTHON[args.env_prefix]]
    else:
        client_cmd_prefix = ["uv", "run", "python"]

    log.info("=== Dynamic Horizon Sweep Configuration ===")
    log.info(f"  Tasks:            {args.tasks}")
    log.info(f"  Phase 1 latencies:  {args.latencies_1}")
    log.info(f"  Phase 1 nact:       {args.n_action_steps_1}")
    log.info(f"  Phase 1 staleness:  {args.staleness_1}")
    if args.auto_latencies_2 == "pow2":
        log.info("  Phase 2 latencies:  auto-pow2 [0, 1, 2, 4, ..., lat1]")
    elif args.auto_latencies_2 == "endpoints":
        log.info("  Phase 2 latencies:  auto-endpoints [0, lat1]")
    else:
        log.info(f"  Phase 2 latencies:  {args.latencies_2}")
    log.info(f"  Phase 2 nact:       {args.n_action_steps_2}")
    log.info(f"  Phase 2 staleness:  {args.staleness_2}")
    log.info(f"  Experiments:      {num_experiments} ({num_skipped} skipped, {num_total} total)")
    log.info(f"  GPUs:             {num_gpus}")
    log.info(f"  Scheduling:       work-queue (GPUs pick up next job when idle)")
    log.info(f"  Model:            {args.model_path}")
    log.info(f"  Embodiment:       {args.embodiment_tag}")
    log.info(f"  Trials/task:      {args.num_trials}")
    log.info(
        f"  Seed:             {effective_seed}"
        + (" (explicit)" if args.seed is not None else " (default)")
    )
    log.info(f"  Log dir:          {log_dir}")
    log.info(f"  Server python:    {' '.join(server_cmd_prefix)}")
    log.info(f"  Client python:    {' '.join(client_cmd_prefix)}")
    log.info("")

    if num_experiments == 0:
        log.info("Nothing to do.")
        return

    all_procs: list[subprocess.Popen] = []
    server_procs: list[subprocess.Popen] = []

    def cleanup(signum=None, frame=None):
        log.info("Shutting down all processes...")
        wait_and_kill(all_procs)
        log.info("Done.")
        sys.exit(1)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # --- Launch persistent servers (one per GPU) ---
    log.info("=== Launching policy servers ===")
    for gpu_id in range(num_gpus):
        port = args.base_port + gpu_id
        server_log = os.path.join(log_dir, f"server_gpu{gpu_id}_{run_ts}.log")
        log.info(f"  [server] GPU {gpu_id} | port {port} | log: {server_log}")

        with open(server_log, "w") as f:
            p = subprocess.Popen(
                [
                    *server_cmd_prefix, "gr00t/eval/run_gr00t_server.py",
                    "--model-path", args.model_path,
                    "--embodiment-tag", args.embodiment_tag,
                    "--use-sim-policy-wrapper",
                    "--port", str(port),
                    "--seed", str(effective_seed),
                ],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        server_procs.append(p)
        all_procs.append(p)

    # Wait for all servers to be ready.
    log.info("  Waiting for servers to load model...")
    for gpu_id in range(num_gpus):
        port = args.base_port + gpu_id
        if wait_for_server(port, timeout=args.server_wait):
            log.info(f"  [server] GPU {gpu_id} ready (port {port})")
        else:
            log.error(f"  [server] GPU {gpu_id} FAILED to start (port {port})")
            cleanup()
    log.info("")

    # --- Helper: launch a client experiment on a given GPU ---
    client_env = {
        **os.environ,
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0",
    }

    def launch_client(exp: Experiment, gpu_id: int) -> tuple[subprocess.Popen, str]:
        port = args.base_port + gpu_id
        tag = exp_tag(exp)
        env_name = f"{args.env_prefix}/{exp.task}"
        results_json = os.path.join(log_dir, f"results_{tag}.json")
        client_log = os.path.join(log_dir, f"client_{tag}_{run_ts}.log")
        video_dir = os.path.join(log_dir, f"videos_{tag}")

        log.info(
            f"  [client] GPU {gpu_id} | {exp.task}"
            f" | phase1: lat={exp.latency_1} nact={exp.n_action_steps_1}"
            f" stale={exp.staleness_1}"
            f" | phase2: lat={exp.latency_2} nact={exp.n_action_steps_2}"
            f" stale={exp.staleness_2}"
        )

        client_cmd = [
            *client_cmd_prefix, "gr00t/eval/rollout_policy_dynamic_horizon_latency.py",
            "--policy_client_host", "127.0.0.1",
            "--policy_client_port", str(port),
            "--env_name", env_name,
            "--n_episodes", str(args.num_trials),
            "--n_envs", str(args.n_envs),
            "--max_episode_steps", str(args.max_episode_steps),
            "--latency_1", str(exp.latency_1),
            "--staleness_1", str(exp.staleness_1),
            "--n_action_steps_1", str(exp.n_action_steps_1),
            "--latency_2", str(exp.latency_2),
            "--staleness_2", str(exp.staleness_2),
            "--n_action_steps_2", str(exp.n_action_steps_2),
            "--results_json_path", results_json,
            "--video_dir", video_dir,
            "--seed", str(effective_seed),
            *(["--extend_episode_budget"] if args.extend_episode_budget
              else ["--no-extend_episode_budget"]),
        ]

        with open(client_log, "w") as f:
            p = subprocess.Popen(
                client_cmd,
                env=client_env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        all_procs.append(p)
        return p, client_log

    # --- Run experiments with work-queue scheduling ---
    exp_queue = list(experiments)
    active_jobs: dict[int, tuple[subprocess.Popen, Experiment, str]] = {}
    completed = 0
    skipped_at_dispatch = 0

    def pop_next_experiment() -> Experiment | None:
        """Pop the next experiment from the queue, skipping any whose results
        already exist (e.g., written by a concurrent run on another machine)."""
        nonlocal skipped_at_dispatch
        while exp_queue:
            exp = exp_queue.pop(0)
            results_json = os.path.join(log_dir, f"results_{exp_tag(exp)}.json")
            if not args.overwrite and os.path.exists(results_json):
                skipped_at_dispatch += 1
                log.info(f"  Skipping {exp_tag(exp)} (results appeared concurrently)")
                continue
            return exp
        return None

    # Seed initial jobs (one per GPU).
    for gpu_id in range(num_gpus):
        exp = pop_next_experiment()
        if exp is None:
            break
        proc, client_log = launch_client(exp, gpu_id)
        active_jobs[gpu_id] = (proc, exp, client_log)

    log.info(f"  Monitor: tail -f {log_dir}/client_*.log")

    # Poll for finished jobs and dispatch next experiment to freed GPU.
    while active_jobs:
        for gpu_id in list(active_jobs):
            proc, exp, client_log = active_jobs[gpu_id]
            ret = proc.poll()
            if ret is not None:
                completed += 1
                status = "OK" if ret == 0 else f"FAILED (exit {ret})"
                log.info(
                    f"  [{completed}/{num_experiments - skipped_at_dispatch}] "
                    f"GPU {gpu_id} finished {exp_tag(exp)} — {status}"
                    f"  log: {client_log}"
                )
                del active_jobs[gpu_id]
                next_exp = pop_next_experiment()
                if next_exp is not None:
                    next_proc, next_log = launch_client(next_exp, gpu_id)
                    active_jobs[gpu_id] = (next_proc, next_exp, next_log)
        if active_jobs:
            time.sleep(2)

    # --- Kill servers ---
    log.info("Shutting down policy servers...")
    wait_and_kill(server_procs)

    # --- Summarize ---
    log.info(f"=== All {num_experiments} runs complete ({num_skipped} skipped). ===")
    subprocess.run([*server_cmd_prefix, "scripts/eval/summarize_latency_sweep.py", log_dir])


if __name__ == "__main__":
    main(tyro.cli(Args))
