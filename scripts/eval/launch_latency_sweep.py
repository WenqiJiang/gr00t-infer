"""Launch a latency x n_action_steps (x staleness) sweep across RoboCasa tasks.

Launches persistent policy servers (one per GPU) and schedules client experiments
in rounds. Servers are reused across experiments to avoid repeated model loading.

Usage:
    uv run python scripts/eval/launch_latency_sweep.py \
        --tasks CoffeeSetupMug_PandaOmron_Env OpenDrawer_PandaOmron_Env
    uv run python scripts/eval/launch_latency_sweep.py \
        --tasks CoffeeSetupMug_PandaOmron_Env \
        --latencies 0 5 10 20 --n-action-steps 1 4 8 16
    uv run python scripts/eval/launch_latency_sweep.py \
        --tasks CoffeeSetupMug_PandaOmron_Env \
        --staleness 0 5 10 --num-trials 50

After completion, view results:
    uv run python scripts/eval/summarize_latency_sweep.py data/robocasa/latency_sweep/

NOTE on reproducibility: even with fixed seeds (--seed), results may vary across
runs due to GPU non-determinism (CUDA, cuDNN), async vectorized env scheduling,
and floating-point non-associativity in MuJoCo physics. The best way to report
reliable accuracy is to run a large number of episodes (50+) and report mean ± stderr.
"""

import dataclasses
import itertools
import logging
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

    latencies: list[int] = dataclasses.field(default_factory=lambda: [0, 5, 10, 20])
    """Simulated inference latency in env steps."""

    n_action_steps: list[int] = dataclasses.field(default_factory=lambda: [1, 4, 8, 16])
    """Number of model actions to execute before replanning."""

    staleness: list[int] = dataclasses.field(default_factory=list)
    """Observation staleness values for async inference. Empty = sync only (staleness=0).
    When set, only valid combos where staleness <= latency are generated."""

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

    seed: int = 42
    """Random seed for reproducibility."""

    extend_episode_budget: bool = True
    """Extend max episode steps to compensate for hold actions during latency."""

    client_python: str = ""
    """Python interpreter for client processes (e.g., a sim-specific venv).
    If empty, auto-detected from env_prefix (robocasa uses its own venv)."""


@dataclasses.dataclass
class Experiment:
    task: str
    latency: int
    n_action_steps: int
    staleness: int | None = None  # None = sync (staleness=0)


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
    tag = f"{exp.task}_lat{exp.latency}_nact{exp.n_action_steps}"
    if exp.staleness is not None and exp.staleness > 0:
        tag += f"_stale{exp.staleness}"
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
    # Timestamp for log filenames so concurrent runs on different servers don't collide.
    import datetime

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    num_gpus = args.num_gpus or detect_num_gpus()

    # Build experiment list.
    if args.staleness:
        all_experiments = [
            Experiment(task, lat, nact, stale)
            for task, lat, nact, stale in itertools.product(
                args.tasks, args.latencies, args.n_action_steps, args.staleness
            )
            if stale <= lat
        ]
    else:
        all_experiments = [
            Experiment(task, lat, nact)
            for task, lat, nact in itertools.product(
                args.tasks, args.latencies, args.n_action_steps
            )
        ]

    log_dir = args.log_dir or f"data/robocasa/latency_sweep/trials{args.num_trials}"
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

    # Per-benchmark default client Python interpreters.
    _DEFAULT_CLIENT_PYTHON = {
        "robocasa_panda_omron": "gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python",
    }
    if args.client_python:
        client_cmd_prefix = args.client_python.split()
    elif args.env_prefix in _DEFAULT_CLIENT_PYTHON:
        client_cmd_prefix = [_DEFAULT_CLIENT_PYTHON[args.env_prefix]]
    else:
        client_cmd_prefix = ["uv", "run", "python"]

    log.info("=== Latency Sweep Configuration ===")
    log.info(f"  Tasks:            {args.tasks}")
    log.info(f"  Latencies:        {args.latencies}")
    log.info(f"  N action steps:   {args.n_action_steps}")
    log.info(f"  Staleness:        {args.staleness or 'sync (staleness=0)'}")
    log.info(f"  Experiments:      {num_experiments} ({num_skipped} skipped, {num_total} total)")
    log.info(f"  GPUs:             {num_gpus}")
    log.info(f"  Scheduling:       work-queue (GPUs pick up next job when idle)")
    log.info(f"  Model:            {args.model_path}")
    log.info(f"  Embodiment:       {args.embodiment_tag}")
    log.info(f"  Trials/task:      {args.num_trials}")
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
                    "--seed", str(args.seed),
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
        # Clients only need EGL for MuJoCo rendering, not CUDA.
        # Always use EGL device 0 to avoid errors on single-GPU machines.
        "MUJOCO_EGL_DEVICE_ID": "0",
    }

    def launch_client(exp: Experiment, gpu_id: int) -> subprocess.Popen:
        port = args.base_port + gpu_id
        tag = exp_tag(exp)
        env_name = f"{args.env_prefix}/{exp.task}"
        results_json = os.path.join(log_dir, f"results_{tag}.json")
        client_log = os.path.join(log_dir, f"client_{tag}_{run_ts}.log")
        video_dir = os.path.join(log_dir, f"videos_{tag}")

        stale_str = str(exp.staleness) if exp.staleness is not None else "sync"
        log.info(
            f"  [client] GPU {gpu_id} | {exp.task} | latency {exp.latency}"
            f" | nact {exp.n_action_steps} | staleness {stale_str}"
        )

        client_cmd = [
            *client_cmd_prefix, "gr00t/eval/rollout_policy_with_latency.py",
            "--policy_client_host", "127.0.0.1",
            "--policy_client_port", str(port),
            "--env_name", env_name,
            "--n_episodes", str(args.num_trials),
            "--n_envs", str(args.n_envs),
            "--max_episode_steps", str(args.max_episode_steps),
            "--latency", str(exp.latency),
            "--staleness", str(exp.staleness if exp.staleness is not None else 0),
            "--n_action_steps", str(exp.n_action_steps),
            "--results_json_path", results_json,
            "--video_dir", video_dir,
            "--seed", str(args.seed),
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
        return p

    # --- Run experiments with work-queue scheduling ---
    # Each GPU picks up the next experiment as soon as it finishes the current one,
    # so no GPU sits idle waiting for others to complete a round.
    exp_queue = list(experiments)  # work queue
    # active_jobs: gpu_id -> (Popen, Experiment)
    active_jobs: dict[int, tuple[subprocess.Popen, Experiment]] = {}
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
        active_jobs[gpu_id] = (launch_client(exp, gpu_id), exp)

    log.info(f"  Monitor: tail -f {log_dir}/client_*.log")

    # Poll for finished jobs and dispatch next experiment to freed GPU.
    while active_jobs:
        for gpu_id in list(active_jobs):
            proc, exp = active_jobs[gpu_id]
            ret = proc.poll()
            if ret is not None:
                completed += 1
                status = "OK" if ret == 0 else f"FAILED (exit {ret})"
                log.info(
                    f"  [{completed}/{num_experiments - skipped_at_dispatch}] "
                    f"GPU {gpu_id} finished {exp_tag(exp)} — {status}"
                )
                del active_jobs[gpu_id]
                # Dispatch next experiment to this GPU.
                next_exp = pop_next_experiment()
                if next_exp is not None:
                    active_jobs[gpu_id] = (launch_client(next_exp, gpu_id), next_exp)
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
