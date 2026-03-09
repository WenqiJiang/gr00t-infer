# GR00T Simulation Evaluation Guide

## Benchmark Overview

### Sim vs Real

| # | Benchmark | Type | Simulator | Robot | GPU Constraint |
|---|-----------|------|-----------|-------|----------------|
| 1 | LIBERO | **Simulation** | MuJoCo | Panda arm | Any GPU |
| 2 | SimplerEnv | **Simulation** | ManiSkill2 | Google Robot / WidowX | Any GPU (GPU-accelerated) |
| 3 | RoboCasa | **Simulation** | MuJoCo | Panda + Omron gripper | Any GPU |
| 4 | RoboCasa GR1 Tabletop | **Simulation** | MuJoCo | GR-1 humanoid | Any GPU |
| 5 | BEHAVIOR | **Simulation** | Isaac Sim / OmniGibson | R1 Pro humanoid | **RT cores required (L40/L40s only, NOT A100/H100)** |
| 6 | G1 LocoManipulation | **Simulation** | MuJoCo | Unitree G1 humanoid | Any GPU |
| 7 | PointNav | **Simulation** | External (COMPASS framework) | G1 navigation | Any GPU |
| 8 | SO100 | **Real robot** | N/A (physical SO101 arm) | LeRobot SO101 | Any GPU (server only) |

### Reference Accuracy (from NVIDIA's reported results)

Sorted from easiest (highest accuracy) to hardest (lowest accuracy):

| Benchmark | Metric | Reference Accuracy | Checkpoint | Difficulty |
|-----------|--------|-------------------|------------|------------|
| LIBERO (Spatial) | Success rate | **97.65%** | Finetuned | Easy |
| LIBERO (Object) | Success rate | **98.45%** | Finetuned | Easy |
| LIBERO (Goal) | Success rate | **97.50%** | Finetuned | Easy |
| LIBERO (10 Long) | Success rate | **94.35%** | Finetuned | Easy |
| PointNav (In-Dist) | Success rate | **86.3%** | Finetuned | Medium |
| PointNav (Out-of-Dist) | Success rate | **76.5%** | Finetuned | Medium |
| SimplerEnv (Fractal/Google) | Success rate | **67.66%** avg | Finetuned | Medium |
| RoboCasa | Success rate | **66.22%** avg (24 tasks) | Zero-shot (base model) | Medium |
| SimplerEnv (Bridge/WidowX) | Success rate | **62.07%** avg | Finetuned | Medium |
| G1 LocoManipulation | Success rate | **58%** (±15% variance) | Finetuned | Hard |
| RoboCasa GR1 Tabletop | Success rate | **47.6%** avg (24 tasks) | Zero-shot (base model) | Hard |
| BEHAVIOR | Task progress | **26.30%** avg (50 tasks) | Finetuned | Very Hard |

**Notes:**
- LIBERO benchmarks are the easiest to get working and achieve high scores — good starting point.
- RoboCasa and RoboCasa GR1 Tabletop use the **base model zero-shot** (no finetuning), so accuracy is lower.
- BEHAVIOR is the hardest: 50 household tasks, complex multi-step goals, and metric is "task progress" (% of sub-goals), not binary success.
- G1 LocoManipulation has high variance (±15%) across runs.
- Training variance across runs can be 5-6% even with same config/seed.

### Recommended Starting Order

1. **LIBERO** — easiest setup, highest accuracy, good for validating your pipeline works
2. **SimplerEnv** — GPU-accelerated (10-15x faster), moderate difficulty
3. **RoboCasa** — zero-shot eval (no finetuning needed), straightforward MuJoCo setup
4. **G1 LocoManipulation** — requires git-lfs, moderate setup complexity
5. **RoboCasa GR1 Tabletop** — similar to RoboCasa but humanoid embodiment
6. **BEHAVIOR** — hardest setup (Isaac Sim, RT-core GPUs only), lowest accuracy

---

## General Pattern: Server-Client Architecture

**Terminal 1 (GPU server):** Runs the GR00T policy model via ZeroMQ.
**Terminal 2 (Sim client):** Runs the simulation in an isolated venv, connects to the server for actions.

---

## Per-Benchmark Guide

### 1. LIBERO (Panda robot, MuJoCo)

**Setup (one-time):**
```bash
sudo apt install libegl1-mesa-dev libglu1-mesa
bash gr00t/eval/sim/LIBERO/setup_libero.sh
```

**Server (Terminal 1):**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path <CHECKPOINT_PATH> \
    --embodiment-tag LIBERO_PANDA \
    --use-sim-policy-wrapper
```

**Client (Terminal 2):**
```bash
gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 --policy_client_port 5555 \
    --max_episode_steps 720 \
    --env_name libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
    --n_action_steps 8 --n_envs 5
```

4 suites available: `libero_spatial`, `libero_goal`, `libero_object`, `libero_10` (10 tasks each).

---

### 2. SimplerEnv (Google Robot / WidowX, ManiSkill2)

**Setup (one-time):**
```bash
sudo apt install libegl1-mesa-dev libglu1-mesa
bash gr00t/eval/sim/SimplerEnv/setup_SimplerEnv.sh
```

**Server (Terminal 1):**
```bash
# Google Robot (Fractal)
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-fractal \
    --embodiment-tag OXE_GOOGLE --use-sim-policy-wrapper

# WidowX (Bridge)
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-bridge \
    --embodiment-tag OXE_WIDOWX --use-sim-policy-wrapper
```

**Client (Terminal 2):**
```bash
gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 --policy_client_port 5555 \
    --max_episode_steps 300 \
    --env_name simpler_env_google/google_robot_pick_coke_can \
    --n_action_steps 1 --n_envs 5
```

Note: use `--n_action_steps 1` for Google robot, `8` for WidowX.

---

### 3. RoboCasa (Panda + Omron gripper, MuJoCo)

**Setup (one-time):**
```bash
sudo apt install libegl1-mesa-dev libglu1-mesa
bash gr00t/eval/sim/robocasa/setup_RoboCasa.sh
```

**Server (Terminal 1):**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON --use-sim-policy-wrapper
```

**Client (Terminal 2):**
```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 --policy_client_port 5555 \
    --max_episode_steps 720 \
    --env_name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
    --n_action_steps 8 --n_envs 5
```

---

### 4. RoboCasa GR1 Tabletop (GR-1 humanoid, MuJoCo)

**Setup (one-time):**
```bash
sudo apt install libegl1-mesa-dev libglu1-mesa
bash gr00t/eval/sim/robocasa-gr1-tabletop-tasks/setup_RoboCasaGR1TabletopTasks.sh
```

**Server (Terminal 1):**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag GR1 --use-sim-policy-wrapper
```

**Client (Terminal 2):**
```bash
gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 --policy_client_port 5555 \
    --max_episode_steps 720 \
    --env_name gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env \
    --n_action_steps 8 --n_envs 5
```

---

### 5. BEHAVIOR (R1 Pro humanoid, Isaac Sim / OmniGibson)

**Requires RT cores** (L40/L40s only; A100/H100 NOT supported).

**Setup (one-time):**
```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git && cd BEHAVIOR-1K
git checkout feat/task-progress
source PATH_TO_GR00T/.venv/bin/activate
bash ./setup_uv.sh
# Download test instances
python gr00t/eval/sim/BEHAVIOR/prepare_test_instances.py
```

**Server (Terminal 1):**
```bash
uv run gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-BEHAVIOR1k \
    --embodiment-tag BEHAVIOR_R1_PRO --use-sim-policy-wrapper
```

**Client (Terminal 2):**
```bash
uv run python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 --policy_client_port 5555 \
    --max_episode_steps 999999999 \
    --env_name sim_behavior_r1_pro/turning_on_radio \
    --n_action_steps 8 --n_envs 1
```

**Critical**: must use `--n_envs 1` (OmniGibson can't parallelize) and very large `--max_episode_steps` (relies on built-in termination).

---

### 6. G1 LocoManipulation (Unitree G1, MuJoCo)

**Setup (one-time):**
```bash
sudo apt install libegl1-mesa-dev libglu1-mesa
# Requires git-lfs
bash gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
```

**Server (Terminal 1):**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    --embodiment-tag UNITREE_G1 --use-sim-policy-wrapper
```

**Client (Terminal 2):**
```bash
gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --max_episode_steps 1440 \
    --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc \
    --n_action_steps 20 --n_envs 5
```

Note: `--n_action_steps 20` (higher due to WBC control frequency).

---

## Pre-flight Check

Before running any benchmark, verify your setup:
```bash
python scripts/eval/check_sim_eval_ready.py
```

This checks: `uv` version, Vulkan/EGL drivers, and smoke-tests each sim environment.

---

## Key Things to Know

| Detail | Value |
|---|---|
| Each benchmark gets its own isolated venv | `gr00t/eval/sim/<BENCHMARK>/<name>_uv/.venv` |
| Headless rendering | All use `MUJOCO_GL=egl` / `PYOPENGL_PLATFORM=egl` |
| Server flag | Always pass `--use-sim-policy-wrapper` for sim eval |
| Server default port | 5555 (ZeroMQ) |
| Episodes end on success | `terminate_on_success=True` in `MultiStepWrapper` |
