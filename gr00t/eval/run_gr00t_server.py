# NOTE on reproducibility: even with fixed seeds (--seed), results may vary across
# runs due to GPU non-determinism (CUDA, cuDNN), async vectorized env scheduling,
# and floating-point non-associativity in MuJoCo physics. The best way to report
# reliable accuracy is to run a large number of episodes (50+) and report mean ± stderr.

from dataclasses import dataclass
import json
import os
import random

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
import numpy as np
import torch
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


@dataclass
class ServerConfig:
    """Configuration for running the Groot N1.5 inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag"""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""

    seed: int = 42
    """Random seed for reproducibility (seeds torch, numpy, random)"""


def main(config: ServerConfig):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    print("Starting GR00T inference server...")
    print(f"  Seed: {config.seed}")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # check if the model path exists
    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Create and start the server
    if config.model_path is not None:
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
    elif config.dataset_path is not None:
        if config.modality_config_path is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS[config.embodiment_tag.value]
        else:
            with open(config.modality_config_path, "r") as f:
                modality_configs = json.load(f)
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Print model info after loading.
    if isinstance(policy, Gr00tPolicy):
        mc = policy.modality_configs
        action_horizon = len(mc["action"].delta_indices)
        state_keys = mc["state"].modality_keys
        action_keys = mc["action"].modality_keys
        video_keys = mc["video"].modality_keys
        model_action_horizon = policy.model.config.action_horizon
        print(f"\n=== Model Info ===")
        print(f"  Model action_horizon (max): {model_action_horizon}")
        print(f"  Embodiment action_horizon:  {action_horizon}")
        print(f"  Action delta_indices:       {mc['action'].delta_indices}")
        print(f"  Video keys:   {video_keys}")
        print(f"  State keys:   {state_keys}")
        print(f"  Action keys:  {action_keys}")
        print(f"  Max n_action_steps you can use: {action_horizon}")
        print(f"==================\n")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
