from rl_games.torch_runner import _restore
from rl_games.torch_runner import players
from typing import Optional
from gym import spaces
import numpy as np
from rl_player_utils import read_cfg
import torch
import torch.nn as nn


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


# HACK: Use global to workaround fact that we can't pass in the observation and action space
_GLOBAL_NUM_OBSERVATIONS = -1
_GLOBAL_NUM_ACTIONS = -1


class DummyEnv:
    """
    PpoPlayerContinuous creates an env for the model to interact with in the run() method
    However, we don't want to create an env, as we only want to use the model to get an action.
    So we override the create_env() method to return a dummy env with the correct observation and action space.
    """

    def __init__(self, observation_space: spaces.Box, action_space: spaces.Box) -> None:
        self.observation_space = observation_space
        self.action_space = action_space


class InferenceOnlyPlayer(players.PpoPlayerContinuous):
    def create_env(self) -> DummyEnv:
        if _GLOBAL_NUM_OBSERVATIONS <= 0 or _GLOBAL_NUM_ACTIONS <= 0:
            raise ValueError("Global observation and action space not set")

        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(_GLOBAL_NUM_OBSERVATIONS,),
            dtype=np.float32,
        )
        action_space = spaces.Box(
            low=-1, high=1, shape=(_GLOBAL_NUM_ACTIONS,), dtype=np.float32
        )
        return DummyEnv(observation_space=observation_space, action_space=action_space)


class RlPlayer(nn.Module):
    def __init__(
        self,
        num_observations: int,
        num_actions: int,
        config_path: str,
        checkpoint_path: Optional[str],
        device: str,
        deterministic_actions: bool = True,
    ) -> None:
        super().__init__()
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.deterministic_actions = deterministic_actions

        # Modify globals
        global _GLOBAL_NUM_OBSERVATIONS, _GLOBAL_NUM_ACTIONS
        _GLOBAL_NUM_OBSERVATIONS = num_observations
        _GLOBAL_NUM_ACTIONS = num_actions

        # Create player
        self.cfg = read_cfg(config_path=config_path, device=device)
        self.player = InferenceOnlyPlayer(
            params=self.cfg["train"]["params"],
        )
        if checkpoint_path is not None:
            _restore(agent=self.player, args={"checkpoint": checkpoint_path})
        self.player.init_rnn()

        # Sanity check
        self._run_sanity_checks()

        self.to(device)

    def _run_sanity_checks(self):
        cfg_num_observations = self.cfg["task"]["env"]["numObservations"]
        cfg_num_actions = self.cfg["task"]["env"]["numActions"]

        if cfg_num_observations != self.num_observations and cfg_num_observations > 0:
            print(
                f"WARNING: num_observations in config ({cfg_num_observations}) does not match num_observations passed to RlPlayer ({self.num_observations})"
            )
        if cfg_num_actions != self.num_actions and cfg_num_actions > 0:
            print(
                f"WARNING: num_actions in config ({cfg_num_actions}) does not match num_actions passed to RlPlayer ({self.num_actions})"
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.player.get_action(
            obs=obs,
            is_deterministic=self.deterministic_actions,
        )

    def get_normalized_action(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        assert_equals(obs.shape, (batch_size, self.num_observations))

        normalized_action = self.forward(obs=obs).reshape(batch_size, self.num_actions)
        return normalized_action


def main() -> None:
    import pathlib

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path_to_this_dir = pathlib.Path(__file__).parent.absolute()

    CONFIG_PATH = path_to_this_dir / "config.yaml"
    CHECKPOINT_PATH = path_to_this_dir / "checkpoint.pt"
    NUM_OBSERVATIONS = 100
    NUM_ACTIONS = 100

    player = RlPlayer(
        num_observations=NUM_OBSERVATIONS,
        num_actions=NUM_ACTIONS,
        config_path=str(CONFIG_PATH),
        checkpoint_path=str(CHECKPOINT_PATH),
        device=device,
    ).to(device)

    batch_size = 2
    obs = torch.rand(batch_size, NUM_OBSERVATIONS).to(device)
    normalized_action = player.get_normalized_action(obs=obs)
    print(f"Using player with config: {CONFIG_PATH} and checkpoint: {CHECKPOINT_PATH}")
    print(f"And num_observations: {NUM_OBSERVATIONS} and num_actions: {NUM_ACTIONS}")
    print(f"Sampled obs: {obs} with shape: {obs.shape}")
    print(
        f"Got normalized_action: {normalized_action} with shape: {normalized_action.shape}"
    )
    print(f"player: {player.player.model}")


if __name__ == "__main__":
    main()
