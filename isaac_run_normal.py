#!/usr/bin/env python
from isaacgymenvs.utils.cross_embodiment.create_env import create_env  # isort:skip
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch

# Import from the fabrics_sim package
from isaacgym.torch_utils import to_torch
from isaacgymenvs.utils.cross_embodiment.constants import (
    NUM_XYZ,
)
from isaacgymenvs.utils.cross_embodiment.kuka_allegro_constants import (
    ALLEGRO_FINGERTIP_LINK_NAMES,
    NUM_FINGERS,
    PALM_LINK_NAME,
    PALM_X_LINK_NAME,
    PALM_Y_LINK_NAME,
    PALM_Z_LINK_NAME,
)
from isaacgymenvs.utils.cross_embodiment.kuka_allegro_constants import (
    NUM_HAND_ARM_DOFS as KUKA_ALLEGRO_NUM_DOFS,
)
from isaacgymenvs.utils.cross_embodiment.object_constants import (
    NUM_OBJECT_KEYPOINTS,
    OBJECT_KEYPOINT_OFFSETS,
)
from isaacgymenvs.utils.cross_embodiment.utils import (
    assert_equals,
    rescale,
)
from isaacgymenvs.utils.wandb_utils import restore_file_from_wandb

from rl_player import RlPlayer

FABRIC_MODE: Literal["PCA", "ALL"] = "PCA"

NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


def taskmap_helper(
    q: torch.Tensor, qd: torch.Tensor, taskmap, taskmap_link_names: List[str]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = q.shape[0]
    assert_equals(q.shape, (N, KUKA_ALLEGRO_NUM_DOFS))
    assert_equals(qd.shape, (N, KUKA_ALLEGRO_NUM_DOFS))

    x, jac = taskmap(q, None)
    n_points = len(taskmap_link_names)
    assert_equals(x.shape, (N, NUM_XYZ * n_points))
    assert_equals(jac.shape, (N, NUM_XYZ * n_points, KUKA_ALLEGRO_NUM_DOFS))

    # Calculate the velocity in the task space
    xd = torch.bmm(jac, qd.unsqueeze(2)).squeeze(2)
    assert_equals(xd.shape, (N, NUM_XYZ * n_points))

    return (
        x.reshape(N, n_points, NUM_XYZ),
        xd.reshape(N, n_points, NUM_XYZ),
        jac.reshape(N, n_points, NUM_XYZ, KUKA_ALLEGRO_NUM_DOFS),
    )


def create_observation(
    iiwa_position: np.ndarray,
    iiwa_velocity: np.ndarray,
    allegro_position: np.ndarray,
    allegro_velocity: np.ndarray,
    fabric_q: np.ndarray,
    fabric_qd: np.ndarray,
    object_position_R: np.ndarray,
    object_quat_xyzw_R: np.ndarray,
    goal_object_pos_R: np.ndarray,
    goal_object_quat_xyzw_R: np.ndarray,
    object_pos_R_prev: np.ndarray,
    object_quat_xyzw_R_prev: np.ndarray,
    object_pos_R_prev_prev: np.ndarray,
    object_quat_xyzw_R_prev_prev: np.ndarray,
    device: torch.device,
    taskmap,
    taskmap_link_names: List[str],
    num_observations: int,
) -> Optional[torch.Tensor]:
    keypoint_offsets = to_torch(
        OBJECT_KEYPOINT_OFFSETS,
        device=device,
        dtype=torch.float,
    )
    assert_equals(keypoint_offsets.shape, (NUM_OBJECT_KEYPOINTS, NUM_XYZ))

    q = np.concatenate([iiwa_position, allegro_position])
    qd = np.concatenate([iiwa_velocity, allegro_velocity])

    taskmap_positions, _, _ = taskmap_helper(
        q=torch.from_numpy(q).float().unsqueeze(0).to(device),
        qd=torch.from_numpy(qd).float().unsqueeze(0).to(device),
        taskmap=taskmap,
        taskmap_link_names=taskmap_link_names,
    )
    taskmap_positions = taskmap_positions.squeeze(0).cpu().numpy()
    palm_pos = taskmap_positions[taskmap_link_names.index(PALM_LINK_NAME)]
    palm_x_pos = taskmap_positions[taskmap_link_names.index(PALM_X_LINK_NAME)]
    palm_y_pos = taskmap_positions[taskmap_link_names.index(PALM_Y_LINK_NAME)]
    palm_z_pos = taskmap_positions[taskmap_link_names.index(PALM_Z_LINK_NAME)]
    fingertip_positions = np.stack(
        [
            taskmap_positions[taskmap_link_names.index(link_name)]
            for link_name in ALLEGRO_FINGERTIP_LINK_NAMES
        ],
        axis=0,
    )

    obs_dict = {}
    obs_dict["q"] = np.concatenate([iiwa_position, allegro_position])
    obs_dict["qd"] = np.concatenate([iiwa_velocity, allegro_velocity])
    obs_dict["fingertip_positions"] = fingertip_positions.reshape(NUM_FINGERS * NUM_XYZ)
    obs_dict["palm_pos"] = palm_pos
    obs_dict["palm_x_pos"] = palm_x_pos
    obs_dict["palm_y_pos"] = palm_y_pos
    obs_dict["palm_z_pos"] = palm_z_pos
    obs_dict["object_pos"] = object_position_R
    obs_dict["object_quat_xyzw"] = object_quat_xyzw_R
    obs_dict["goal_pos"] = goal_object_pos_R
    obs_dict["goal_quat_xyzw"] = goal_object_quat_xyzw_R

    obs_dict["prev_object_pos"] = object_pos_R_prev
    obs_dict["prev_object_quat_xyzw"] = object_quat_xyzw_R_prev
    obs_dict["prev_prev_object_pos"] = object_pos_R_prev_prev
    obs_dict["prev_prev_object_quat_xyzw"] = object_quat_xyzw_R_prev_prev

    # object_keypoint_positions = (
    #     compute_keypoint_positions(
    #         pos=torch.tensor(object_position_R, device=self.device)
    #         .unsqueeze(0)
    #         .float(),
    #         quat_xyzw=torch.tensor(object_quat_xyzw_R, device=self.device)
    #         .unsqueeze(0)
    #         .float(),
    #         keypoint_offsets=keypoint_offsets.unsqueeze(0).float(),
    #     )
    #     .squeeze(0)
    #     .cpu()
    #     .numpy()
    # )
    # goal_object_keypoint_positions = (
    #     compute_keypoint_positions(
    #         pos=torch.tensor(goal_object_pos_R, device=self.device)
    #         .unsqueeze(0)
    #         .float(),
    #         quat_xyzw=torch.tensor(goal_object_quat_xyzw_R, device=self.device)
    #         .unsqueeze(0)
    #         .float(),
    #         keypoint_offsets=keypoint_offsets.unsqueeze(0).float(),
    #     )
    #     .squeeze(0)
    #     .cpu()
    #     .numpy()
    # )
    # object_vel = np.zeros(3)
    # object_angvel = np.zeros(3)
    # obs_dict["object_keypoint_positions"] = (
    #     object_keypoint_positions.reshape(
    #         NUM_OBJECT_KEYPOINTS * NUM_XYZ
    #     )
    # )
    # obs_dict["goal_object_keypoint_positions"] = (
    #     goal_object_keypoint_positions.reshape(
    #         NUM_OBJECT_KEYPOINTS * NUM_XYZ
    #     )
    # )
    # obs_dict["object_vel"] = object_vel
    # obs_dict["object_angvel"] = object_angvel
    # obs_dict["t"] = np.array([self.t]).reshape(1)
    # self.t += 1

    obs_dict["fabric_q"] = fabric_q
    obs_dict["fabric_qd"] = fabric_qd

    for k, v in obs_dict.items():
        assert len(v.shape) == 1, f"Shape of {k} is {v.shape}, expected 1D tensor"

    # DEBUG
    # for k, v in obs_dict.items():
    #     print(f"{k}: {v}")
    # print()

    # Concatenate all observations into a 1D tensor
    observation = np.concatenate(
        [obs for obs in obs_dict.values()],
        axis=-1,
    )
    assert_equals(observation.shape, (num_observations,))

    return torch.from_numpy(observation).float().unsqueeze(0).to(device)


def rescale_action(
    action: torch.Tensor,
    palm_mins: torch.Tensor,
    palm_maxs: torch.Tensor,
    hand_mins: torch.Tensor,
    hand_maxs: torch.Tensor,
    num_actions: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = action.shape[0]
    assert_equals(action.shape, (N, num_actions))

    # Rescale the normalized actions from [-1, 1] to the actual target ranges
    palm_target = rescale(
        values=action[:, :6],
        old_mins=torch.ones_like(palm_mins) * -1,
        old_maxs=torch.ones_like(palm_maxs) * 1,
        new_mins=palm_mins,
        new_maxs=palm_maxs,
    )
    hand_target = rescale(
        values=action[:, 6:],
        old_mins=torch.ones_like(hand_mins) * -1,
        old_maxs=torch.ones_like(hand_maxs) * 1,
        new_mins=hand_mins,
        new_maxs=hand_maxs,
    )
    return palm_target, hand_target


def main():
    """
    Env
    """
    _, CONFIG_PATH = restore_file_from_wandb(
        # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-05_cup_fabric_reset-early_multigpu/files/runs/TOP_4-freq_coll-on_juno1_2_2024-10-07_23-27-58-967674/config_resolved.yaml?runName=TOP_4-freq_coll-on_juno1_2_2024-10-07_23-27-58-967674"
        "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-16_experiments/files/runs/plate_hard_65damp_move1_2025-01-16_00-53-04-618360/config_resolved.yaml?runName=plate_hard_65damp_move1_2025-01-16_00-53-04-618360_2amo0e8y"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = create_env(
        config_path=CONFIG_PATH,
        device=device,
        # headless=True,
        headless=False,
        # enable_viewer_sync_at_start=False,
        enable_viewer_sync_at_start=True,
    )

    # Set control rate
    dt = env.control_dt

    """
    RL Player
    """
    # RL Player setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_observations = 144  # Update this number based on actual dimensions
    if FABRIC_MODE == "PCA":
        num_actions = 11  # First 6 for palm, last 5 for hand
    elif FABRIC_MODE == "ALL":
        num_actions = 22  # First 6 for palm, last 16 for hand
    else:
        raise ValueError(f"Invalid FABRIC_MODE: {FABRIC_MODE}")

    _, config_path = restore_file_from_wandb(
        # 2025-01-15
        # "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-14_updated_pregrasps/files/runs/snackbox_pivot_hard_onestep_juno2_2025-01-14_23-42-05-001698/config_resolved.yaml?runName=snackbox_pivot_hard_onestep_juno2_2025-01-14_23-42-05-001698_394k5u5l"
        # "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-14_updated_pregrasps/files/runs/ladel_hard_scoop_move1_2025-01-14_23-37-10-016140/config_resolved.yaml?runName=ladel_hard_scoop_move1_2025-01-14_23-37-10-016140_lq616xtz"
        # "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-14_updated_pregrasps/files/runs/plate_hard_juno2_2025-01-14_23-34-23-000027/config_resolved.yaml?runName=plate_hard_juno2_2025-01-14_23-34-23-000027_435jouzp"
        # "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-15_updated_pregrasps_2/files/runs/plate_hard_FORCES_juno2-lo_2025-01-15_11-48-36-651851/config_resolved.yaml?runName=plate_hard_FORCES_juno2-lo_2025-01-15_11-48-36-651851_7xp5thny"
        # 2025-01-16
        # "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-15_updated_pregrasps_2/files/runs/ladel_hard_scoop_FORCES_juno2-lo_2025-01-15_11-48-35-403864/config_resolved.yaml?runName=ladel_hard_scoop_FORCES_juno2-lo_2025-01-15_11-48-35-403864_vi1sft9j"
        "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-16_experiments/files/runs/plate_hard_65damp_move1_2025-01-16_00-53-04-618360/config_resolved.yaml?runName=plate_hard_65damp_move1_2025-01-16_00-53-04-618360_2amo0e8y"
        # "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-15_updated_pregrasps_2/files/runs/plate_hard_FORCES_juno2-lo_2025-01-15_11-48-36-651851/config_resolved.yaml?runName=plate_hard_FORCES_juno2-lo_2025-01-15_11-48-36-651851_7xp5thny"
        # "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-15_updated_pregrasps_2/files/runs/watering_can_move2_2025-01-15_11-45-55-011129/config_resolved.yaml?runName=watering_can_move2_2025-01-15_11-45-55-011129_kb5rd4jf"
    )
    _, checkpoint_path = restore_file_from_wandb(
        # 2025-01-15
        # "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-14_updated_pregrasps/files/runs/snackbox_pivot_hard_onestep_juno2_2025-01-14_23-42-05-001698/nn/best.pth?runName=snackbox_pivot_hard_onestep_juno2_2025-01-14_23-42-05-001698_394k5u5l"
        # "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-14_updated_pregrasps/files/runs/ladel_hard_scoop_move1_2025-01-14_23-37-10-016140/nn/best.pth?runName=ladel_hard_scoop_move1_2025-01-14_23-37-10-016140_lq616xtz"
        # "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-14_updated_pregrasps/files/runs/plate_hard_juno2_2025-01-14_23-34-23-000027/nn/best.pth?runName=plate_hard_juno2_2025-01-14_23-34-23-000027_435jouzp"
        # "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-15_updated_pregrasps_2/files/runs/plate_hard_FORCES_juno2-lo_2025-01-15_11-48-36-651851/nn/best.pth?runName=plate_hard_FORCES_juno2-lo_2025-01-15_11-48-36-651851_7xp5thny"
        # 2025-01-16
        # "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-15_updated_pregrasps_2/files/runs/ladel_hard_scoop_FORCES_juno2-lo_2025-01-15_11-48-35-403864/nn/best.pth?runName=ladel_hard_scoop_FORCES_juno2-lo_2025-01-15_11-48-35-403864_vi1sft9j"
        "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-16_experiments/files/runs/plate_hard_65damp_move1_2025-01-16_00-53-04-618360/nn/best.pth?runName=plate_hard_65damp_move1_2025-01-16_00-53-04-618360_2amo0e8y"
        # "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-15_updated_pregrasps_2/files/runs/plate_hard_FORCES_juno2-lo_2025-01-15_11-48-36-651851/nn/best.pth?runName=plate_hard_FORCES_juno2-lo_2025-01-15_11-48-36-651851_7xp5thny"
        # "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-15_updated_pregrasps_2/files/runs/watering_can_move2_2025-01-15_11-45-55-011129/nn/best.pth?runName=watering_can_move2_2025-01-15_11-45-55-011129_kb5rd4jf"
    )

    # Create the RL player
    player = RlPlayer(
        num_observations=num_observations,
        num_actions=num_actions,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    obs_dict = env.reset()

    """
    Loop
    """
    while True:
        """
        Compute action from RL player
        """
        # Get the normalized action from the RL player
        normalized_action = player.get_normalized_action(
            obs=obs_dict["obs"], deterministic_actions=False
        )
        assert_equals(normalized_action.shape, (1, num_actions))

        """
        Step env
        """
        obs_dict, _, _, _ = env.step(normalized_action)


if __name__ == "__main__":
    main()
