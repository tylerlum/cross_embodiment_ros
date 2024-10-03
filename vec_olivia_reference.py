from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_jit_utils import matrix_to_quat_wxyz

class VecOliviaReferenceMotion:
    def __init__(
        self,
        trajectory_folder: Path,
        hand_folder: Path,
        batch_size: int,
        device: str,
        dt: float,
    ) -> None:
        self._trajectory_folder = trajectory_folder
        self._hand_folder = hand_folder
        self._batch_size = batch_size
        self._device = device
        self._dt = dt

        self._load_motion(
            trajectory_folder=trajectory_folder, hand_folder=hand_folder, device=device
        )
        self._substeps = 1
        self._data_substeps = 1

        self._step = torch.zeros(batch_size, device=device, dtype=torch.long)

    def _load_motion(
        self, trajectory_folder: Path, hand_folder: Path, device: str
    ) -> None:
        assert trajectory_folder.is_dir(), f"{trajectory_folder} is not a directory"
        txt_files = sorted(list(trajectory_folder.rglob("**/ob_in_cam/*.txt")))
        if len(txt_files) == 0:
            raise ValueError(f"No txt files found in {trajectory_folder}")

        self.T_list = torch.stack(
            [
                torch.from_numpy(np.loadtxt(filename)).to(device).float()
                for filename in txt_files
            ],
            dim=0,
        )

        N = self.T_list.shape[0]
        assert self.T_list.shape == (
            N,
            4,
            4,
        ), f"Expected shape {(N, 4, 4)}, got {self.T_list.shape}"

        USE_HAND_TO_CLIP = False
        if USE_HAND_TO_CLIP:
            hand_jpg_files = sorted(list(hand_folder.glob("*_all.jpg")))
            if len(hand_jpg_files) == 0:
                raise ValueError(f"No jpg files found in {hand_folder}")
            idxs = [
                int(filename.name.split("_all.jpg")[0]) for filename in hand_jpg_files
            ]

            self._start_step = idxs[0]
            self._end_step = idxs[-1]
        else:
            all_xyz = self.T_list[:, :3, 3].cpu().numpy()

            speeds = np.linalg.norm(all_xyz[1:] - all_xyz[:-1], axis=-1) / (
                2 * self._dt
            )
            MIN_SPEED = 0.02
            WINDOW = 10
            # import matplotlib.pyplot as plt
            # plt.plot(speeds)
            # plt.show()
            # breakpoint()
            moving = speeds > MIN_SPEED

            is_consecutively_moving = np.array(
                [
                    np.sum(moving[i : i + WINDOW]) / WINDOW > 0.5
                    for i in range(N - WINDOW)
                ]
            )
            is_consecutively_moving_idxs = np.where(is_consecutively_moving)[0]
            if is_consecutively_moving_idxs.size == 0:
                raise ValueError(f"No moving object found in {trajectory_folder}, speeds = {speeds}, moving = {moving}, np.max(speeds) = {np.max(speeds)}")

            self._start_step = is_consecutively_moving_idxs[0]

            END_EARLY = False
            if END_EARLY:
                self._end_step = is_consecutively_moving_idxs[-1] + WINDOW - 1
            else:
                self._end_step = N - 1

    def reset_idx(self, reset_env_ids: torch.Tensor) -> None:
        self._step[reset_env_ids] = 0

    def step_idx(self, step_env_ids: torch.Tensor) -> None:
        self._check_valid_step(step_env_ids)
        self._step[step_env_ids] += self._data_substeps

    def revert(self) -> None:
        self._step -= self._data_substeps
        self._check_valid_step()

    def __len__(self) -> int:
        return self.length

    @property
    def t(self) -> torch.Tensor:
        return self._step

    @property
    def data_substep(self) -> int:
        return self._data_substeps

    @property
    def time(self) -> torch.Tensor:
        return self._step / self.length

    @property
    def substeps(self) -> int:
        return self._substeps

    @property
    def done(self) -> torch.Tensor:
        return self._step >= self.length

    @property
    def next_done(self) -> torch.Tensor:
        return self._step >= self.length - self._data_substeps

    @property
    def n_left(self) -> torch.Tensor:
        n_left = (self.length - self._step) / float(self._data_substeps) - 1
        return torch.clamp(n_left, min=0)

    @property
    def n_steps(self) -> int:
        n_steps = self.length / float(self._data_substeps) - 1
        return int(max(n_steps, 0))

    @property
    def length(self) -> int:
        return self._end_step - self._start_step + 1

    @property
    def start_step(self) -> int:
        return self._start_step

    @property
    def end_step(self) -> int:
        return self._end_step

    def _check_valid_step(self, check_env_ids: Optional[torch.Tensor] = None) -> None:
        if check_env_ids is None:
            check_env_ids = (
                torch.ones(self._batch_size, device=self._device)
                .nonzero(as_tuple=False)
                .squeeze(dim=-1)
            )

        assert not torch.any(
            self.done[check_env_ids]
        ), "Attempting access data and/or step 'done' motion"
        assert torch.all(
            self._step[check_env_ids] >= 0
        ), "step must be at least start_step"
        assert torch.all(
            self._step[check_env_ids] < self.length
        ), "step must be at most end_step"

    @property
    def object_pos(self) -> torch.Tensor:
        self._check_valid_step()
        return self.T_list[self._step + self._start_step, :3, 3].clone()

    @property
    def floor_z(self) -> float:
        first_T = self.T_list[self._start_step]
        return float(first_T[2, 3].item())

    @property
    def object_rot(self) -> torch.Tensor:
        self._check_valid_step()
        return matrix_to_quat_wxyz(self.T_list[self._step + self._start_step, :3, :3])

    @property
    def object_quat_wxyz(self) -> torch.Tensor:
        return self.object_rot

    @property
    def object_quat_xyzw(self) -> torch.Tensor:
        return self.object_quat_wxyz[..., [1, 2, 3, 0]]

    @property
    def init_object_pos(self) -> torch.Tensor:
        self._check_valid_step()
        first_T = self.T_list[self._start_step]
        return first_T[:3, 3].clone()

    @property
    def init_object_rot(self) -> torch.Tensor:
        self._check_valid_step()
        first_T = self.T_list[self._start_step]
        return matrix_to_quat_wxyz(first_T[:3, :3])

    @property
    def init_object_quat_wxyz(self) -> torch.Tensor:
        return self.init_object_rot

    @property
    def init_object_quat_xyzw(self) -> torch.Tensor:
        return self.init_object_quat_wxyz[..., [1, 2, 3, 0]]
