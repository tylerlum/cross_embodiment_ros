import numpy as np
from scipy.spatial.transform import Rotation as R


def transform_str_to_T(transform_str: str) -> np.ndarray:
    # Go from string like f"0 0 {HEIGHT} 0 0 0 1"  # x y z qx qy qz qw
    # To 4x4 matrix
    xyz_qxyzw = np.array([float(x) for x in transform_str.split()])
    from scipy.spatial.transform import Rotation as R

    rotation_matrix = R.from_quat(xyz_qxyzw[3:]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = xyz_qxyzw[:3]
    return T


def T_to_transform_str(T: np.ndarray) -> str:
    # Go from 4x4 matrix to string like f"0 0 {HEIGHT} 0 0 0 1"  # x y z qx qy qz qw
    r = R.from_matrix(T[:3, :3])
    xyz_qxyzw = np.zeros(7)
    xyz_qxyzw[:3] = T[:3, 3]
    xyz_qxyzw[3:] = r.as_quat()
    return " ".join([str(x) for x in xyz_qxyzw])


THICKNESS = 0.02
HEIGHT = 1
world_dict_table_frame = {
    "right_wall": {
        "env_index": "all",
        "type": "box",
        "scaling": f"0.9 {THICKNESS} {HEIGHT}",
        "transform": f"0 0.37 {0.5 * HEIGHT} 0 0 0 1",  # x y z qx qy qz qw
    },
    "left_wall": {
        "env_index": "all",
        "type": "box",
        "scaling": f"0.9 {THICKNESS} {HEIGHT}",
        "transform": f"0 -0.37 {0.5 * HEIGHT} 0 0 0 1",  # x y z qx qy qz qw
    },
    "back_wall": {
        "env_index": "all",
        "type": "box",
        "scaling": f"{THICKNESS} 0.74 {HEIGHT}",
        "transform": f"-0.45 0 {0.5 * HEIGHT} 0 0 0 1",  # x y z qx qy qz qw
    },
    "front_wall": {
        "env_index": "all",
        "type": "box",
        "scaling": f"{THICKNESS} 0.74 {HEIGHT}",
        "transform": f"0.45 0 {0.5 * HEIGHT} 0 0 0 1",  # x y z qx qy qz qw
    },
    "table": {
        "env_index": "all",
        "type": "box",
        "scaling": f"0.45 0.74 {THICKNESS}",
        "transform": "0.225 0 0 0 0 0 1",  # x y z qx qy qz qw
    },
    "ceiling": {
        "env_index": "all",
        "type": "box",
        "scaling": f"0.9 0.74 {THICKNESS}",
        "transform": f"0 0 {HEIGHT} 0 0 0 1",  # x y z qx qy qz qw
    },
}

T_R_O = np.eye(4)
T_R_O[:3, 3] = [0.2, -0.2, 0.14]
T_R_O[:3, :3] = R.from_euler("z", -36, degrees=True).as_matrix()

world_dict_robot_frame = {
    k: {
        "env_index": v["env_index"],
        "type": v["type"],
        "scaling": v["scaling"],
        "transform": T_to_transform_str(T_R_O @ transform_str_to_T(v["transform"])),
    }
    for k, v in world_dict_table_frame.items()
}
