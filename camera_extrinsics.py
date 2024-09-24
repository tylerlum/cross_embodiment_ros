import numpy as np


# C is camera frame with Z forward and Y down
# From Claire's extrinsics
T_R_C = np.eye(4)
T_R_C[:3, :3] = np.array(
    [
        [0.9993793163153581, -0.032378481601275176, 0.013878618455879281],
        [-0.026045538333648588, -0.41386448768757556, 0.9099658321959191],
        [-0.0237194446384908, -0.9097625073393034, -0.4144509237361466],
    ]
)
T_R_C[:3, 3] = np.array([0.3462484470027277, -0.8607951520007057, 0.7653137967937953])


# C is camera frame with Z forward and Y down
# From Claire's extrinsics
T_R_C = np.eye(4)
T_R_C[:3, :3] = np.array(
    [
        [0.9993793163153581, -0.032378481601275176, 0.013878618455879281],
        [-0.026045538333648588, -0.41386448768757556, 0.9099658321959191],
        [-0.0237194446384908, -0.9097625073393034, -0.4144509237361466],
    ]
)
T_R_C[:3, 3] = np.array([0.3462484470027277, -0.8607951520007057, 0.7653137967937953])

# C2 is camera frame with X forward and Y left
# https://community.stereolabs.com/t/coordinate-system-of-pointcloud/908/2
T_C_C2 = np.eye(4)
T_C_C2[:3, :3] = np.array(
    [
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0],
    ]
)
T_R_C2 = T_R_C @ T_C_C2
