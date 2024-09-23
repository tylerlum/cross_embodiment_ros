import numpy as np
import rospy
from live_plotter import FastLivePlotter
from sensor_msgs.msg import JointState

# Constants
N_JOINTS = 7
JOINT_ANGLE_LIMITS = np.deg2rad(
    [165, 115, 165, 115, 165, 115, 170]
)  # from iiwa_hardware/iiwa_control/config
assert len(JOINT_ANGLE_LIMITS) == N_JOINTS

# Global variables to store joint positions
joint_angles = []


def joint_state_callback(msg: JointState) -> None:
    global joint_angles
    # Store joint angles over time
    joint_angles.append(np.array(msg.position))


def main():
    rospy.init_node("joint_state_subscriber", anonymous=True)

    # Initialize plotter
    titles = [f"iiwa_joint_{i + 1}" for i in range(N_JOINTS)]
    legends = [
        [
            "pos",
            "lower limit",
            "upper limit",
        ]
        for _ in range(N_JOINTS)
    ]
    ylims = [
        [-JOINT_ANGLE_LIMITS[i] - 0.1, JOINT_ANGLE_LIMITS[i] + 0.1]
        for i in range(N_JOINTS)
    ]
    plotter = FastLivePlotter(
        titles=titles,
        legends=legends,
        ylims=ylims,
        n_plots=N_JOINTS,
    )

    # Subscribe to the /iiwa/joint_states topic
    rospy.Subscriber("/iiwa/joint_states", JointState, joint_state_callback)

    rospy.loginfo("Joint state subscriber started. Plotting joint angles...")

    while not rospy.is_shutdown():
        joint_angles_now = np.array(joint_angles)
        if len(joint_angles_now.shape) < 2:
            continue

        T, D = joint_angles_now.shape
        assert D == N_JOINTS
        lowers = -JOINT_ANGLE_LIMITS[None].repeat(T, axis=0)
        uppers = JOINT_ANGLE_LIMITS[None].repeat(T, axis=0)
        assert lowers.shape == joint_angles_now.shape
        assert uppers.shape == joint_angles_now.shape

        y_data_list = []
        for i in range(N_JOINTS):
            # Match order of legend
            y_data = np.stack(
                [joint_angles_now[:, i], lowers[:, i], uppers[:, i]], axis=1
            )
            assert y_data.shape == (T, 3)
            y_data_list.append(y_data)

        plotter.plot(
            y_data_list=y_data_list,
        )


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
