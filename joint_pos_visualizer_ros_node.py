import numpy as np
import rospy
from live_plotter import FastLivePlotter
from sensor_msgs.msg import JointState

# Constants
# Limits from iiwa_hardware/iiwa_control/config
N_JOINTS = 7
JOINT_ANGLE_LIMITS_DEG = np.array([165, 115, 165, 115, 165, 115, 170])
assert len(JOINT_ANGLE_LIMITS_DEG) == N_JOINTS

# Global variables to store joint positions
JOINT_ANGLES_DEG = []


def joint_state_callback(msg: JointState) -> None:
    # Comes in radians
    global JOINT_ANGLES_DEG
    JOINT_ANGLES_DEG.append(np.rad2deg(msg.position))


def main():
    rospy.init_node("joint_pos_visualizer", anonymous=True)

    # Initialize plotter
    titles = [f"iiwa_joint_{i + 1}_pos" for i in range(N_JOINTS)]
    legends = [
        [
            "pos",
            "lower limit",
            "upper limit",
        ]
        for _ in range(N_JOINTS)
    ]
    ylims = [
        [-JOINT_ANGLE_LIMITS_DEG[i] - 10, JOINT_ANGLE_LIMITS_DEG[i] + 10]
        for i in range(N_JOINTS)
    ]
    ylabels = ["deg" for _ in range(N_JOINTS)]
    plotter = FastLivePlotter(
        titles=titles,
        legends=legends,
        ylims=ylims,
        ylabels=ylabels,
        n_plots=N_JOINTS,
    )

    # Subscribe to the /iiwa/joint_states topic
    rospy.Subscriber("/iiwa/joint_states", JointState, joint_state_callback)
    rospy.loginfo("Joint state subscriber started. Plotting joint angles...")
    rate = rospy.Rate(60)  # 60Hz

    while not rospy.is_shutdown():
        current_joint_angles_deg = np.array(JOINT_ANGLES_DEG)
        if len(current_joint_angles_deg.shape) < 2:
            continue

        T, D = current_joint_angles_deg.shape
        assert D == N_JOINTS
        lowers_deg = -JOINT_ANGLE_LIMITS_DEG[None].repeat(T, axis=0)
        uppers_deg = JOINT_ANGLE_LIMITS_DEG[None].repeat(T, axis=0)
        assert lowers_deg.shape == current_joint_angles_deg.shape
        assert uppers_deg.shape == current_joint_angles_deg.shape

        y_data_list = []
        for i in range(N_JOINTS):
            # Match order of legend
            y_data = np.stack(
                [current_joint_angles_deg[:, i], lowers_deg[:, i], uppers_deg[:, i]],
                axis=1,
            )
            assert y_data.shape == (T, 3)
            y_data_list.append(y_data)

        plotter.plot(
            y_data_list=y_data_list,
        )
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
