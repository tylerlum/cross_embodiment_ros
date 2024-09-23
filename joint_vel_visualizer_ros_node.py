import numpy as np
import rospy
from live_plotter import FastLivePlotter
from sensor_msgs.msg import JointState

# Constants
# Velocity limits (deg/sec) from iiwa_hardware/iiwa_control/config
N_JOINTS = 7
JOINT_VELOCITY_LIMITS_DEG = np.array([80, 80, 95, 70, 125, 130, 130])
assert len(JOINT_VELOCITY_LIMITS_DEG) == N_JOINTS

# Global variables to store joint velocities
JOINT_VELOCITIES_DEG = []


def joint_state_callback(msg: JointState) -> None:
    # Velocity comes in radians, convert to degrees
    global JOINT_VELOCITIES_DEG
    JOINT_VELOCITIES_DEG.append(np.rad2deg(msg.velocity))


def main():
    rospy.init_node("joint_vel_visualizer", anonymous=True)

    # Initialize plotter
    titles = [f"iiwa_joint_{i + 1}_vel" for i in range(N_JOINTS)]
    legends = [
        [
            "velocity",
            "lower limit",
            "upper limit",
        ]
        for _ in range(N_JOINTS)
    ]
    ylims = [
        [-JOINT_VELOCITY_LIMITS_DEG[i] - 10, JOINT_VELOCITY_LIMITS_DEG[i] + 10]
        for i in range(N_JOINTS)
    ]
    ylabels = ["deg/sec" for _ in range(N_JOINTS)]
    plotter = FastLivePlotter(
        titles=titles,
        legends=legends,
        ylims=ylims,
        ylabels=ylabels,
        n_plots=N_JOINTS,
    )

    # Subscribe to the /iiwa/joint_states topic
    rospy.Subscriber("/iiwa/joint_states", JointState, joint_state_callback)
    rospy.loginfo("Joint velocity subscriber started. Plotting joint velocities...")
    rate = rospy.Rate(60)  # 60Hz

    while not rospy.is_shutdown():
        current_joint_velocities_deg = np.array(JOINT_VELOCITIES_DEG)
        if len(current_joint_velocities_deg.shape) < 2:
            continue

        T, D = current_joint_velocities_deg.shape
        assert D == N_JOINTS
        lowers_deg = -JOINT_VELOCITY_LIMITS_DEG[None].repeat(T, axis=0)
        uppers_deg = JOINT_VELOCITY_LIMITS_DEG[None].repeat(T, axis=0)
        assert lowers_deg.shape == current_joint_velocities_deg.shape
        assert uppers_deg.shape == current_joint_velocities_deg.shape

        y_data_list = []
        for i in range(N_JOINTS):
            # Match order of legend
            y_data = np.stack(
                [
                    current_joint_velocities_deg[:, i],
                    lowers_deg[:, i],
                    uppers_deg[:, i],
                ],
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
