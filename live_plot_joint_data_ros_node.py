import numpy as np
import rospy
from live_plotter import FastLivePlotter
from sensor_msgs.msg import JointState


class LiveJointPlotterWithLimits:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("live_joint_plotter_with_limits")

        # Initialize storage for latest joint state and command
        self.latest_joint_state_msg = None
        self.latest_joint_cmd_msg = None

        # Initialize lists to store joint states and commands
        self.joint_poss_deg = []
        self.joint_cmds_deg = []

        # Joint limits
        self.upper_joint_limits_deg = np.rad2deg(
            [
                2.96705972839,
                2.09439510239,
                2.96705972839,
                2.09439510239,
                2.96705972839,
                2.09439510239,
                3.05432619099,
            ]
        )
        self.lower_joint_limits_deg = np.rad2deg(
            [
                -2.96705972839,
                -2.09439510239,
                -2.96705972839,
                -2.09439510239,
                -2.96705972839,
                -2.09439510239,
                -3.05432619099,
            ]
        )

        # Create subscribers to both topics
        self.sub_joint_state = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.joint_state_callback
        )
        self.sub_joint_cmd = rospy.Subscriber(
            "/iiwa/joint_cmd", JointState, self.joint_cmd_callback
        )

        # Set up a live plotter for joint states and commands
        self.num_joints = 7  # Assuming the robot has 7 joints
        plot_titles = [f"Joint {i+1} State and Command" for i in range(self.num_joints)]
        ylims = [
            (lower_deg - 10, upper_deg + 10)
            for lower_deg, upper_deg in zip(
                self.lower_joint_limits_deg[: self.num_joints],
                self.upper_joint_limits_deg[: self.num_joints],
            )
        ]
        ylabels = ["(deg)"] * self.num_joints
        legends = [
            ["Joint State", "Joint Command", "Lower Limit", "Upper Limit"]
        ] * self.num_joints

        self.live_plotter = FastLivePlotter(
            n_plots=self.num_joints,
            titles=plot_titles,
            ylims=ylims,
            ylabels=ylabels,
            legends=legends,
        )

    def joint_state_callback(self, msg: JointState) -> None:
        """Callback function for joint state messages."""
        self.latest_joint_state_msg = msg

    def joint_cmd_callback(self, msg: JointState) -> None:
        """Callback function for joint command messages."""
        self.latest_joint_cmd_msg = msg

    def record_and_plot_data(self) -> None:
        """Main loop to record and plot data at 60Hz."""
        rate = rospy.Rate(60)  # 60Hz
        rospy.loginfo("Waiting for both joint states and commands to be available...")

        # Wait until both joint state and command are available
        while not rospy.is_shutdown():
            if (
                self.latest_joint_state_msg is not None
                and self.latest_joint_cmd_msg is not None
            ):
                rospy.loginfo(
                    "Both joint states and commands are available, starting to record and plot data..."
                )
                break
            rospy.sleep(0.1)

        assert self.latest_joint_state_msg is not None
        assert self.latest_joint_cmd_msg is not None

        # Start recording and plotting
        while not rospy.is_shutdown():
            # Record current joint states and commands
            latest_joint_pos_deg = np.rad2deg(self.latest_joint_state_msg.position)
            latest_joint_cmd_deg = np.rad2deg(self.latest_joint_cmd_msg.position)
            assert latest_joint_pos_deg is not None
            assert latest_joint_cmd_deg is not None

            self.joint_poss_deg.append(latest_joint_pos_deg)
            self.joint_cmds_deg.append(latest_joint_cmd_deg)

            # Plot the joint states and commands in real time
            if len(self.joint_poss_deg) > 1:
                y_data_list = []

                joint_poss_deg = np.array(self.joint_poss_deg)
                joint_cmds_deg = np.array(self.joint_cmds_deg)
                T = joint_poss_deg.shape[0]
                assert (
                    joint_poss_deg.shape == joint_cmds_deg.shape == (T, self.num_joints)
                )
                for i in range(self.num_joints):
                    joint_pos_deg = joint_poss_deg[:, i]
                    joint_cmd_deg = joint_cmds_deg[:, i]
                    lower_limit = np.ones(T) * self.lower_joint_limits_deg[i]
                    upper_limit = np.ones(T) * self.upper_joint_limits_deg[i]
                    y_data = np.stack(
                        [joint_pos_deg, joint_cmd_deg, lower_limit, upper_limit], axis=1
                    )
                    y_data_list.append(y_data)

                # Update the plot
                self.live_plotter.plot(
                    y_data_list=y_data_list,
                )

            # Check for joint limit violations
            for i in range(self.num_joints):
                if (
                    latest_joint_pos_deg[i] > self.upper_joint_limits_deg[i]
                    or latest_joint_pos_deg[i] < self.lower_joint_limits_deg[i]
                ):
                    rospy.logerr(
                        f"Joint {i+1} limit violation: {latest_joint_pos_deg[i]} not in [{self.lower_joint_limits_deg[i]}, {self.upper_joint_limits_deg[i]}]"
                    )
                if (
                    latest_joint_cmd_deg[i] > self.upper_joint_limits_deg[i]
                    or latest_joint_cmd_deg[i] < self.lower_joint_limits_deg[i]
                ):
                    rospy.logerr(
                        f"Joint {i+1} command limit violation: {latest_joint_cmd_deg[i]} not in [{self.lower_joint_limits_deg[i]}, {self.upper_joint_limits_deg[i]}]"
                    )

            rate.sleep()

    def run(self):
        try:
            # Start recording and plotting
            self.record_and_plot_data()
        except rospy.ROSInterruptException:
            pass


if __name__ == "__main__":
    joint_plotter = LiveJointPlotterWithLimits()
    joint_plotter.run()
