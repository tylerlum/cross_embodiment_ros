import copy

import numpy as np
import rospy
from live_plotter import FastLivePlotter
from sensor_msgs.msg import JointState


class LiveJointPlotterWithLimits:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("live_joint_plotter_with_limits", anonymous=True)

        # Initialize storage for latest joint state and command
        self.latest_joint_state = None
        self.latest_joint_cmd = None

        # Initialize lists to store joint states and commands
        self.joint_states = []
        self.joint_cmds = []

        # Joint limits (assumed to be for the robot)
        self.upper_joint_limits = (
            np.rad2deg(
                [
                    2.96705972839,
                    2.09439510239,
                    2.96705972839,
                    2.09439510239,
                    2.96705972839,
                    2.09439510239,
                    3.05432619099,
                    0.558488888889,
                    1.727825,
                    1.727825,
                    1.727825,
                    0.558488888889,
                    1.727825,
                    1.727825,
                    1.727825,
                    0.558488888889,
                    1.727825,
                    1.727825,
                    1.727825,
                    1.57075,
                    1.15188333333,
                    1.727825,
                    1.76273055556,
                ]
            )
            - 5
        )
        self.lower_joint_limits = (
            np.rad2deg(
                [
                    -2.96705972839,
                    -2.09439510239,
                    -2.96705972839,
                    -2.09439510239,
                    -2.96705972839,
                    -2.09439510239,
                    -3.05432619099,
                    -0.558488888889,
                    -0.279244444444,
                    -0.279244444444,
                    -0.279244444444,
                    -0.558488888889,
                    -0.279244444444,
                    -0.279244444444,
                    -0.279244444444,
                    -0.558488888889,
                    -0.279244444444,
                    -0.279244444444,
                    -0.279244444444,
                    0.279244444444,
                    -0.331602777778,
                    -0.279244444444,
                    -0.279244444444,
                ]
            )
            + 5
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
        plot_titles = [f"Joint {i+1} State" for i in range(self.num_joints)] + [
            f"Joint {i+1} Command" for i in range(self.num_joints)
        ]

        self.live_plotter = FastLivePlotter(
            n_cols=self.num_joints,
            n_rows=2,
            titles=plot_titles,
            ylims=[
                (lower, upper)
                for lower, upper in zip(
                    self.lower_joint_limits[: self.num_joints],
                    self.upper_joint_limits[: self.num_joints],
                )
            ]
            + [
                (lower, upper)
                for lower, upper in zip(
                    self.lower_joint_limits[: self.num_joints],
                    self.upper_joint_limits[: self.num_joints],
                )
            ],
        )

    def joint_state_callback(self, msg: JointState) -> None:
        """Callback function for joint state messages."""
        self.latest_joint_state = np.rad2deg(msg.position)

    def joint_cmd_callback(self, msg: JointState) -> None:
        """Callback function for joint command messages."""
        self.latest_joint_cmd = np.rad2deg(msg.position)

    def record_and_plot_data(self) -> None:
        """Main loop to record and plot data at 60Hz."""
        rate = rospy.Rate(60)  # 60Hz
        rospy.loginfo("Waiting for both joint states and commands to be available...")

        # Wait until both joint state and command are available
        while not rospy.is_shutdown():
            if (
                self.latest_joint_state is not None
                and self.latest_joint_cmd is not None
            ):
                rospy.loginfo(
                    "Both joint states and commands are available, starting to record and plot data..."
                )
                break
            rospy.sleep(0.1)

        # Start recording and plotting
        while not rospy.is_shutdown():
            # Record current joint states and commands
            latest_joint_state = copy.deepcopy(self.latest_joint_state)
            latest_joint_cmd = copy.deepcopy(self.latest_joint_cmd)
            assert latest_joint_state is not None
            assert latest_joint_cmd is not None

            self.joint_states.append(copy.deepcopy(self.latest_joint_state))
            self.joint_cmds.append(copy.deepcopy(self.latest_joint_cmd))

            # Plot the joint states and commands in real time
            if len(self.joint_states) > 1:
                y_data_list = []

                joint_states = np.array(self.joint_states)
                joint_cmds = np.array(self.joint_cmds)
                T = joint_states.shape[0]
                assert joint_states.shape == joint_cmds.shape == (T, self.num_joints)
                for i in range(self.num_joints):
                    y_data_list.append(joint_states[:, i])
                for i in range(self.num_joints):
                    y_data_list.append(joint_cmds[:, i])

                # Update the plot
                self.live_plotter.plot(
                    y_data_list=y_data_list,
                )

            # Check for joint limit violations
            for i in range(self.num_joints):
                if (
                    latest_joint_state[i] > self.upper_joint_limits[i]
                    or latest_joint_state[i] < self.lower_joint_limits[i]
                ):
                    rospy.logerr(
                        f"Joint {i+1} limit violation: {latest_joint_state[i]} not in [{self.lower_joint_limits[i]}, {self.upper_joint_limits[i]}]"
                    )
                if (
                    latest_joint_cmd[i] > self.upper_joint_limits[i]
                    or latest_joint_cmd[i] < self.lower_joint_limits[i]
                ):
                    rospy.logerr(
                        f"Joint {i+1} command limit violation: {latest_joint_cmd[i]} not in [{self.lower_joint_limits[i]}, {self.upper_joint_limits[i]}]"
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
