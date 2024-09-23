import numpy as np
import rospy
from live_plotter import FastLivePlotter
from sensor_msgs.msg import JointState


class FabricVisualizer:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("fabric_visualizer")

        # Initialize storage for latest fabric, iiwa, and allegro joint states
        self.latest_fabric_state_msg = None
        self.latest_iiwa_state_msg = None
        self.latest_allegro_state_msg = None

        # Initialize lists to store positions and velocities
        self.fabric_joint_positions_deg = []
        self.fabric_joint_velocities_deg_s = []
        self.joint_positions_deg = []
        self.joint_velocities_deg_s = []

        # Position and velocity limits
        self.position_limits_deg = np.array([165, 115, 165, 115, 165, 115, 170])
        self.velocity_limits_deg_s = np.array([80, 80, 95, 70, 125, 130, 130])

        # Create subscribers to the /fabric_state, /iiwa/joint_state, and /allegroHand_0/joint_state topics
        self.sub_fabric_state = rospy.Subscriber(
            "/fabric_state", JointState, self.fabric_state_callback
        )
        self.sub_iiwa_state = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.iiwa_state_callback
        )
        self.sub_allegro_state = rospy.Subscriber(
            "/allegroHand_0/joint_states", JointState, self.allegro_state_callback
        )

        # Set up a live plotter for positions and velocities
        self.num_joints = 7  # First 7 elements for plotting
        plot_titles = [f"Joint {i+1} Position" for i in range(self.num_joints)] + [
            f"Joint {i+1} Velocity" for i in range(self.num_joints)
        ]
        ylims = [(-limit - 10, limit + 10) for limit in self.position_limits_deg] + [
            (-limit - 10, limit + 10) for limit in self.velocity_limits_deg_s
        ]
        ylabels = ["(deg)"] * self.num_joints + ["(deg/s)"] * self.num_joints
        legends = [
            ["Position", "Fabric", "Lower Limit", "Upper Limit"]
        ] * self.num_joints + [
            ["Velocity", "Fabric", "Lower Limit", "Upper Limit"]
        ] * self.num_joints

        self.live_plotter = FastLivePlotter(
            n_plots=2 * self.num_joints,
            titles=plot_titles,
            ylims=ylims,
            ylabels=ylabels,
            legends=legends,
            n_rows=2,  # 2 rows: positions and velocities
        )

    def fabric_state_callback(self, msg: JointState) -> None:
        """Callback function for fabric state messages."""
        self.latest_fabric_state_msg = msg

    def iiwa_state_callback(self, msg: JointState) -> None:
        """Callback function for iiwa state messages."""
        self.latest_iiwa_state_msg = msg

    def allegro_state_callback(self, msg: JointState) -> None:
        """Callback function for allegro hand state messages."""
        self.latest_allegro_state_msg = msg

    def record_and_plot_data(self) -> None:
        """Main loop to record and plot data at 60Hz."""
        rate = rospy.Rate(60)  # 60Hz
        rospy.loginfo(
            "Waiting for fabric, iiwa, and allegro joint states to be available..."
        )

        # Wait until all state messages are available
        while not rospy.is_shutdown():
            if (
                self.latest_fabric_state_msg is not None
                and self.latest_iiwa_state_msg is not None
                and self.latest_allegro_state_msg is not None
            ):
                rospy.loginfo(
                    "All joint states are available, starting to record and plot data..."
                )
                break
            rospy.sleep(0.1)

        assert self.latest_fabric_state_msg is not None
        assert self.latest_iiwa_state_msg is not None
        assert self.latest_allegro_state_msg is not None

        # Start recording and plotting
        while not rospy.is_shutdown():
            # Record current joint positions and velocities (in radians and rad/s)
            latest_fabric_joint_pos_deg = np.rad2deg(
                self.latest_fabric_state_msg.position
            )
            latest_fabric_joint_vel_deg_s = np.rad2deg(
                self.latest_fabric_state_msg.velocity
            )

            latest_joint_pos_deg = np.concatenate(
                [
                    np.rad2deg(self.latest_iiwa_state_msg.position),
                    np.rad2deg(self.latest_allegro_state_msg.position),
                ]
            )
            latest_joint_vel_deg_s = np.concatenate(
                [
                    np.rad2deg(self.latest_iiwa_state_msg.velocity),
                    np.rad2deg(self.latest_allegro_state_msg.velocity),
                ]
            )

            assert latest_fabric_joint_pos_deg is not None
            assert latest_fabric_joint_vel_deg_s is not None
            assert latest_joint_pos_deg is not None
            assert latest_joint_vel_deg_s is not None

            self.fabric_joint_positions_deg.append(
                latest_fabric_joint_pos_deg[: self.num_joints]
            )
            self.fabric_joint_velocities_deg_s.append(
                latest_fabric_joint_vel_deg_s[: self.num_joints]
            )
            self.joint_positions_deg.append(latest_joint_pos_deg[: self.num_joints])
            self.joint_velocities_deg_s.append(
                latest_joint_vel_deg_s[: self.num_joints]
            )

            # Plot the joint positions and velocities in real time
            if len(self.fabric_joint_positions_deg) <= 1:
                rospy.loginfo(
                    f"Waiting for more data to plot, len(self.fabric_joint_positions_deg) = {len(self.fabric_joint_positions_deg)} ..."
                )
                continue

            y_data_list = []

            fabric_positions_deg = np.array(self.fabric_joint_positions_deg)
            fabric_velocities_deg_s = np.array(self.fabric_joint_velocities_deg_s)
            joint_positions_deg = np.array(self.joint_positions_deg)
            joint_velocities_deg_s = np.array(self.joint_velocities_deg_s)
            T = fabric_positions_deg.shape[0]
            assert (
                fabric_positions_deg.shape
                == fabric_velocities_deg_s.shape
                == (T, self.num_joints)
            )

            # Prepare data for positions
            for i in range(self.num_joints):
                fabric_pos_deg = fabric_positions_deg[:, i]
                joint_pos_deg = joint_positions_deg[:, i]
                lower_limit_pos = np.ones(T) * (-self.position_limits_deg[i])
                upper_limit_pos = np.ones(T) * self.position_limits_deg[i]
                y_data_pos = np.stack(
                    [joint_pos_deg, fabric_pos_deg, lower_limit_pos, upper_limit_pos],
                    axis=1,
                )
                y_data_list.append(y_data_pos)

            # Prepare data for velocities
            for i in range(self.num_joints):
                fabric_vel_deg_s = fabric_velocities_deg_s[:, i]
                joint_vel_deg_s = joint_velocities_deg_s[:, i]
                lower_limit_vel = np.ones(T) * (-self.velocity_limits_deg_s[i])
                upper_limit_vel = np.ones(T) * self.velocity_limits_deg_s[i]
                y_data_vel = np.stack(
                    [
                        joint_vel_deg_s,
                        fabric_vel_deg_s,
                        lower_limit_vel,
                        upper_limit_vel,
                    ],
                    axis=1,
                )
                y_data_list.append(y_data_vel)

            # Update the plot
            self.live_plotter.plot(
                y_data_list=y_data_list,
            )

            # Check for position and velocity limit violations
            for i in range(self.num_joints):
                if (
                    latest_fabric_joint_pos_deg[i] > self.position_limits_deg[i]
                    or latest_fabric_joint_pos_deg[i] < -self.position_limits_deg[i]
                ):
                    rospy.logerr(
                        f"Joint {i+1} fabric position limit violation: {latest_fabric_joint_pos_deg[i]} not in [{-self.position_limits_deg[i]}, {self.position_limits_deg[i]}]"
                    )
                if (
                    latest_fabric_joint_vel_deg_s[i] > self.velocity_limits_deg_s[i]
                    or latest_fabric_joint_vel_deg_s[i] < -self.velocity_limits_deg_s[i]
                ):
                    rospy.logerr(
                        f"Joint {i+1} fabric velocity limit violation: {latest_fabric_joint_vel_deg_s[i]} not in [{-self.velocity_limits_deg_s[i]}, {self.velocity_limits_deg_s[i]}]"
                    )
                if (
                    latest_joint_pos_deg[i] > self.position_limits_deg[i]
                    or latest_joint_pos_deg[i] < -self.position_limits_deg[i]
                ):
                    rospy.logerr(
                        f"Joint {i+1} position limit violation: {latest_joint_pos_deg[i]} not in [{-self.position_limits_deg[i]}, {self.position_limits_deg[i]}]"
                    )
                if (
                    latest_joint_vel_deg_s[i] > self.velocity_limits_deg_s[i]
                    or latest_joint_vel_deg_s[i] < -self.velocity_limits_deg_s[i]
                ):
                    rospy.logerr(
                        f"Joint {i+1} velocity limit violation: {latest_joint_vel_deg_s[i]} not in [{-self.velocity_limits_deg_s[i]}, {self.velocity_limits_deg_s[i]}]"
                    )

            rate.sleep()

    def run(self):
        try:
            # Start recording and plotting
            self.record_and_plot_data()
        except rospy.ROSInterruptException:
            pass


if __name__ == "__main__":
    fabric_visualizer = FabricVisualizer()
    fabric_visualizer.run()
