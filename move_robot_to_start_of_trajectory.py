#!/usr/bin/env python

from pathlib import Path

import numpy as np
import rospy
from sensor_msgs.msg import JointState

IIWA_NAMES = [
    "iiwa_joint_1",
    "iiwa_joint_2",
    "iiwa_joint_3",
    "iiwa_joint_4",
    "iiwa_joint_5",
    "iiwa_joint_6",
    "iiwa_joint_7",
]
ALLEGRO_NAMES = [
    "allegro_joint_0",
    "allegro_joint_1",
    "allegro_joint_2",
    "allegro_joint_3",
    "allegro_joint_4",
    "allegro_joint_5",
    "allegro_joint_6",
    "allegro_joint_7",
    "allegro_joint_8",
    "allegro_joint_9",
    "allegro_joint_10",
    "allegro_joint_11",
    "allegro_joint_12",
    "allegro_joint_13",
    "allegro_joint_14",
    "allegro_joint_15",
]


class MoveRobotToTrajectoryStart:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("move_robot_to_trajectory_start")

        # Initial joint positions
        self.init_iiwa_joint_pos = None
        self.init_allegro_joint_pos = None

        # Subscribers
        self.iiwa_sub = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.iiwa_joint_state_callback
        )
        self.allegro_sub = rospy.Subscriber(
            "/allegroHand_0/joint_states", JointState, self.allegro_joint_state_callback
        )

        # Publishers
        self.iiwa_cmd_pub = rospy.Publisher(
            "/iiwa/joint_cmd", JointState, queue_size=10
        )
        self.allegro_cmd_pub = rospy.Publisher(
            "/allegroHand_0/joint_cmd", JointState, queue_size=10
        )

        # Load trajectory
        TASK_NAME = "snackbox_pivot_easy"
        self.filepath = Path(
            f"/juno/u/tylerlum/github_repos/interactive_robot_visualizer/outputs/{TASK_NAME}.npz"
        )
        data = np.load(self.filepath)
        self.target_q = data["qs"][0]  # Get first configuration
        assert self.target_q.shape == (
            23,
        ), f"Expected 23 joints, got {self.target_q.shape}"

        # Parameters
        self.rate = rospy.Rate(60)  # 60 Hz
        self.MOVE_DURATION = 10.0  # seconds
        self.STATIONARY_TIME = 2.0  # seconds to wait before moving

        # Wait for initial joint positions
        self.get_initial_joint_positions()

    def iiwa_joint_state_callback(self, msg: JointState) -> None:
        if self.init_iiwa_joint_pos is None:
            self.init_iiwa_joint_pos = np.array(msg.position)
            rospy.loginfo(f"Got initial IIWA positions: {self.init_iiwa_joint_pos}")

    def allegro_joint_state_callback(self, msg: JointState) -> None:
        if self.init_allegro_joint_pos is None:
            self.init_allegro_joint_pos = np.array(msg.position)
            rospy.loginfo(
                f"Got initial Allegro positions: {self.init_allegro_joint_pos}"
            )

    def get_initial_joint_positions(self):
        while not rospy.is_shutdown():
            if (
                self.init_iiwa_joint_pos is not None
                and self.init_allegro_joint_pos is not None
            ):
                rospy.loginfo("Got all initial joint positions")
                break
            rospy.loginfo("Waiting for initial joint positions...")
            rospy.sleep(0.1)

    def run(self):
        start_time = rospy.Time.now()
        last_publish_time = rospy.Time.now()
        init_q = np.concatenate([self.init_iiwa_joint_pos, self.init_allegro_joint_pos])

        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            elapsed_time = (current_time - start_time).to_sec()

            # Wait for STATIONARY_TIME before moving
            if elapsed_time < self.STATIONARY_TIME:
                rospy.loginfo_throttle(
                    1.0,
                    f"Holding position for {self.STATIONARY_TIME - elapsed_time:.1f} more seconds",
                )
                current_q = init_q
            else:
                # Interpolate between initial and target position
                alpha = (elapsed_time - self.STATIONARY_TIME) / self.MOVE_DURATION
                alpha = min(1.0, alpha)
                current_q = init_q * (1 - alpha) + self.target_q * alpha

                if alpha >= 1.0:
                    rospy.loginfo_throttle(1.0, "Reached target position")

            # Publish IIWA command
            iiwa_msg = JointState()
            iiwa_msg.header.stamp = current_time
            iiwa_msg.name = IIWA_NAMES
            iiwa_msg.position = current_q[:7].tolist()
            iiwa_msg.velocity = []
            iiwa_msg.effort = []
            self.iiwa_cmd_pub.publish(iiwa_msg)

            # Publish Allegro command
            allegro_msg = JointState()
            allegro_msg.header.stamp = current_time
            allegro_msg.name = ALLEGRO_NAMES
            allegro_msg.position = current_q[7:].tolist()
            allegro_msg.velocity = []
            allegro_msg.effort = []
            self.allegro_cmd_pub.publish(allegro_msg)

            # Log publishing rate
            time_since_last_publish = (current_time - last_publish_time).to_sec()
            if time_since_last_publish > 0.2:
                rospy.loginfo("\n" + "=" * 80)
                rospy.loginfo("SLOW")
                rospy.loginfo("=" * 80 + "\n")

            rospy.loginfo(
                f"Publishing ({np.round(time_since_last_publish * 1000)} ms, {np.round(1./time_since_last_publish)} Hz)"
            )

            last_publish_time = current_time
            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = MoveRobotToTrajectoryStart()
        node.run()
    except rospy.ROSInterruptException:
        pass
