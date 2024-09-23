#!/usr/bin/env python

from typing import Literal

import numpy as np
import rospy
from sensor_msgs.msg import JointState

NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16

DEFAULT_ARM_Q = np.array(
    [
        0,
        0,
        0,
        -1.57,
        0,
        1.57,
        0,
    ]
)
DEFAULT_HAND_Q = np.zeros(NUM_HAND_JOINTS)


class FakeRobotNode:
    def __init__(self):
        # ROS setup
        rospy.init_node("fake_robot_ros_node")

        # ROS msgs
        self.iiwa_joint_cmd = None
        self.allegro_joint_cmd = None

        # Publisher and subscriber
        self.iiwa_pub = rospy.Publisher("/iiwa/joint_states", JointState, queue_size=10)
        self.allegro_pub = rospy.Publisher(
            "/allegroHand_0/joint_states", JointState, queue_size=10
        )
        self.iiwa_cmd_sub = rospy.Subscriber(
            "/iiwa/joint_cmd", JointState, self.iiwa_joint_cmd_callback
        )
        self.allegro_cmd_sub = rospy.Subscriber(
            "/allegroHand_0/joint_cmd", JointState, self.allegro_joint_cmd_callback
        )

        # State
        self.iiwa_joint_q = DEFAULT_ARM_Q
        self.allegro_joint_q = DEFAULT_HAND_Q
        self.iiwa_joint_qd = np.zeros(NUM_ARM_JOINTS)
        self.allegro_joint_qd = np.zeros(NUM_HAND_JOINTS)

        # Set control rate to 60Hz
        self.rate_hz = 60
        self.dt = 1 / self.rate_hz
        self.rate = rospy.Rate(self.rate_hz)

        # When only testing the arm, set this to False to ignore the Allegro hand
        self.WAIT_FOR_ALLEGRO_CMD = False
        if not self.WAIT_FOR_ALLEGRO_CMD:
            rospy.logwarn("NOT WAITING FOR ALLEGRO CMD")
            self.allegro_joint_cmd = np.zeros(NUM_HAND_JOINTS)

    def iiwa_joint_cmd_callback(self, msg: JointState):
        """Callback to update the commanded joint positions."""
        self.iiwa_joint_cmd = np.array(msg.position)

    def allegro_joint_cmd_callback(self, msg: JointState):
        """Callback to update the commanded joint positions."""
        self.allegro_joint_cmd = np.array(msg.position)

    def update_joint_states(self):
        """Update the PyBullet simulation with the commanded joint positions."""
        if self.iiwa_joint_cmd is None or self.allegro_joint_cmd is None:
            rospy.loginfo(
                f"Waiting: iiwa_joint_cmd: {self.iiwa_joint_cmd}, allegro_joint_cmd: {self.allegro_joint_cmd}"
            )
            return

        rospy.loginfo(
            f"Updating PyBullet with iiwa joint commands: {self.iiwa_joint_cmd}, allegro joint commands: {self.allegro_joint_cmd}"
        )

        delta_iiwa = self.iiwa_joint_cmd - self.iiwa_joint_q
        delta_allegro = self.allegro_joint_cmd - self.allegro_joint_q

        MODE: Literal["INTERPOLATE", "PD_CONTROL"] = "INTERPOLATE"
        if MODE == "INTERPOLATE":
            delta_iiwa_norm = np.linalg.norm(delta_iiwa)
            delta_allegro_norm = np.linalg.norm(delta_allegro)

            MAX_DELTA_IIWA = 0.1
            MAX_DELTA_ALLEGRO = 0.1
            if delta_iiwa_norm > MAX_DELTA_IIWA:
                delta_iiwa = MAX_DELTA_IIWA * delta_iiwa / delta_iiwa_norm
            if delta_allegro_norm > MAX_DELTA_ALLEGRO:
                delta_allegro = MAX_DELTA_ALLEGRO * delta_allegro / delta_allegro_norm

            self.iiwa_joint_q += delta_iiwa
            self.allegro_joint_q += delta_allegro
            self.iiwa_joint_qd = delta_iiwa / self.dt
            self.allegro_joint_qd = np.zeros(NUM_HAND_JOINTS)
        elif MODE == "PD_CONTROL":
            P = 10
            D = 0
            iiwa_qd_cmd = 0
            allegro_qd_cmd = 0
            delta_iiwa_qd = iiwa_qd_cmd - self.iiwa_joint_qd
            delta_allegro_qd = allegro_qd_cmd - self.allegro_joint_qd

            iiwa_qdd = P * delta_iiwa + D * delta_iiwa_qd
            allegro_qdd = P * delta_allegro + D * delta_allegro_qd
            self.iiwa_joint_qd += iiwa_qdd * self.dt
            self.allegro_joint_qd += allegro_qdd * self.dt
            self.iiwa_joint_q += self.iiwa_joint_qd * self.dt
            self.allegro_joint_q += self.allegro_joint_qd * self.dt
        else:
            raise ValueError(f"Invalid mode: {MODE}")

    def publish_joint_states(self):
        """Publish the current joint states from PyBullet."""
        iiwa_msg = JointState()
        iiwa_msg.header.stamp = rospy.Time.now()
        iiwa_msg.name = ["iiwa_joint_" + str(i) for i in range(NUM_ARM_JOINTS)]
        iiwa_msg.position = self.iiwa_joint_q.tolist()
        iiwa_msg.velocity = self.iiwa_joint_qd.tolist()
        self.iiwa_pub.publish(iiwa_msg)

        allegro_msg = JointState()
        allegro_msg.header.stamp = rospy.Time.now()
        allegro_msg.name = ["allegro_joint_" + str(i) for i in range(NUM_HAND_JOINTS)]
        allegro_msg.position = self.allegro_joint_q.tolist()
        allegro_msg.velocity = self.allegro_joint_qd.tolist()
        self.allegro_pub.publish(allegro_msg)

    def run(self):
        """Main loop to run the node, update simulation, and publish joint states."""
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Update the joint states
            self.update_joint_states()

            # Publish the current joint states to ROS
            self.publish_joint_states()

            # Sleep to maintain the loop rate
            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()
            rospy.loginfo(
                f"Max rate: {1 / (before_sleep_time - start_time).to_sec()} Hz ({(before_sleep_time - start_time).to_sec() * 1000}ms), Actual rate: {1 / (after_sleep_time - start_time).to_sec()} Hz"
            )


if __name__ == "__main__":
    try:
        # Create and run the FakeRobotNode
        node = FakeRobotNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
