#!/usr/bin/env python

import numpy as np
import rospy
from sensor_msgs.msg import JointState
import copy

# Constants
NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16
INTERPOLATION_DT = 1.0 / 120.0  # Time to reach the new target (seconds)


class FabricUpsampler:
    def __init__(self):
        rospy.init_node("fabric_upsampler")

        # ROS subscribers and publishers
        self.fabric_sub = rospy.Subscriber(
            "/fabric_state", JointState, self.fabric_callback
        )
        self.iiwa_cmd_pub = rospy.Publisher(
            "/iiwa/joint_cmd", JointState, queue_size=10
        )
        self.allegro_cmd_pub = rospy.Publisher(
            "/allegroHand_0/joint_cmd", JointState, queue_size=10
        )

        # ROS rate
        self.rate = rospy.Rate(200)  # 200 Hz

        # Last published commands
        self.last_published_iiwa_cmd_pos = np.zeros(NUM_ARM_JOINTS)
        self.last_published_allegro_cmd_pos = np.zeros(NUM_HAND_JOINTS)

        # Latest fabric command
        self.last_fabric_msg = None
        self.last_command_time = None

        # Wait for the first fabric message
        while not rospy.is_shutdown():
            if self.last_fabric_msg is not None and self.last_command_time is not None:
                rospy.loginfo("Got the first fabric message")
                break

            rospy.loginfo(
                f"Waiting: self.last_fabric_msg={self.last_fabric_msg}, self.last_command_time={self.last_command_time}"
            )
            rospy.sleep(0.1)

    def fabric_callback(self, msg: JointState):
        self.last_fabric_msg = msg
        self.last_command_time = rospy.Time.now()

    def run(self):
        assert self.last_fabric_msg is not None
        assert self.last_command_time is not None

        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            last_fabric_msg = copy.deepcopy(self.last_fabric_msg)
            last_command_time = copy.copy(self.last_command_time)

            # Calculate the elapsed time since the last command was received
            dt = (current_time - last_command_time).to_sec()

            # Compute the interpolation factor (alpha) clipped between 0 and 1
            alpha = np.clip(dt / INTERPOLATION_DT, a_min=0, a_max=1)

            # Extract the latest fabric joint states
            fabric_pos = np.array(last_fabric_msg.position)
            iiwa_target_pos = fabric_pos[:NUM_ARM_JOINTS]
            allegro_target_pos = fabric_pos[
                NUM_ARM_JOINTS : NUM_ARM_JOINTS + NUM_HAND_JOINTS
            ]

            fabric_vel = np.array(last_fabric_msg.velocity)
            iiwa_target_vel = fabric_vel[:NUM_ARM_JOINTS]
            _allegro_target_vel = fabric_vel[
                NUM_ARM_JOINTS : NUM_ARM_JOINTS + NUM_HAND_JOINTS
            ]

            fabric_name = last_fabric_msg.name
            iiwa_name = fabric_name[:NUM_ARM_JOINTS]
            allegro_name = fabric_name[
                NUM_ARM_JOINTS : NUM_ARM_JOINTS + NUM_HAND_JOINTS
            ]

            # Interpolate
            iiwa_interpolated_pos = (
                1 - alpha
            ) * self.last_published_iiwa_cmd_pos + alpha * iiwa_target_pos

            # Interpolate allegro joint
            allegro_interpolated_pos = (
                1 - alpha
            ) * self.last_published_allegro_cmd_pos + alpha * allegro_target_pos

            # Update the stored last commands
            self.last_published_iiwa_cmd_pos = iiwa_interpolated_pos
            self.last_published_allegro_cmd_pos = allegro_interpolated_pos

            timestamp = rospy.Time.now()

            # Create and publish iiwa command message
            iiwa_msg = JointState()
            iiwa_msg.header.stamp = timestamp
            iiwa_msg.name = iiwa_name
            iiwa_msg.position = iiwa_interpolated_pos.tolist()
            iiwa_msg.velocity = iiwa_target_vel.tolist()  # Keep uninterpolated velocity
            iiwa_msg.effort = []  # No effort information
            self.iiwa_cmd_pub.publish(iiwa_msg)

            # Create and publish allegro command message
            allegro_msg = JointState()
            allegro_msg.header.stamp = timestamp
            allegro_msg.name = allegro_name
            allegro_msg.position = allegro_interpolated_pos.tolist()
            allegro_msg.velocity = []  # Leave velocity as empty
            allegro_msg.effort = []  # Leave effort as empty
            self.allegro_cmd_pub.publish(allegro_msg)

            # Maintain the loop rate
            self.rate.sleep()


if __name__ == "__main__":
    try:
        fabric_upsampler = FabricUpsampler()
        fabric_upsampler.run()
    except rospy.ROSInterruptException:
        pass
