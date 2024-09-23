#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R


class FakeObjectPose:
    def __init__(self):
        # Publisher for the object pose
        self.pose_pub = rospy.Publisher("/object_pose", Pose, queue_size=1)

        # Set a fixed transformation matrix T (4x4)
        self.T = np.array(
            [
                [1, 0, 0, 1],  # Rotation + Translation
                [0, 1, 0, 2],
                [0, 0, 1, 3],
                [0, 0, 0, 1],
            ]
        )

        # Publish rate of 60Hz
        self.rate = rospy.Rate(60)

    def publish_pose(self):
        # Extract translation and quaternion from the transformation matrix
        trans = self.T[:3, 3]
        quat_xyzw = R.from_matrix(self.T[:3, :3]).as_quat()

        # Create Pose message
        msg = Pose()
        msg.position.x, msg.position.y, msg.position.z = trans
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = (
            quat_xyzw
        )

        # Publish the pose message
        self.pose_pub.publish(msg)
        rospy.logdebug("Pose published to /object_pose")

    def run(self):
        while not rospy.is_shutdown():
            self.publish_pose()
            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("fake_object_pose")
    node = FakeObjectPose()
    rospy.loginfo("Publishing fixed object pose at 60Hz to /object_pose")
    node.run()
