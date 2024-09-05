#!/usr/bin/env python

import os
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import Header

# Assuming your SAM2Model is in a file named sam2_model
from sam2_model import SAM2Model

class SAM2RosNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('sam2_ros_node', anonymous=True)
        
        # Create an instance of SAM2Model
        self.sam2_model = SAM2Model()
        
        # Initialize the CvBridge to convert between ROS images and OpenCV images
        self.bridge = CvBridge()
        
        # Subscribe to the camera topic
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', ROSImage, self.callback)
        
        # Publisher for the predicted mask
        self.mask_pub = rospy.Publisher('/sam2_mask', ROSImage, queue_size=10)
        
        # Frame count for distinguishing first frame
        self.frame_count = 0
        
        rospy.loginfo("SAM2 ROS Node initialized and waiting for images...")

    def callback(self, data):
        try:
            # Convert the ROS image message to a format OpenCV can work with
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            rospy.loginfo("Image received from /camera/color/image_raw")
            
            # Predict the mask using SAM2Model
            mask = self.sam2_model.predict(image, first=(self.frame_count == 0), viz=False)
            self.frame_count += 1
            
            # Convert mask to a format that can be published
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            # HACK
            # mask_rgb[440-10:440+10, 480-10:480+10] = [255, 0, 0]

            # Convert OpenCV image (mask) to ROS Image message
            mask_msg = self.bridge.cv2_to_imgmsg(mask_rgb, encoding="rgb8")
            mask_msg.header = Header(stamp=rospy.Time.now())

            # Publish the mask to the /sam2_mask topic
            self.mask_pub.publish(mask_msg)
            rospy.loginfo("Predicted mask published to /sam2_mask")
            
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert ROS Image to OpenCV image: {e}")
        except Exception as e:
            rospy.logerr(f"Error during prediction or publishing: {e}")

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = SAM2RosNode()
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down SAM2 ROS Node.")
