#!/usr/bin/env python

import os

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError

# Assuming your SAM2Model is in a file named sam2_model
from sam2rt_model import SAM2Model
from sensor_msgs.msg import Image as ROSImage


class SAM2RosNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("sam2_ros_node", anonymous=True)

        # Create an instance of SAM2Model
        self.sam2_model = SAM2Model()

        # Initialize the CvBridge to convert between ROS images and OpenCV images
        self.bridge = CvBridge()

        # Subscribe to the camera topic
        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw", ROSImage, self.callback
        )

        # Output directory for saving results
        self.output_dir = "/tmp/sam2_output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        rospy.loginfo("SAM2 ROS Node initialized and waiting for images...")
        self.frame_count = 0

    def callback(self, data):
        try:
            # Convert the ROS image message to a format OpenCV can work with
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            rospy.loginfo("Image received from /camera/color/image_raw")

            # Predict the mask using SAM2Model
            mask = self.sam2_model.predict(
                image, first=(self.frame_count == 0), viz=False
            )
            self.frame_count += 1

            # Save the original image and the mask
            timestamp = rospy.Time.now().to_nsec()
            original_image_path = os.path.join(
                self.output_dir, f"original_{timestamp}.png"
            )
            mask_image_path = os.path.join(self.output_dir, f"mask_{timestamp}.png")

            # Save original image
            cv2.imwrite(original_image_path, image)
            rospy.loginfo(f"Original image saved to {original_image_path}")

            # Save mask image
            rospy.loginfo(f"{image.shape}, {type(image)}, {image.dtype}")
            rospy.loginfo(f"{mask.shape}, {type(mask)}, {mask.dtype}")
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            cv2.imwrite(mask_image_path, mask_rgb)
            rospy.loginfo(f"Predicted mask saved to {mask_image_path}")

            # Optionally, visualize in a separate window (OpenCV visualization)
            cv2.imshow("Original Image", image)
            cv2.imshow("Predicted Mask", mask)
            cv2.waitKey(1)  # Necessary for OpenCV windows to update

        except CvBridgeError as e:
            rospy.logerr(f"Could not convert ROS Image to OpenCV image: {e}")

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = SAM2RosNode()
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down SAM2 ROS Node.")
