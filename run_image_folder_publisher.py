import rospy
from sensor_msgs.msg import Image
from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2
from cv_bridge import CvBridge, CvBridgeError
import os

class RGBDImageSaver:
    def __init__(self):
        self.frame_count = 0
        self.bridge = CvBridge()
        
        # Create directories if they don't exist
        if not os.path.exists('color'):
            os.makedirs('color')
        if not os.path.exists('depth'):
            os.makedirs('depth')
        
        # Initialize ROS node
        rospy.init_node('rgbd_image_saver_node', anonymous=True)
        
        # Subscribers for the color and depth images
        self.color_sub = Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = Subscriber('/camera/depth/image_rect_raw', Image)
        
        # Synchronize the subscriptions
        self.ts = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)
        
        # Timer to save images at 10 Hz
        self.timer = rospy.Timer(rospy.Duration(0.1), self.save_images)

        rospy.spin()

    def callback(self, color_image, depth_image):
        try:
            # Convert ROS Images to OpenCV images
            self.cv_color_image = self.bridge.imgmsg_to_cv2(color_image, "bgr8")
            self.cv_depth_image = self.bridge.imgmsg_to_cv2(depth_image, "16UC1")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def save_images(self, event):
        # Pad frame count with leading zeros to 5 digits
        frame_str = str(self.frame_count).zfill(5)
        
        # Save color image
        color_filename = f'color/{frame_str}.png'
        cv2.imwrite(color_filename, self.cv_color_image)
        
        # Save depth image
        depth_filename = f'depth/{frame_str}.png'
        cv2.imwrite(depth_filename, self.cv_depth_image)
        
        # Increment frame count
        self.frame_count += 1

def main():
    RGBDImageSaver()

if __name__ == "__main__":
    main()

