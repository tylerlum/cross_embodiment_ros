#!/usr/bin/env python

import os
import rospy
import cv2
import numpy as np
import torch
import trimesh
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import Header
from estimater import *
from datareader import *
from Utils import *

class FoundationPoseROS:
    def __init__(self, args):
        set_logging_format()
        set_seed(0)
        
        # Variables for storing the latest images
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_mask = None
        self.frame_count = 0

        self.video_dir = args.test_scene_dir
        self.est_refine_iter = args.est_refine_iter
        self.track_refine_iter = args.track_refine_iter
        self.debug = args.debug
        self.debug_dir = args.debug_dir
        os.makedirs(self.debug_dir, exist_ok=True)
        os.makedirs(f'{self.debug_dir}/track_vis', exist_ok=True)
        os.makedirs(f'{self.debug_dir}/ob_in_cam', exist_ok=True)

        rospy.init_node('fp_node')
        self.bridge = CvBridge()

        # Load object mesh
        self.object_mesh = trimesh.load(args.mesh_file)
        self.to_origin, extents = trimesh.bounds.oriented_bounds(self.object_mesh)
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

        # FOUNDATION POSE initialization
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.FPModel = FoundationPose(model_pts=self.object_mesh.vertices, model_normals=self.object_mesh.vertex_normals, mesh=self.object_mesh,
                                      scorer=self.scorer, refiner=self.refiner, debug_dir=self.debug_dir, debug=self.debug, glctx=self.glctx)
        logging.info("Estimator initialization done")

        # Camera parameters
        self.cam_K = np.loadtxt(f'{self.video_dir}/cam_K.txt').reshape(3, 3)

        # Subscribers for RGB, depth, and mask images
        self.rgb_sub = rospy.Subscriber('/camera/color/image_raw', ROSImage, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', ROSImage, self.depth_callback)
        self.mask_sub = rospy.Subscriber('/sam2_mask', ROSImage, self.mask_callback)

        # Publisher for the object pose
        self.pose_pub = rospy.Publisher('/object_pose', ROSImage, queue_size=10)

    def rgb_callback(self, data):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # self.latest_rgb = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert RGB image: {e}")

    def depth_callback(self, data):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(data, "64FC1")
            # self.latest_depth = self.bridge.imgmsg_to_cv2(data, "8UC1")
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert depth image: {e}")

    def mask_callback(self, data):
        try:
            self.latest_mask = self.bridge.imgmsg_to_cv2(data, "mono8")
            self.process_images()
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert mask image: {e}")

    def process_images(self):
        if self.latest_rgb is None or self.latest_depth is None or self.latest_mask is None:
            rospy.logwarn("Missing one of the required images (RGB, depth, mask). Waiting...")
            return

        logging.info(f'Processing frame: {self.frame_count}')
        color = self.process_color(self.latest_rgb)
        depth = self.process_depth(self.latest_depth)
        mask = self.process_mask(self.latest_mask)
        rospy.loginfo(f"color: {color.shape}, {color.dtype}, {np.max(color)}, {np.min(color)}")
        rospy.loginfo(f"depth: {depth.shape}, {depth.dtype}, {np.max(depth)}, {np.min(depth)}, {np.mean(depth)}, {np.median(depth)}")
        rospy.loginfo(f"mask: {mask.shape}, {mask.dtype}, {np.max(mask)}, {np.min(mask)}")


        # Estimation and tracking
        if self.frame_count == 0:  # Slow 1Hz mask generation + estimation
            pose = self.FPModel.register(K=self.cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=self.est_refine_iter)
            logging.info("First frame estimation done")



            if self.debug>=3:
                m = self.object_mesh.copy()
                m.apply_transform(pose)
                m.export(f'{self.debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, self.cam_K)
                valid = depth>=0.001
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply', pcd)




        else:  # Fast 30Hz tracking
            pose = self.FPModel.track_one(rgb=color, depth=depth, K=self.cam_K, iteration=self.track_refine_iter)

        # Publish pose
        self.publish_pose(pose)

        if self.debug >= 1:
            center_pose = pose@np.linalg.inv(self.to_origin)
            vis = draw_posed_3d_box(self.cam_K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=self.cam_K, thickness=3, transparency=0, is_input_rgb=True)

            # vis = draw_posed_3d_box(self.cam_K, img=color, ob_in_cam=pose, bbox=self.bbox)
            cv2.imshow('Pose Visualization', vis)
            cv2.waitKey(1)

        self.frame_count += 1

    def process_color(self, color):
        rospy.loginfo(f"color.shape = {color.shape}")
        color = cv2.resize(color, (640, 480), interpolation=cv2.INTER_NEAREST)
        rospy.loginfo(f"AFTER color.shape = {color.shape}")
        return color

    def process_depth(self, depth):
        rospy.loginfo(f"depth.shape = {depth.shape}")
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST) 
        rospy.loginfo(f"AFTER depth.shape = {depth.shape}")
        depth = depth / 1000

        depth[depth < 0.1] = 0
        depth[depth > 4] = 0

        return depth

    def process_mask(self, mask):
        rospy.loginfo(f"mask.shape = {mask.shape}")
        mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST).astype(bool)
        rospy.loginfo(f"AFTER mask.shape = {mask.shape}")
        return mask

    def publish_pose(self, pose):
        # Convert the pose matrix into a ROS message
        pose_msg = ROSImage()
        pose_msg.header = Header(stamp=rospy.Time.now())
        pose_msg.encoding = 'rgb8'
        pose_msg.height = 1
        pose_msg.width = 4
        pose_msg.data = pose.flatten().tolist()

        # Publish the pose
        self.pose_pub.publish(pose_msg)
        rospy.loginfo("Pose published to /object_pose")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/kiri_meshes/snackbox/3DModel.obj')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/blueblock/blueblock_occ_slide')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=3)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    node = FoundationPoseROS(args)
    rospy.spin()
