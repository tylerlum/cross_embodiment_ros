import rospy

from estimater import *
from datareader import *
from Utils import *

import argparse

class FoundationPoseROS:
    def __init__(self, args):
        set_logging_format()
        set_seed(0)
        
        self.video_dir = args.test_scene_dir
        self.est_refine_iter = args.est_refine_iter
        self.debug = args.debug
        self.debug_dir = args.debug_dir
        os.system(f'rm -rf {self.debug_dir}/* && mkdir -p {self.debug_dir}/track_vis {self.debug_dir}/ob_in_cam')

        rospy.init_node()
        self.pub = rospy.Publisher("/object_pose")
        self.sub = rospy.Subscriber("/rgbd_image")

        # FOUNDATION POSE
        self.object_mesh = trimesh.load(args.mesh_file)
        self.to_origin, extents = trimesh.bounds.oriented_bounds(self.object_mesh)
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.FPModel = FoundationPose(model_pts=self.object_mesh.vertices, model_normals=self.object_mesh.vertex_normals, mesh=self.object_mesh, scorer=self.scorer, refiner=self.refiner, debug_dir=self.debug_dir, debug=self.debug, glctx=self.glctx)
        logging.info("estimator initialization done")

        self.cam_K = np.loadtxt(f'{self.video_dir}/cam_K.txt').reshape(3,3)

        self.frame_count = 0     
        
    
    def publish(self):
        logging.info(f'i:{self.frame_count}')

        color = self.get_latest_color.copy()
        color = self.process_color(color)
        depth = self.get_latest_depth.copy()
        depth = self.process_depth(depth)

        if self.frame_count == 0: # Slow 1Hz mask generation + estimation
            mask = self.get_latest_mask().copy()
            mask = self.process_mask(mask, mask.shape[0], mask.shape[1])

            pose = self.FP.register(K=self.cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=self.est_refine_iter)
        
            if self.debug>=3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{self.debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, self.cam_K)
                valid = depth>=0.1
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply', pcd)
        else: # Fast 30Hz tracking
            pose = self.FP.track_one(rgb=color, depth=depth, K=self.cam_K, iteration=self..track_refine_iter)

            # USE SAM2 MASK TO GET SCORE AND IGNORE IF BELOW THRESHOLD?
            score_threshold = 0.2 # TO TUNE
            mask = self.get_latest_mask().copy()
            poses = torch.as_tensor(pose, device='cuda', dtype=torch.float)
            diameter = compute_mesh_diameter(model_pts=self.object_mesh.vertices, n_sample=10000)
            scores, _ = scorer.predict(mesh=mesh, rgb=color, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), glctx=self.glctx, mesh_diameter=diameter)
            print("SCORE: ", max(scores))
            if max(scores) < score_threshold: # Re-estimate or ignore?
                return np.zeros((4, 4))
                # mask = self.SAM2.predict(color, prompts=prompt, first=True).astype(bool)
                # mask = self.process_mask(mask, mask.shape[0], mask.shape[1])
                # pose = self.FP.register(K=self.cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
        
        # For logging purposes
        os.makedirs(f'{self.debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{self.debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

        if self.debug>=1:
            center_pose = pose@np.linalg.inv(self.to_origin)
            vis = draw_posed_3d_box(self.cam_K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=self.cam_K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[...,::-1])
            cv2.waitKey(1)

        if self.debug>=2:
            os.makedirs(f'{self.debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{self.debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

        self.frame_count += 1

        return pose # 4x4 matrix
    
    def process_color(self, color):
        color = imageio.imread(self.color_files[i])[...,:3]
        color = cv2.resize(color, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
        return color
    
    def process_depth(self, depth):
        depth = cv2.imread(self.color_files[i].replace('rgb','depth'),-1)/1e3
        depth = cv2.resize(depth, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
        depth[(depth<0.1) | (depth>=self.zfar)] = 0
        return depth

    def process_mask(self, mask, width, height):
        if len(mask.shape)==3:
        for c in range(3):
            if mask[...,c].sum()>0:
            mask = mask[...,c]
            break
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
        return mask
    

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/kiri_meshes/woodblock/3DModel.obj') # demo_data/snackbox_horizontal/mesh/textured.obj
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/blueblock/blueblock_occ_slide')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  

  
    

    

    

