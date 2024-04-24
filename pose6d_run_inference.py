# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import os
import argparse
# Add FoundationPose to sys.path so that the module inside can be imported
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory path
foundationdir = os.path.join(current_dir, 'FoundationPose')
# Add the parent directory to sys.path
sys.path.append(foundationdir)

from FoundationPose.estimater import *
from FoundationPose.datareader import *
from grounded_sam import GD_SAM
from realsense_camera import RealSenseCamera


class ObjectEstimator:
    def __init__(self, args):
        self.args = args
        self.debug = args.debug
        self.debug_dir = args.debug_dir
        self.test_scene_dir = args.test_scene_dir
        self.est_refine_iter = args.est_refine_iter
        self.track_refine_iter = args.track_refine_iter

        # Initialize directories
        os.system(f'rm -rf {self.debug_dir}/* && mkdir -p {self.debug_dir}/track_vis {self.debug_dir}/ob_in_cam')

        # Initialize predictors and context
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        logging.info("Estimator initialization done")

        # Initialize camera and reader
        self.update_camera()
        self.gd_sam = GD_SAM()

    
    def update_est_target(self, UniName, mesh, to_origin, bbox):
       self.target_name = UniName
       self.mesh = mesh
       self.to_origin = to_origin
       self.bbox = bbox
       self.ReMASK = True
       self.est = FoundationPose(
            model_pts=self.mesh.vertices, 
            model_normals=self.mesh.vertex_normals, 
            mesh=self.mesh, 
            scorer=self.scorer, 
            refiner=self.refiner, 
            debug_dir=self.debug_dir, 
            debug=self.debug, 
            glctx=self.glctx
        )

      
    def update_camera(self, camera=None):
      if camera is not None:
          self.camera = camera
          self.intrinsic_K = self.camera.color_intrin_mat
      else:
          self.camera = RealSenseCamera(color_align=True)
          self.intrinsic_K = self.camera.color_intrin_mat
       

    def run(self):
        HasMask = False
        for i in range(10000):
            color, depth = self.camera.get_rgbd_frame()
            image_source, image = self.gd_sam.img2tensor(color)
            if not HasMask:
                obj_mask = self.obj_detection(color)
                if obj_mask is not None:
                  cv2.imshow('mask', obj_mask)
                  obj_pose = self.obj_6dpose_est(color, depth, obj_mask)
                  HasMask = True
                else:
                    HasMask = False
                    logging.info(f'Cant detect {self.target_name} in frame {i}')
                    continue
            
            obj_pose = self.track_object(color, depth)
            self.visualize_pose(obj_pose, color)


    def obj_detection(self, color):
        image_source, image = self.gd_sam.img2tensor(color)
        masks, phrases, annotated_frame_sam = self.gd_sam.predict(image, image_source, self.target_name, gd_annotated=False, sam_annotated=False)
        if (masks[0] > 0).any():
            mask = masks[0].squeeze(0).cpu().numpy()
            return mask


    def obj_6dpose_est(self, color, depth, obj_mask):
        pose = self.est.register(K=self.intrinsic_K, rgb=color, depth=depth, ob_mask=obj_mask, iteration=self.est_refine_iter)
        return pose


    def track_object(self, color, depth):
        pose = self.est.track_one(rgb=color, depth=depth, K=self.intrinsic_K, iteration=self.track_refine_iter)
        return pose


    def save_pose(self, index, pose):
        os.makedirs(f'{self.reader.video_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{self.reader.video_dir}/ob_in_cam/{self.reader.id_strs[index]}.txt', pose.reshape(4,4))


    def visualize_pose(self, pose, color):
        center_pose = pose@np.linalg.inv(self.to_origin)
        vis = draw_posed_3d_box(self.intrinsic_K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=self.intrinsic_K, thickness=3, transparency=0, is_input_rgb=True)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imshow('1', vis)
        cv2.waitKey(1)
        if self.debug >= 2:
            os.makedirs(f'{self.reader.video_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{self.reader.video_dir}/track_vis/{self.reader.id_strs[index]}.png', vis)

    
    def save_mesh(self, pose, depth, color):
        m = self.mesh.copy()
        m.apply_transform(pose)
        m.export(f'{self.debug_dir}/model_tf.obj')
        
    
    def get_pcd_from_rgbd(self, color, depth):
        xyz_map = depth2xyzmap(depth, self.intrinsic_K)
        valid = depth >= 0.1
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply', pcd)


if __name__=='__main__':
  RealObjectsDict = ["Chinese Ceramic Bowl.", "White M Mug", "Blue Tape", "Blue Pepsi", 
                     "Transparent Wine Glass Cup.", "Transparent Water Glass Cup.", "Pink Spray.", 
                     "Yellow Mustard Bottle.", "Red Pepper Powder Container.", "Blue Dish Wash Bottle.", "Spam Can.", "Yellow Domino Sugar Box"]
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--target_name', type=str, default="Yellow Domino Sugar Box")
  parser.add_argument('--mesh_file', type=str, default=f'assets/group_objects/group4_real_objects_mesh_downsampled/133_domino_suger.ply')
  parser.add_argument('--video_file', type=str, default=f'{code_dir}/FoundationPose/demo_data/custom_test/red_cube.MOV')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/FoundationPose/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/FoundationPose/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  obj_detector = ObjectEstimator(args)

  mesh = trimesh.load(args.mesh_file)
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
  obj_detector.update_est_target(args.target_name, mesh, to_origin, bbox)
  obj_detector.run()

  # scorer = ScorePredictor()
  # refiner = PoseRefinePredictor()
  # glctx = dr.RasterizeCudaContext()
  # est = FoundationPose(
  #    model_pts=mesh.vertices, 
  #    model_normals=mesh.vertex_normals, 
  #    mesh=mesh, scorer=scorer, 
  #    refiner=refiner, 
  #    debug_dir=debug_dir, 
  #    debug=debug, 
  #    glctx=glctx
  # )
  # logging.info("estimator initialization done")

  # camera = RealSenseCamera(color_align=False)
  # reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
  # intrinsic_K = camera.color_intrin_mat
  # gd_sam = GD_SAM()
  # HasMask = False
  # for i in range(10000):
  #   logging.info(f'i:{i}')
  #   color, depth = camera.get_rgbd_frame()
  #   if not HasMask:
  #     image_source, image = gd_sam.img2tensor(color)
  #     masks, phrases, annotated_frame_sam = gd_sam.predict(image, image_source , "red cube", gd_annotated=True, sam_annotated=True)
  #     if (masks[0] > 0).any():
  #       HasMask = True
  #     else:
  #       continue

  #     mask = masks[0].squeeze(0).cpu().numpy()
  #     pose = est.register(K=intrinsic_K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

  #     if debug>=3:
  #       m = mesh.copy()
  #       m.apply_transform(pose)
  #       m.export(f'{debug_dir}/model_tf.obj')
  #       xyz_map = depth2xyzmap(depth, intrinsic_K)
  #       valid = depth>=0.1
  #       pcd = toOpen3dCloud(xyz_map[valid], color[valid])
  #       o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
  #   else:
  #     pose = est.track_one(rgb=color, depth=depth, K=intrinsic_K, iteration=args.track_refine_iter)

  #   os.makedirs(f'{reader.video_dir}/ob_in_cam', exist_ok=True)
  #   np.savetxt(f'{reader.video_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

  #   if debug>=1 and HasMask:
  #     center_pose = pose@np.linalg.inv(to_origin)
  #     vis = draw_posed_3d_box(intrinsic_K, img=color, ob_in_cam=center_pose, bbox=bbox)
  #     vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=intrinsic_K, thickness=3, transparency=0, is_input_rgb=True)
  #     vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
  #     cv2.imshow('1', vis)
  #     cv2.waitKey(1)

  #   if debug>=2:
  #     os.makedirs(f'{reader.video_dir}/track_vis', exist_ok=True)
  #     imageio.imwrite(f'{reader.video_dir}/track_vis/{reader.id_strs[i]}.png', vis)

