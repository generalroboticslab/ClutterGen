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
current_dir = os.path.dirname(os.path.realpath(__file__))
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

    
    def update_est_target(self, semantic_label, mesh, to_origin, bbox, UniName=None):
       self.target_name = UniName
       self.semantic_label = semantic_label
       self.mesh = mesh
       self.to_origin = to_origin
       self.bbox = bbox
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
       self.prev_pose = None

      
    def update_camera(self, camera=None):
      if camera is not None:
          self.camera = camera
      else:
          self.camera = RealSenseCamera(color_align=True)
      self.intrinsic_K = self.camera.color_intrin_mat
      self.img_h, self.img_w = self.camera.img_h, self.camera.img_w
       

    def est_obj_pose6d(self, strict=False, visualization=False):
        color, depth = self.camera.get_rgbd_frame()
        image_source, image = self.gd_sam.img2tensor(color)
        obj_mask = self.obj_detection(color)
        if obj_mask is not None:
            if not strict and self.prev_pose is not None:
                obj_pose = self.track_object(color, depth)
            else:
                obj_pose = self.obj_6dpose_est(color, depth, obj_mask)
            if visualization:
                self.visualize_pose(obj_pose, color)
            self.prev_pose = obj_pose
            return color, obj_mask, obj_pose
        else:
            self.prev_pose = None
            logging.warning(f'Did not detect {self.semantic_label} in current frame or the object is out of track')
            return color, None, None


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
                    logging.info(f'Cant detect {self.semantic_label} in frame {i}')
                    continue
            
            obj_pose = self.track_object(color, depth)
            self.visualize_pose(obj_pose, color)


    def obj_detection(self, color):
        image_source, image = self.gd_sam.img2tensor(color)
        masks, phrases, annotated_frame_sam = self.gd_sam.predict(image, image_source, self.semantic_label, gd_annotated=False, sam_annotated=False)
        if (masks[0] > 0).any():
            mask = masks[0].squeeze(0).cpu().numpy()
            return mask


    def obj_6dpose_est(self, color, depth, obj_mask):
        pose = self.est.register(K=self.intrinsic_K, rgb=color, depth=depth, ob_mask=obj_mask, iteration=self.est_refine_iter)
        pose = pose@np.linalg.inv(self.to_origin) # Why do we need to invert the to_origin matrix? Seems the to_origin has some transformation that needs to be undone
        return pose


    def track_object(self, color, depth):
        pose = self.est.track_one(rgb=color, depth=depth, K=self.intrinsic_K, iteration=self.track_refine_iter)
        pose = pose@np.linalg.inv(self.to_origin) # Why do we need to invert the to_origin matrix? Seems the to_origin has some transformation that needs to be undone
        return pose


    def save_pose(self, index, pose):
        os.makedirs(f'{self.reader.video_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{self.reader.video_dir}/ob_in_cam/{self.reader.id_strs[index]}.txt', pose.reshape(4,4))


    def visualize_pose(self, pose, color, index=None):
        vis = draw_posed_3d_box(self.intrinsic_K, img=color, ob_in_cam=pose, bbox=self.bbox)
        vis = draw_xyz_axis(color, ob_in_cam=pose, scale=0.1, K=self.intrinsic_K, thickness=3, transparency=0, is_input_rgb=True)
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


    def draw_xyz_axis(self, color, ob_in_cam, scale=0.1, thickness=3, transparency=0, is_input_rgb=True):
        vis = draw_xyz_axis(color, ob_in_cam=ob_in_cam, scale=scale, K=self.intrinsic_K, thickness=thickness, transparency=transparency, is_input_rgb=is_input_rgb)
        return vis
    

    def draw_posed_3d_box(self, color, ob_in_cam):
        vis = draw_posed_3d_box(self.intrinsic_K, img=color, ob_in_cam=ob_in_cam, bbox=self.bbox)
        return vis
    

    def get_raw_rgbd_frame(self):
        return self.camera.get_rgbd_frame()


if __name__=='__main__':
  RealObjectsDict = ["Chinese Ceramic Bowl.", "White M Mug", "Blue Tape", "Blue Pepsi", 
                     "Transparent Wine Glass Cup.", "Transparent Water Glass Cup.", "Pink Spray.", 
                     "Yellow Mustard Bottle.", "Red Pepper Powder Container.", "Blue Dish Wash Bottle.", "Spam Can.", "Yellow Domino Sugar Box"]
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--semantic_label', type=str, default="Blue Tape")
  parser.add_argument('--mesh_file', type=str, default=f'assets/group_objects/group4_real_objects/44_paper_tape/0/textured_objs/textured.obj')
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
  obj_detector.update_est_target(args.semantic_label, mesh, to_origin, bbox)
  for i in range(100000):
      obj_detector.est_obj_pose6d(visualization=True)
  obj_detector.run()