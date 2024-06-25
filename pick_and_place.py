# Full pipeline for the pick and place task using frankapanda and realsense camera

import os
import sys
import time
import logging
import numpy as np
import cv2
import torch
import trimesh
import pybullet as p
import pybullet_utils_cust as pu
import xml.etree.ElementTree as ET
import argparse
from distutils.util import strtobool
from copy import deepcopy
from typing import List

from utils import read_json, natural_keys, set_logging_format, combine_images, se3_transform_pc, pc_random_downsample, combine_envs_float_info2list
from pose6d_run_inference import ObjectEstimator
from robot_related.franka_controller_client import FrankaPandaCtrl
from pybullet_visualizer import PybulletVisualizer

# Stable Placement related imports
from StablePlacement.sp_model import get_sp_model
from StablePlacement.sp_utils import visualize_pred_pose

# Scene Generator related imports
from RoboSensai_bullet import RoboSensaiBullet
from PPO.PPO_continuous_sg import *

import open3d as o3d
import pyrender
import copy
from pynput import keyboard

# Set up logging
# logging.basicConfig(level=logging.INFO)

def parse_args():
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description='Train Tactile Pushing Experiment')
    
    # Env hyper parameters
    parser.add_argument('--task', type=str, default='rearrangement', help="Task Name")
    parser.add_argument('--object_pool_name', type=str, default='Union', help="Target object to be grasped. Ex: cube")
    parser.add_argument('--asset_root', type=str, default='assets', help="folder path that stores all urdf files")
    parser.add_argument('--object_pool_folder', type=str, default='group_objects/group4_real_objects', help="folder path that stores all urdf files")
    parser.add_argument('--object_textured_mesh_folder', type=str, default='assets/group_objects/group4_real_objects_mesh_downsampled', help="folder path that stores all urdf files")

    parser.add_argument('--pybullet_debug', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--realtime', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--quiet', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--panda_urdf_path', type=str, default=f'{code_dir}/robot_related/franka_description/robots/franka_panda.urdf')
    parser.add_argument('--camera_extrinsics_path', type=str, default=f'{code_dir}/assets/rs_camera/camera_extrinsics4.txt')
    parser.add_argument('--panda2crop_center_path', type=str, default=f'{code_dir}/assets/rs_camera/panda2crop_center_pos.txt')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/FoundationPose/demo_data/mustard0')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/FoundationPose/debug')

    # Stable Placement Model Args
    parser.add_argument('--use_sp_model', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--sp_result_dir', type=str, default='StablePlacement/SP_Result')
    parser.add_argument('--sp_checkpoint', type=str, default='StablePlacement_05-09_18:02_Deterministic_PNPlusGroupAll_WeightedLoss_EntCoef0.0_weiDecay0.0')
    parser.add_argument('--sp_index_episode', type=str, default="1000")

    # Scene Generator Model Args
    parser.add_argument('--use_sg_model', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--sg_result_dir', type=str, default='train_res/Union')
    parser.add_argument('--sg_checkpoint', type=str, default='Union_2024_04_23_213414_Sync_Beta_group4_real_objects_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool12_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_entropy0.01_seed123456')
    parser.add_argument('--sg_index_episode', type=str, default="best")
    parser.add_argument('--sg_rendering', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)

    args = parser.parse_args()
    
    # Stable Placement Model Args
    args.sp_args = argparse.Namespace()
    sp_json_path = os.path.join(args.sp_result_dir, args.sp_checkpoint, "Json", f"{args.sp_checkpoint}.json")
    sp_train_args = read_json(sp_json_path)
    args.sp_args.__dict__.update(sp_train_args)
    args.sp_args.sp_checkpoint_path = os.path.join(args.sp_result_dir, args.sp_checkpoint, "Checkpoint", f"epoch_{args.sp_index_episode}")+".pth"

    # Scene Generator Model Args
    args.sg_args = argparse.Namespace()
    sg_json_path = os.path.join(args.sg_result_dir, "Json", f"{args.sg_checkpoint}.json")
    sg_train_args = read_json(sg_json_path)
    args.sg_args.__dict__.update(sg_train_args)
    args.sg_args.sg_checkpoint_path = os.path.join(args.sg_result_dir, "checkpoints", args.sg_checkpoint, args.sg_checkpoint + '_' + args.sg_index_episode)
    args.sg_args.num_envs = 1 #Only support single environment for now!
    args.sg_args.rendering = args.sg_rendering
    args.sg_args.eval_result = True
    args.sg_args.strict_checking = True

    return args



class PickAndPlace:
    RealObjectsDict = ["Chinese Ceramic Bowl.", "White M Mug.", "Blue Tape.", "Blue Pepsi.", 
                       "Transparent Wine Glass Cup.", "Transparent Water Glass Cup.", "Pink Spray.", 
                       "Yellow Mustard Bottle.", "Red Pepper Powder Container.", "Blue Dish Wash Bottle.", "Spam Can.", "Yellow Domino Sugar Box."]
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_logging_format(logging.WARNING)

        # Misc variable
        self.crop_size = np.array([0.25, 0.25, 0.2])
        self.gsp_miti_z_offset = 0.01 # 2cm offset for both grasp and placement; pred_grasp2real_grasp / pred_place2real_place
        # 0.02~0.025; 0.035
        self.rls_miti_z_offset = 0.025 # 2cm offset for both grasp and placement; pred_grasp2real_grasp / pred_place2real_place
        # small lift
        self.smaller_lift_name = ["34_meyerscleanday", "23_dawn", "125_wine_glass", "133_domino_suger"]

        self.pandaeef2handbase = self.get_pandaeef2handbase(args.panda_urdf_path)
        self.pandahandbase2eef = pu.invert_transform(self.pandaeef2handbase)
        self.panda2camera = self.get_pandabase2camera(args.camera_extrinsics_path)
        self.pandabase2crop_center = self.get_pandabase2crop_center(args.panda2crop_center_path)
        self.update_objects_database()
        self.obj_est = ObjectEstimator(args)
        self.fp_ctrl = FrankaPandaCtrl()
        self.fp_ctrl.reset() # Reset the robot to the home position

        if self.args.use_sp_model:
            self.sp_model = get_sp_model(args.sp_args, 
                                         checkpoint_path=args.sp_args.sp_checkpoint_path, 
                                         evaluate=True, device=self.device)
            
        if self.args.use_sg_model:
            self.sg_env = RoboSensaiBullet(args.sg_args)
            self.sg_agent = Agent(self.sg_env).to(self.device)
            self.sg_agent.load_checkpoint(args.sg_args.sg_checkpoint_path, evaluate=True, map_location="cuda:0")

        if self.args.pybullet_debug:
            self.pb_visualizer = PybulletVisualizer(self.obj_uni_names_dataset, rendering=True)

        self.retrieve_camera_intrinsics()
        self.clear_registeration()


    
    def update_objects_database(self):
        # Right now we only have two stages, object can not appear in both stages! We need to figure out how to deal with this problem
        self.obj_dataset_folder = os.path.join(self.args.asset_root, self.args.object_pool_folder)
        self.obj_textured_mesh_folder = os.path.join(self.args.asset_root, self.args.object_textured_mesh_folder)
        # 0: "Table", "Bookcase", "Dishwasher", "Microwave", all storage furniture
        self.obj_uni_names_dataset = {}
        obj_categories = sorted(os.listdir(self.obj_dataset_folder), key=natural_keys)
        for cate in obj_categories:
            obj_folder = os.path.join(self.obj_dataset_folder, cate)
            obj_indexes = sorted(os.listdir(obj_folder), key=natural_keys)
            for idx in obj_indexes:
                obj_uni_name = f"{cate}_{idx}"
                print(f"Loading object: {obj_uni_name}")
                obj_urdf_path = f"{self.obj_dataset_folder}/{cate}/{idx}/mobility.urdf"
                obj_label_path = f"{self.obj_dataset_folder}/{cate}/{idx}/label.json"
                obj_mesh_path = self.get_collision_mesh_path_by_urdf(obj_urdf_path)
                assert os.path.exists(obj_urdf_path), f"Object {obj_uni_name} does not exist! Given path: {obj_urdf_path}"
                assert os.path.exists(obj_label_path), f"Object {obj_uni_name} does not exist! Given path: {obj_label_path}"
                assert os.path.exists(obj_mesh_path), f"Object {obj_uni_name} does not exist! Given path: {obj_mesh_path}"

                obj_label = read_json(obj_label_path)
                assert "semantic_label" in obj_label, f"Object {obj_uni_name} does not have 'semantic_label' in label.json! Given path: {obj_label_path}"
                assert "Center2GripperBase_pose" in obj_label, f"Object {obj_uni_name} does not have 'Center2GripperBase_pose' in label.json! Given path: {obj_label_path}"
                # Retrieve the mesh, extents and the bounding box of the object
                mesh = trimesh.load(obj_mesh_path)
                extents = mesh.bounding_box.extents
                base2bboxcenter = mesh.bounding_box.centroid
                to_origin = trimesh.transformations.translation_matrix(-base2bboxcenter)
                bbox = np.stack([-extents/2, extents/2], axis=0)
                mesh_height = extents[2]
                panda2target_center_z = self.pandabase2crop_center[0][2] + mesh_height/2 + self.rls_miti_z_offset # Manually Fix the height of the object on the table
                if "bowl" in obj_uni_name:
                    panda2target_center_z += 0.025
                
                # Foundation pose can only support trimesh loaded mesh; However our RoboSensai is using open3d sampled pc to compute the origin
                # The best way is to use mesh to compute the origin, since the gap between these methods is very small we use both and omit the gap for now.
                o3d_mesh = o3d.io.read_triangle_mesh(obj_mesh_path) 
                obj_pc = pu.sample_pc_from_mesh(mesh_path=obj_mesh_path, num_points=1024)
                bboxcenter2base = pu.invert_transform([base2bboxcenter[:3], [0., 0., 0., 1.]])
                centered_obj_pc = se3_transform_pc(*bboxcenter2base, obj_pc)
                
                self.obj_uni_names_dataset.update(
                    {obj_uni_name: 
                                {
                                    "urdf_path": obj_urdf_path, 
                                    "label": obj_label,
                                    "mesh": mesh,
                                    "o3d_mesh": o3d_mesh,
                                    "to_origin": to_origin,
                                    "bbox": bbox,
                                    "base2bboxcenter": base2bboxcenter,
                                    "centered_obj_pc": centered_obj_pc,
                                    "panda2target_center_z": panda2target_center_z
                                }
                    })
    

    def get_collision_mesh_path_by_urdf(self, urdf_path):
        # Parse the URDF file
        assert os.path.exists(urdf_path), f"URDF file does not exist! Given path: {urdf_path}"
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Iterate through each link to find the collision mesh path
        for link in root.findall('link'):
            collision = link.find('collision')
            if collision:
                geometry = collision.find('geometry')
                mesh = geometry.find('mesh') if geometry else None
                if mesh is not None and 'filename' in mesh.attrib:
                    mesh_path_rela = mesh.get('filename')
                    mesh_path = os.path.join(os.path.dirname(urdf_path), mesh_path_rela)
                    return mesh_path
                else:
                    print(f"Link Name: {link.get('name')} has no mesh in collision.")
                
    
    def get_pandaeef2handbase(self, panda_urdf_path, jointname="panda_hand_joint"):
        # Parse the URDF file
        # EEf is the end effector link (link8), hand is the panda_hand link (we compute their relative pose which is the joint pose)
        assert os.path.exists(panda_urdf_path), f"URDF file does not exist! Given path: {panda_urdf_path}"
        tree = ET.parse(panda_urdf_path)
        root = tree.getroot()
        # Find the specific joint and read its RPY and XYZ values
        for joint in root.findall('joint'):
            if joint.attrib['name'] == jointname:
                origin = joint.find('origin')
                rpy = origin.attrib.get('rpy', '0 0 0').split()
                xyz = origin.attrib.get('xyz', '0 0 0').split()

                # Convert RPY strings to floats
                roll, pitch, yaw = map(float, rpy)
                x, y, z = map(float, xyz)
                quat = pu.get_quaternion_from_euler([roll, pitch, yaw])
                return [x, y, z], quat
        raise ValueError(f"Joint {jointname} not found in the URDF file.")

    
    def get_pandabase2camera(self, camera_extrinsics_path):
        # .txt file containing the extrinsics of the camera
        assert os.path.exists(camera_extrinsics_path), f"Camera extrinsics file does not exist! Given path: {camera_extrinsics_path}"
        with open(camera_extrinsics_path, 'r') as f:
            lines = f.readlines()
            # Extract the rotation and translation values
            panda2camera = np.array([list(map(float, line.split())) for line in lines[:]]).reshape(4, 4)
        panda2camera = pu.matrix2pose2d(panda2camera)
        return panda2camera
    

    def get_pandabase2crop_center(self, panda2crop_center_path):
        # .txt file containing the [x, y, z] translation values
        if not os.path.exists(panda2crop_center_path):
            logging.warning(f"Pandabase2CropCenter file does not exist! Given path: {panda2crop_center_path}; Please clean the table and record the crop center first!")
            return None
        with open(panda2crop_center_path, 'r') as f:
            lines = f.readlines()
            # Extract the translation values
            pandabase2crop_center_pos = np.array([list(map(float, line.split())) for line in lines[:]]).reshape(3, ).tolist()
        pandabase2crop_center = [pandabase2crop_center_pos, [0., 0., 0., 1.]]
        return pandabase2crop_center


    def register_target_object(self, targetUniName):
        self.cur_targetUniName = targetUniName
        # Get the target object's URDF path
        target_obj_db = self.obj_uni_names_dataset[targetUniName]
        target_semantic_label = target_obj_db["label"]["semantic_label"]
        target_mesh = target_obj_db["mesh"]
        target_to_origin = target_obj_db["to_origin"]
        target_bbox = target_obj_db["bbox"]
        self.panda2target_center_z = target_obj_db["panda2target_center_z"]
        self.obj_est.update_est_target(target_semantic_label, target_mesh, target_to_origin, target_bbox, UniName=targetUniName)
        # Compute the grasp pose (0 and 180 degrees) because the gripper is symmetric
        target_center2handbase = self.obj_uni_names_dataset[targetUniName]["label"]["Center2GripperBase_pose"] # [x, y, z, roll, pitch, yaw]
        self.target_center2handbase = [target_center2handbase[0], pu.get_quaternion_from_euler(target_center2handbase[1])]
        self.handbase2pre_handbase = [[0., 0., -0.1], [0., 0., 0., 1.]] # z-axis is pointing to the center of the frame; should be negative
        self.target_center2pre_handbase = pu.multiply_multi_transforms(self.target_center2handbase, self.handbase2pre_handbase)
        handbase2handbase_reverse = [[0., 0., 0.], pu.getQuaternionFromAxisAngle([0, 0, 1], np.pi)]
        self.target_center2handbase_reverse = pu.multiply_multi_transforms(self.target_center2handbase, handbase2handbase_reverse)
        self.target_center2pre_handbase_reverse = pu.multiply_multi_transforms(self.target_center2pre_handbase, handbase2handbase_reverse)


    def retrieve_camera_intrinsics(self):
        # Get the camera intrinsics
        color_intrin_mat = self.obj_est.intrinsic_K
        fx, fy, cx, cy = color_intrin_mat[0, 0], color_intrin_mat[1, 1], color_intrin_mat[0, 2], color_intrin_mat[1, 2]
        width, height = self.obj_est.img_w, self.obj_est.img_h
        # Define camera intrinsic parameters for mesh rendering
        self.o3d_cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, color_intrin_mat)
        

    def clear_registeration(self):
        self.cur_targetUniName = None
        self.target_center2handbase = None
        self.camera_axes_ids = None
        self.grasp_pose_ids = None
        self.grasp_pose_reverse_ids = None


    def visualize_all_transforms(self, transforms_lst):
        # Define your transformations as lists [pos, quaternion, label]

        # Create an Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Create a coordinate frame and text label for each transformation
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        vis.add_geometry(origin_frame)


        for transform in transforms_lst:
            pos, quat, label = transform
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            rot_mat = pu.pose2d2matrix([[0, 0, 0.], quat])[:3, :3]
            coordinate_frame = coordinate_frame.rotate(rot_mat)
            coordinate_frame = coordinate_frame.translate(pos)
            # Convert quaternion [x, y, z, w] to rotation matrix
            # Create a coordinate frame with the origin at 'pos' and the axes aligned with the rotation matrix
            # Add the coordinate frame to the visualizer
            vis.add_geometry(coordinate_frame)

        # Create camera and set its parameters (must do this after all geometries are added)
        # Access view control and calculate front vector
        # draw camera coordinate frame
        view_ctrl = vis.get_view_control()
        view_ctrl.camera_local_translate(-0.5, 0, 0.5)
        print(view_ctrl.get_field_of_view())
        # view_ctrl.change_field_of_view(step=-15)

        # Run the visualizer
        vis.run()
        vis.destroy_window()
    
    
    def detect_target_object(self, targetUniName, strict=True, draw_bbox=True, draw_xyz=True, fix_target_z=True):
        if self.cur_targetUniName is None or self.cur_targetUniName != targetUniName:
            self.register_target_object(targetUniName)
            logging.info(f"Target object {targetUniName} registered.")

        color_img, target_mask, camera2target_center_mat = self.obj_est.est_obj_pose6d(strict=strict)

        if camera2target_center_mat is not None:
            # Concert the Homogeneous matrix to the pose format
            camera2target_center = pu.matrix2pose2d(camera2target_center_mat)
            # Visualize the grasp pose at the eef link
            panda2target_center = pu.multiply_multi_transforms(self.panda2camera, camera2target_center)

            if "paper_tape" in targetUniName:
                # Manually fix blue tape's orientation; upward z-axis
                panda2target_center = [list(panda2target_center[0]), [0, 0, 0, 1.]]
                camera2target_center = pu.multiply_multi_transforms(pu.invert_transform(self.panda2camera), panda2target_center)
                camera2target_center_mat = pu.pose2d2matrix(camera2target_center)
            
            if draw_bbox:
                color_img = self.obj_est.draw_posed_3d_box(color_img, camera2target_center_mat)
            if draw_xyz:
                color_img = self.obj_est.draw_xyz_axis(color_img, camera2target_center_mat)

            if self.args.pybullet_debug:
                panda_joints_state = self.fp_ctrl.get_joint_state()
                self.pb_visualizer.set_panda_joints(panda_joints_state)
                self.pb_visualizer.set_panda_gripper([0.04, 0.04]) # open the gripper
                self.pb_visualizer.set_obj_pose(targetUniName, panda2target_center)
                self.camera_axes_ids = self.pb_visualizer.draw_camera_axis(self.panda2camera, old_axis_ids=self.camera_axes_ids)

            if fix_target_z:
                # Manually fix the height of the object on the table
                panda2target_center[0][2] += self.gsp_miti_z_offset

            return panda2target_center, color_img
        return None, color_img
    

    def compute_target_grasp_pose(self, panda2target_center, targetUniName=None):
        if targetUniName is not None:
            assert self.cur_targetUniName == targetUniName, "The targetUniName is not match. Please rerun detect_target_object first!"
        targetUniName = self.cur_targetUniName
        target_center2handbase = self.target_center2handbase
        target_center2pre_handbase = self.target_center2pre_handbase
        target_center2handbase_reverse = self.target_center2handbase_reverse
        target_center2pre_handbase_reverse = self.target_center2pre_handbase_reverse

        # Compute the grasp and pre-grasp pose in the robot base frame
        pandabase2handbase_pre_grasp = pu.multiply_multi_transforms(panda2target_center, target_center2pre_handbase)
        pandabase2handbase_grasp = pu.multiply_multi_transforms(panda2target_center, target_center2handbase)
        pandabase2handbase_pre_grasp_reverse = pu.multiply_multi_transforms(panda2target_center, target_center2pre_handbase_reverse)
        pandabase2handbase_grasp_reverse = pu.multiply_multi_transforms(panda2target_center, target_center2handbase_reverse)

        if self.args.pybullet_debug:
            self.grasp_pose_ids = self.pb_visualizer.draw_camera_axis(pandabase2handbase_grasp, old_axis_ids=self.grasp_pose_ids)
            self.grasp_pose_reverse_ids = self.pb_visualizer.draw_camera_axis(pandabase2handbase_grasp_reverse, old_axis_ids=self.grasp_pose_reverse_ids)
        
        return pandabase2handbase_pre_grasp, pandabase2handbase_grasp, pandabase2handbase_pre_grasp_reverse, pandabase2handbase_grasp_reverse
    

    def compute_target_release_pose(self, panda2target_center, new_target_center2handbase, targetUniName=None):
        # Release pose does not have reverse!
        if targetUniName is not None:
            assert self.cur_targetUniName == targetUniName, "The targetUniName is not match. Please rerun detect_target_object first!"
        targetUniName = self.cur_targetUniName
        new_target_center2pre_handbase = pu.multiply_multi_transforms(new_target_center2handbase, self.handbase2pre_handbase)

        # Compute the release and pre-release pose in the robot base frame
        pandabase2handbase_release = pu.multiply_multi_transforms(panda2target_center, new_target_center2handbase)
        pandabase2handbase_pre_release = pu.multiply_multi_transforms(panda2target_center, new_target_center2pre_handbase)
        pandabase2handbase_pre_pre_release = [list(pandabase2handbase_release[0]), list(pandabase2handbase_release[1])]
        pandabase2handbase_pre_pre_release[0][2] += 0.2 # 20cm above the target location

        return pandabase2handbase_release, pandabase2handbase_pre_release, pandabase2handbase_pre_pre_release


    def update_target2hand_pose(self, targetUniName, strict=True):
        if self.cur_targetUniName is None or self.cur_targetUniName != targetUniName:
            self.register_target_object(targetUniName)
            logging.info(f"Target object {targetUniName} registered.")

        color_img, target_mask, camera2target_center_mat = self.obj_est.est_obj_pose6d(strict=strict)

        if camera2target_center_mat is not None:
            camera2target_center = pu.matrix2pose2d(camera2target_center_mat)
            # Visualize the grasp pose at the eef link
            panda2target_center = pu.multiply_multi_transforms(self.panda2camera, camera2target_center)
            pandabase2eef = pu.split_7d(self.fp_ctrl.get_eef_pose())
            new_target_center2handbase = pu.multiply_multi_transforms(pu.invert_transform(panda2target_center), pandabase2eef)
            # z-axis is pointing to the center of the frame; should be negative
            new_target_center2pre_handbase = pu.multiply_multi_transforms(new_target_center2handbase, self.handbase2pre_handbase)
            return new_target_center2handbase, new_target_center2pre_handbase
        return None, None
    

    def update_target2hand_naive(self, panda2handbase_exec, panda2target_center):
        # This is a naive approach to update the target2handbase pose
        new_target_center2handbase = pu.multiply_multi_transforms(pu.invert_transform(panda2target_center), panda2handbase_exec)
        new_target_center2pre_handbase = pu.multiply_multi_transforms(new_target_center2handbase, self.handbase2pre_handbase)
        return new_target_center2handbase, new_target_center2pre_handbase


    def gripper_approach(self, panda2handbase_2dlist: List, maximum_attempts=3, path_len_threshold=550):
        """
        panda2handbase_2dlist: [pose2d1, pose2d2, ...]
        input is panda2handbase, but we need to convert this to panda2eef for the robot control
        """
        panda2eef_7dlist = []
        for panda2handbase in panda2handbase_2dlist:
            panda2eef = pu.multiply_multi_transforms(panda2handbase, self.pandahandbase2eef)
            panda2eef_7dlist.append(pu.merge_pose_2d(panda2eef))

        for _ in range(maximum_attempts):
            path_len = self.fp_ctrl.plan_eef_cartesian_path(panda2eef_7dlist)
            if path_len != 0 and path_len <= path_len_threshold:
                user_input = input(f"Press Any keys to execute the plan {path_len} points, type '1' to replan, type '2' to skip:\n")
                if "1" in user_input.lower():
                    continue
                if "2" in user_input.lower():
                    return None
                self.fp_ctrl.execute_plan()
                return panda2handbase_2dlist
        return None

    
    def gripper_approach_compare(self, panda2handbase_2dlists: List, maximum_attempts=3, path_len_threshold=550):
        """
        panda2handbase_2dlists: [[pose2d1, pose2d2, ...], [pose2d1, pose2d2, ...], ...]
        input is panda2handbase, but we need to convert this to panda2eef for the robot control
        """
        panda2eef_7dlists = []
        for panda2handbase_2dlist in panda2handbase_2dlists:
            panda2eef_7dlist = []
            for panda2handbase in panda2handbase_2dlist:
                panda2eef = pu.multiply_multi_transforms(panda2handbase, self.pandahandbase2eef)
                panda2eef_7dlist.append(pu.merge_pose_2d(panda2eef))
            panda2eef_7dlists.append(panda2eef_7dlist)

        for _ in range(maximum_attempts):
            execute_index, path_len = self.fp_ctrl.plan_eef_cartesian_path_compare(panda2eef_7dlists)
            if execute_index is not None and path_len <= path_len_threshold:
                user_input = input(f"Press Any keys to execute the plan {path_len} points, type '1' to replan, type '2' to skip:\n")
                if "1" in user_input.lower():
                    continue
                if "2" in user_input.lower():
                    return None
                self.fp_ctrl.execute_plan()
                return panda2handbase_2dlists[execute_index]
        return None
        

    def pred_sp_pose(self, scene_pc, centered_qr_obj_pc):
        if not hasattr(self, 'sp_model'):
            logging.warning("Stable Placement Model is not loaded! Please set use_sp_model to True in the args.")
            return None
        with torch.no_grad():
            # scene_pc: [B, N, 3]; centered_qr_obj_pc: [B, N, 3]
            crop_center2pred_qr_obj_pose, _ = self.sp_model(scene_pc, centered_qr_obj_pc)
            return crop_center2pred_qr_obj_pose


    def sp_start(self):
        objectUniNames = list(self.obj_uni_names_dataset.keys())
        height_offset = 0.1 # For 
        target_idx = 0; targetUniName = objectUniNames[target_idx]
        panda2target_center = None; goal_pose_img = None; panda2pred_target_center = None
        with keyboard.Events() as events:
            while True:
                self.refresh_color_frame(mixed_img=goal_pose_img)

                if args.pybullet_debug:
                    self.pb_visualizer.step_simulation()

                key = None
                event = events.get(0.0001)
                if event is not None:
                    if isinstance(event, events.Press) and hasattr(event.key, 'char'):
                        key = event.key.char
                
                if key == "u":
                    target_idx = (target_idx + 1) % len(objectUniNames)
                    targetUniName = objectUniNames[target_idx]
                    panda2target_center = None
                    goal_pose_img = None
                    if any([name in targetUniName for name in self.smaller_lift_name]):
                        height_offset = 0.1
                    else:
                        height_offset = 0.2
                    logging.warning(f"Switching to the next object: {targetUniName}; Target Offset: {height_offset}")

                if key == "o":
                    target_idx = (target_idx - 1) % len(objectUniNames)
                    targetUniName = objectUniNames[target_idx]
                    panda2target_center = None
                    goal_pose_img = None
                    if any([name in targetUniName for name in self.smaller_lift_name]):
                        height_offset = 0.1
                    else:
                        height_offset = 0.2
                    logging.warning(f"Switching to the next object: {targetUniName}; Target Offset: {height_offset}")
                
                if key == "i":
                    logging.warning(f"Current target object: {targetUniName}")

                
                if key == "q":
                    goal_pose_img = None
                    panda2target_center = None
                    for _ in range(10):
                        # Stage 0: Detect the target object
                        panda2target_center, pose_annotated_color_img = self.detect_target_object(targetUniName)
                        if panda2target_center is not None:
                            break
                    
                    if panda2target_center is not None:
                        self.refresh_color_frame(color_img=pose_annotated_color_img, waitkey=2000)
                        # Compute the grasp pose
                        pandabase2handbase_pre_grasp, pandabase2handbase_grasp, \
                            pandabase2handbase_pre_grasp_reverse, pandabase2handbase_grasp_reverse \
                                = self.compute_target_grasp_pose(panda2target_center, targetUniName)
                        
                        # Stage 1: Approach the object
                        exec_plan = self.gripper_approach_compare([[pandabase2handbase_pre_grasp, pandabase2handbase_grasp], \
                                                                 [pandabase2handbase_pre_grasp_reverse, pandabase2handbase_grasp_reverse]])
                        if exec_plan is None:
                            logging.warning("No valid plan found! Please try again.")
                            continue
                        panda2handbase_pre_grasp_exec, panda2handbase_grasp_exec = exec_plan
                        new_target_center2handbase, new_target_center2pre_handbase = self.update_target2hand_naive(panda2handbase_grasp_exec, panda2target_center)
                        # Stage 2: Grasp the object
                        self.fp_ctrl.close_gripper()
                        
                        # Stage 3: Lift the object to the certain pose (We need to find a predefined pose to do this) and update the new target_center2handbase pose
                        # We now assume the transformation between hand and target is unchanged
                        self.fp_ctrl.small_lift(cur_eef_pose=pu.merge_pose_2d(panda2handbase_grasp_exec), height=height_offset) # There is get_eef_pose function in the small_lift, might need to change it
                        self.fp_ctrl.home_arm()
                    logging.warning(f"Current stage is Q stage, next stage is W stage to predict the target stable pose!")


                if key == "w":
                    # Stage 4: Get the scene points cloud and the target object point cloud
                    color_img, depth_img = self.get_raw_rgbd_frame()
                    panda2scene_pc, pc_rgb = self.get_pc_from_rgbd(color_img, depth_img)
                    centered_qr_obj_pc = self.get_obj_cad_pc(targetUniName)
                    # crop the scene point cloud
                    panda2crop_center = self.pandabase2crop_center
                    panda2crop_center_pos = panda2crop_center[0]
                    low_up_bbox = np.array([panda2crop_center_pos, panda2crop_center_pos]) + np.array([-self.crop_size, self.crop_size])
                    # visualize crop region bounding box use open3d
                    crop_bbox = pu.o3d_bounding_box(panda2crop_center_pos, self.crop_size)
                    # pu.visualize_pc(panda2scene_pc, other_geoms=crop_bbox)
                    panda2crop_pc, crop_mask = pu.crop_pc(panda2scene_pc, low_up_bbox)
                    centered_crop_pc = se3_transform_pc(*pu.invert_transform(panda2crop_center), panda2crop_pc)
                    # centered_crop_pc = pc_random_downsample(centered_crop_pc, num_points=30720)
                    centered_crop_pc = torch.tensor(centered_crop_pc, dtype=torch.float32).to(self.device).unsqueeze(0)
                    centered_qr_obj_pc = torch.tensor(centered_qr_obj_pc, dtype=torch.float32).to(self.device).unsqueeze(0)
                    crop_center2pred_target_center_raw = self.pred_sp_pose(centered_crop_pc, centered_qr_obj_pc)
                    crop_center2pred_target_center_2d = pu.split_7d(crop_center2pred_target_center_raw.squeeze(dim=0).cpu().numpy())
                    panda2pred_target_center = pu.multiply_multi_transforms(panda2crop_center, crop_center2pred_target_center_2d)
                    # visualize the crop point cloud and add coordinate frame at the crop center
                    visualize_pred_pose(centered_crop_pc, centered_qr_obj_pc, crop_center2pred_target_center_raw)
                    logging.warning(f"Current stage is W stage, next stage is E stage to place the object to the predicted pose!")


                if key == "e" or key == "a":
                    if panda2pred_target_center is None:
                        logging.warning("Please run the 'e' key first to predict the target pose!")
                        continue
                    if key == "a": # Use the predicted pose as the target pose
                        panda2pred_target_center[1] = panda2target_center[1]
                    # Stage 4: Approach the target location
                    # Compute the goal pose in the robot base frame
                    # Same height as the current pose to there
                    panda2pred_target_center_miti = deepcopy(panda2pred_target_center)
                    panda2pred_target_center_miti[0][2] = self.panda2target_center_z
                    goal_pandabase2handbase_release, goal_pandabase2handbase_pre_release, goal_pandabase2handbase_pre_pre_release = \
                        self.compute_target_release_pose(panda2pred_target_center_miti, new_target_center2handbase, targetUniName)
                    goal_pandabase2eef_release = pu.multiply_multi_transforms(goal_pandabase2handbase_release, self.pandahandbase2eef)
                    exec_plan = self.gripper_approach([goal_pandabase2handbase_pre_pre_release, goal_pandabase2handbase_pre_release, goal_pandabase2handbase_release])
                    if exec_plan is None:
                        logging.warning("No valid plan found! Please try again.")
                        continue
                    # Get and render the target goal pose
                    # camera2target_center should be Retrieved from the model
                    goal_pose_img = self.render_target_goal_pose(targetUniName, panda2pred_target_center_miti)                   
                    # Stage 5: Release the object
                    self.fp_ctrl.open_gripper()
                    self.fp_ctrl.small_lift(cur_eef_pose=pu.merge_pose_2d(goal_pandabase2eef_release), height=height_offset)
                    # Stage 6: Home the arm
                    self.fp_ctrl.home_arm()
                    logging.warning(f"Current stage is E stage, next stage is Q stage to detect object and grasp object!")
                

                if key == "k":
                    # Compute the work space center frame and crop size
                    color_img, depth_img = self.get_raw_rgbd_frame()
                    panda2scene_pc, pc_rgb = self.get_pc_from_rgbd(color_img, depth_img)
                    centered_qr_obj_pc = self.get_obj_cad_pc(targetUniName)
                    # crop the scene point cloud
                    panda2crop_center_pos = np.array([0.525, -0.05, 0.])
                    low_up_bbox = np.array([panda2crop_center_pos, panda2crop_center_pos]) + np.array([-self.crop_size, self.crop_size])
                    # visualize crop region bounding box use open3d
                    crop_bbox = pu.o3d_bounding_box(panda2crop_center_pos, self.crop_size)
                    pu.visualize_pc(panda2scene_pc, other_geoms=crop_bbox)
                    crop_pc, crop_mask = pu.crop_pc(panda2scene_pc, low_up_bbox)
                    # compute the surface center # The table must clean!!
                    panda2crop_center_pos[2] = np.mean(crop_pc[:, 2])
                    # Save the panda2crop_center_pos as .txt file
                    camera_info_dir = os.path.dirname(args.camera_extrinsics_path)
                    # np.savetxt(os.path.join(camera_info_dir, "panda2crop_center_pos.txt"), panda2crop_center_pos)
                    self.pandabase2crop_center = [panda2crop_center_pos, [0., 0., 0., 1.]]
                    # visualize the crop point cloud and add coordinate frame at the crop center
                    crop_center_frame = pu.o3d_coordinate_frame(panda2crop_center_pos)
                    pu.visualize_pc(crop_pc, other_geoms=crop_center_frame)


                if key == "r":
                    # Stage 6: Home the arm
                    self.fp_ctrl.home_gripper()
                    self.fp_ctrl.small_lift(height=0.2)
                    self.fp_ctrl.home_arm()
                    panda2target_center = None
                    goal_pose_img = None
                    panda2pred_target_center = None
                    logging.warning(f"Current stage is R stage, everything has been reset!")

    
    def detect_whole_scene(self, objname_lst=None, vis_detection=True):
        panda2detected_obj = {}
        if objname_lst is not None:
            detecting_objname_lst = []
            for objname in objname_lst:
                for objUniName in self.obj_uni_names_dataset.keys():
                    if objname in objUniName: # objname is a substring of objUniName (align with the folder name)
                        detecting_objname_lst.append(objUniName)
                        break
        else:
            detecting_objname_lst = list(self.obj_uni_names_dataset.keys())
                
        for objUniName in detecting_objname_lst:
            self.refresh_color_frame() # Keep refreshing the color frame avoid stuck
            self.register_target_object(objUniName)
            panda2target_center, pose_annotated_color_img = self.detect_target_object(objUniName)
            if panda2target_center is not None:
                panda2detected_obj.update({objUniName: 
                                           {
                                               "panda2target_center": panda2target_center,
                                               "pose_annotated_color_img": pose_annotated_color_img,
                                           }
                                        })
            else:
                logging.warning(f"Object {objUniName} is not detected!!!")

        if vis_detection:
            for detected_objUniName in panda2detected_obj.keys():
                obj_info = panda2detected_obj[detected_objUniName]
                rgb_pose_annotated_img = obj_info["pose_annotated_color_img"]
                self.refresh_color_frame(color_img=rgb_pose_annotated_img, waitkey=2000)
        return panda2detected_obj


    def rearange_start(self, objname_lst=None):
        sel_cfg_mesh_rendering = None
        sel_cfg_index = 0
        with keyboard.Events() as events:
            while True:
                self.refresh_color_frame(mixed_img=sel_cfg_mesh_rendering)

                key = None
                event = events.get(0.0001)
                if event is not None:
                    if isinstance(event, events.Press) and hasattr(event.key, 'char'):
                        key = event.key.char

                if key == "q":
                    panda2detected_obj = self.detect_whole_scene(objname_lst, vis_detection=True)
                    if len(panda2detected_obj) == 0:
                        logging.warning("No object detected! Please check the scene.")
                        continue
                    targetUniName_lst = list(panda2detected_obj.keys())
                    logging.warning(f"Current stage is Q stage, asked to detect {len(objname_lst)} objs. {len(panda2detected_obj)} has been detected. Next stage is W to propose possible setup!")

                if key == "w":
                    success_scene_center_cfg_dicts = self.online_scene_generation(targetUniName_lst, 
                                                                                  qrsceneCenter2qrregionCenter=[[0., 0.15, 0.], 
                                                                                                                [0., 0., 0.], 
                                                                                                                [0.225, 0.125, 0.]])
                    if len(success_scene_center_cfg_dicts) == 0:
                        logging.warning("No scene generated! Please check the scene.")
                        continue
                    # Rearrange the scene
                    for _, scene_cfg in success_scene_center_cfg_dicts.items():
                        for targetUniName, sceneCenter2placedObjCenter in scene_cfg.items():
                            panda2placedObjCenter = pu.multiply_multi_transforms(self.pandabase2crop_center, sceneCenter2placedObjCenter)
                            scene_cfg.update({targetUniName: panda2placedObjCenter})
                    success_cfg_dicts = success_scene_center_cfg_dicts # in the panda base frame
                    logging.warning(f"Current stage is W stage, Next stage is E to visualize the scene or use 'U/O' to pick the desired scene! After that, use 'A' to pick and place the object!")

                if key == "i":
                    # Visualize the scene
                    color_img, depth_img = self.get_raw_rgbd_frame()
                    for _, scene_cfg in success_cfg_dicts.items():
                        cfg_targetUniName_lst = list(scene_cfg.keys())
                        cfg_panda2target_center_lst = list(scene_cfg.values())
                        mesh_img_np = self.render_target_goal_pose_lst(cfg_targetUniName_lst, cfg_panda2target_center_lst)
                        self.refresh_color_frame(mixed_img=mesh_img_np, waitkey=2000)

                if key == "u":
                    sel_cfg_index = (sel_cfg_index + 1) % len(success_cfg_dicts)
                    sel_cfg = success_cfg_dicts[sel_cfg_index]
                    sel_cfg_mesh_rendering = self.render_target_goal_pose_lst(list(sel_cfg.keys()), list(sel_cfg.values()))

                if key == "o":
                    sel_cfg_index = (sel_cfg_index - 1) % len(success_cfg_dicts)
                    sel_cfg = success_cfg_dicts[sel_cfg_index]
                    sel_cfg_mesh_rendering = self.render_target_goal_pose_lst(list(sel_cfg.keys()), list(sel_cfg.values()))
                                                                              
                if key == "a":
                    # Pick and place the object
                    logging.warning(f"Current stage is A stage, going to the pick and place for the current cfg Next stage is Z to detect the lowest object!")
                    self.pick_and_place(sel_cfg)

                if key == "r":
                    sel_cfg_mesh_rendering = None
                    success_cfg_dicts = None
                    sel_cfg_index = 0
                    logging.warning("Reset the scene.")
                    self.fp_ctrl.home_gripper()
                    self.fp_ctrl.small_lift(height=0.2)
                    self.fp_ctrl.home_arm()
                    self.clear_registeration()

    
    def online_scene_generation(self, targetUniName_lst, num_setups=10, qrsceneCenter2qrregionCenter=None):
        """
        Generate the scene using the RoboSensai model.
        """
        self.sg_env.hard_update_args(targetUniName_lst=targetUniName_lst, qrsceneCenter2qrregionCenter=qrsceneCenter2qrregionCenter)
        success_scene_center_cfg_dicts = {}; success_setup = 0
        with torch.no_grad():
            next_seq_obs = torch.Tensor(self.sg_env.reset()).to(self.device)
            # Scene and obj feature tensor are keeping updated inplace; 1 is the number of envs
            next_scene_ft_obs = torch.zeros((1, ) + (self.sg_env.scene_ft_dim, )).to(self.device)
            next_obj_ft_obs = torch.zeros((1, ) + (self.sg_env.obj_ft_dim, )).to(self.device)
            reset_infos = [self.sg_env.info]
            self.sg_agent.preprocess_pc_update_tensor(next_scene_ft_obs, next_obj_ft_obs, reset_infos, use_mask=True)

            while success_setup < num_setups:
                action, probs = self.sg_agent.select_action([next_seq_obs, next_scene_ft_obs, next_obj_ft_obs])
                
                next_seq_obs, _, done, infos = self.sg_env.step(action)
                self.sg_agent.preprocess_pc_update_tensor(next_scene_ft_obs, next_obj_ft_obs, infos, use_mask=True)
                
                next_seq_obs, done = torch.Tensor(next_seq_obs).to(self.device), torch.Tensor(done).to(self.device)
                
                terminal_index = done == 1
                terminal_nums = terminal_index.sum().item()
                # Compute the average episode rewards.
                if terminal_nums > 0:
                    terminal_ids = terminal_index.nonzero().flatten()
                    success_buf = torch.Tensor(combine_envs_float_info2list(infos, 'success', terminal_ids)).to(self.device)
                    success_ids = terminal_ids[success_buf.to(torch.bool)]

                    scene_center_cfg = combine_envs_float_info2list(infos, 'surfaceCenter2placedObjCenter_poses', terminal_ids)
                    for i, env_id in enumerate(terminal_ids):
                        if env_id in success_ids:
                            success_scene_center_cfg_dicts.update({success_setup: scene_center_cfg[i]})
                            success_setup += 1
        
        return success_scene_center_cfg_dicts
    

    def pick_and_place(self, scene_cfg):
        """
        Pick and place the object to the target location.
        """
        targetUniName_lst = list(scene_cfg.keys())
        targets_height = [self.obj_uni_names_dataset[targetUniName]["bbox"][1][2] * 2 for targetUniName in targetUniName_lst]
        # Sort the targetUniNames by the height; place the lowest object first
        sorted_targetUniName_lst = [x for _, x in sorted(zip(targets_height, targetUniName_lst))]

        scene_cfg_mesh_rendering = self.render_target_goal_pose_lst(list(scene_cfg.keys()), list(scene_cfg.values()))

        with keyboard.Events() as events:
            while True:
                self.refresh_color_frame(mixed_img=scene_cfg_mesh_rendering)
                if args.pybullet_debug:
                    self.pb_visualizer.step_simulation()

                key = None
                event = events.get(0.0001)
                if event is not None:
                    if isinstance(event, events.Press) and hasattr(event.key, 'char'):
                        key = event.key.char
                
                if key == "z":
                    if len(sorted_targetUniName_lst) == 0:
                        logging.warning("All objects have been placed!")
                        return
                    panda2target_center = None
                    for targetUniName in sorted_targetUniName_lst:
                        panda2target_center, annotated_color_img = self.detect_target_object(targetUniName, draw_xyz=False)
                        if panda2target_center is not None:
                            self.refresh_color_frame(color_img=annotated_color_img, waitkey=1000)
                            break
                    # Compute the grasp pose
                    pandabase2handbase_pre_grasp, pandabase2handbase_grasp, \
                        pandabase2handbase_pre_grasp_reverse, pandabase2handbase_grasp_reverse \
                            = self.compute_target_grasp_pose(panda2target_center, targetUniName)
                    # Approach the object
                    exec_plan = self.gripper_approach_compare([[pandabase2handbase_pre_grasp, pandabase2handbase_grasp], \
                                                            [pandabase2handbase_pre_grasp_reverse, pandabase2handbase_grasp_reverse]])
                    if exec_plan is None:
                        logging.warning(f"Object {targetUniName} cannot be approached! Please check the scene.")
                        continue
                    panda2handbase_pre_grasp_exec, panda2handbase_grasp_exec = exec_plan
                    # Update the target2handbase pose
                    new_target_center2handbase, new_target_center2pre_handbase = \
                        self.update_target2hand_naive(panda2handbase_grasp_exec, panda2target_center)
                    # Grasp the object
                    self.fp_ctrl.close_gripper()
                    # Lift the object
                    if any([name in targetUniName for name in self.smaller_lift_name]):
                        height_offset = 0.1
                    else:
                        height_offset = 0.2
                    self.fp_ctrl.small_lift(cur_eef_pose=pu.merge_pose_2d(pu.multiply_multi_transforms(panda2handbase_grasp_exec, self.pandahandbase2eef)), height=height_offset)
                    self.fp_ctrl.home_arm()
                    logging.warning(f"Current stage is Z stage, Next stage is X to place the target!")

        
                if key == "x" or key == "c":
                    if key == "c":
                        # Manually set the orientation of the tape target object; keep it same as grasp
                        scene_cfg[targetUniName][1] = panda2target_center[1]
                        # Update the scene_cfg_mesh_rendering for the new orientation
                        scene_cfg_mesh_rendering = self.render_target_goal_pose_lst(list(scene_cfg.keys()), list(scene_cfg.values()))

                    # Go to the target location
                    panda2pred_target_center = deepcopy(scene_cfg[targetUniName])
                    # Manually set the orientation of the tape target object; keep it same as grasp
                    if "paper_tape" in targetUniName:
                        panda2pred_target_center[1] = panda2target_center[1]

                    # Compute the release pose
                    panda2pred_target_center[0][2] = self.panda2target_center_z
                    pandabase2handbase_release, pandabase2handbase_pre_release, pandabase2handbase_pre_pre_release \
                            = self.compute_target_release_pose(panda2pred_target_center, new_target_center2handbase, targetUniName)
                    # Approach the target location
                    exec_plan = self.gripper_approach([pandabase2handbase_pre_pre_release, pandabase2handbase_pre_release, pandabase2handbase_release])
                    if exec_plan is None:
                        logging.warning("No valid plan found! Please try again.")
                        continue
                    # Release the object
                    self.fp_ctrl.open_gripper()
                    self.fp_ctrl.small_lift(cur_eef_pose=pu.merge_pose_2d(pu.multiply_multi_transforms(pandabase2handbase_release, self.pandahandbase2eef)), height=height_offset)
                    self.fp_ctrl.home_arm()

                    # Remove the object from the list; Already Done
                    sorted_targetUniName_lst.remove(targetUniName)
                    logging.warning(f"Target {targetUniName} has been placed. {len(sorted_targetUniName_lst)} objs left. Current stage is Z stage, Next stage is X to place the target!")

                if key == "r":
                    logging.warning("Reset the scene.")
                    self.fp_ctrl.home_gripper()
                    self.fp_ctrl.home_arm()
                    self.clear_registeration()

                
    # Utils
    def refresh_color_frame(self, color_img=None, mixed_img=None, waitkey=1):
        # color_img and mixed_img both are rgb(a) images
        if color_img is None:
            color_img, _ = self.get_raw_rgbd_frame()

        if mixed_img is None:
            vis_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        else:
            vis_img = combine_images(color_img, mixed_img) # combine_images have converted the images to BGR inside
        cv2.imshow("Main Visualization", vis_img)
        cv2.waitKey(waitkey)


    def render_target_goal_pose(self, targetUniName, panda2target_center):
        # Set open3d renderer; all transformations are in the camera frame
        renderer = o3d.visualization.rendering.OffscreenRenderer(self.o3d_cam_intrinsic.width, self.o3d_cam_intrinsic.height)
        renderer.setup_camera(self.o3d_cam_intrinsic, np.eye(4)) # open3d uese camera-to-world pose
        renderer.scene.set_background([1.0, 1.0, 1.0, 0.])

        target_mesh = deepcopy(self.obj_uni_names_dataset[targetUniName]["o3d_mesh"])
        # Create a transformation matrix for the mesh
        # Example: Rotate 45 degrees around the z-axis and translate by (0.5, 0.5, 0)
        camera2target_center = pu.multiply_multi_transforms(pu.invert_transform(self.panda2camera), panda2target_center)
        transformation_matrix = pu.pose2d2matrix(camera2target_center)

        # Apply the transformation to the mesh
        target_mesh.transform(transformation_matrix)

        # Add mesh to scene for rendering
        renderer.scene.add_geometry(targetUniName, target_mesh, o3d.visualization.rendering.MaterialRecord())  # Adding default material

        # Create an image using the renderer
        mesh_img = renderer.render_to_image()
        # Convert Open3D Image to numpy array
        mesh_img_np = np.asarray(mesh_img)

        return mesh_img_np
    

    def render_target_goal_pose_lst(self, targetUniName_lst, panda2target_center_lst):
        # Render the mesh; all transformations are in the camera frame
        renderer = o3d.visualization.rendering.OffscreenRenderer(self.o3d_cam_intrinsic.width, self.o3d_cam_intrinsic.height)
        renderer.setup_camera(self.o3d_cam_intrinsic, np.eye(4)) # open3d uese camera-to-world pose
        renderer.scene.set_background([1.0, 1.0, 1.0, 0.])

        camera2target_center_lst = [pu.multiply_multi_transforms(pu.invert_transform(self.panda2camera), panda2target_center) for panda2target_center in panda2target_center_lst]
        for targetUniName, camera2target_center in zip(targetUniName_lst, camera2target_center_lst):
            target_mesh = deepcopy(self.obj_uni_names_dataset[targetUniName]["o3d_mesh"])
            # Create a transformation matrix for the mesh
            # Example: Rotate 45 degrees around the z-axis and translate by (0.5, 0.5, 0)
            transformation_matrix = pu.pose2d2matrix(camera2target_center)
            # Apply the transformation to the mesh
            target_mesh.transform(transformation_matrix)
            # Add mesh to scene for rendering
            renderer.scene.add_geometry(targetUniName, target_mesh, o3d.visualization.rendering.MaterialRecord())  # Adding default material
                # Create an image using the renderer
        mesh_img = renderer.render_to_image()
        # Convert Open3D Image to numpy array
        mesh_img_np = np.asarray(mesh_img)
        return mesh_img_np


    def get_raw_rgbd_frame(self):
        # Get the raw color image from the camera
        color, depth = self.obj_est.get_raw_rgbd_frame()
        return color, depth
    

    def get_obj_cad_pc(self, targetUniName):
        return self.obj_uni_names_dataset[targetUniName]["centered_obj_pc"]


    def get_pc_from_rgbd(self, color_img, depth_img, min_depth=0., to_panda_base=True):
        camera2pc, pc_rgb = self.obj_est.get_pc_from_rgbd(color_img, depth_img, min_depth=min_depth)
        if to_panda_base:
            panda2pc = se3_transform_pc(*self.panda2camera, camera2pc)
            return panda2pc, pc_rgb
        return camera2pc, pc_rgb

    
    def __del__(self):
        if hasattr(self, 'fp_ctrl'):
            self.fp_ctrl.disconnect()



if __name__ == "__main__":
    # TODO: Tape is symmetry, we need to manually make sure the prediction axis is up
    # Wine Glass is not very accurate in the prediction. Make sure the z-axis is up
    RealObjectNames = ["6_chinese_ceramic_bowl_small", "23_dawn", "34_meyerscleanday", "36_mustard", 
                       "37_M_mug", "44_paper_tape", "45_pepper_powder", 
                       "46_pepsi", "125_wine_glass.", "133_domino_suger"], 
    
    args = parse_args()
    pap = PickAndPlace(args)

    if args.task == "stable_placement":
        pap.sp_start()
    elif args.task == "rearrangement":
        # rearrange_objnames = ["44_paper_tape", "37_M_mug", "46_pepsi", "36_mustard", "133_domino_suger"]
        # rearrange_objnames = ["133_domino_suger", "37_M_mug", "46_pepsi", "34_meyerscleanday", "6_chinese_ceramic_bowl_small"]
        # rearrange_objnames = ["133_domino_suger", "37_M_mug", "125_wine_glass", "34_meyerscleanday", "23_dawn"]
        # rearrange_objnames = ["44_paper_tape", "37_M_mug", "46_pepsi", "125_wine_glass", "36_mustard"]
        rearrange_objnames = ["125_wine_glass", "44_paper_tape", "37_M_mug", "23_dawn", "36_mustard"]
        pap.rearange_start(objname_lst=RealObjectNames)
        pap.rearange_start(objname_lst=rearrange_objnames)