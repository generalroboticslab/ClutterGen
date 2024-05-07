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

from utils import read_json, natural_keys, set_logging_format, combine_images
from pose6d_run_inference import ObjectEstimator
from robot_related.franka_controller_client import FrankaPandaCtrl
from pybullet_visualizer import PybulletVisualizer

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
    parser.add_argument('--collect_data', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True) # https://docs.python.org/3/library/argparse.html#:~:text=%27%3F%27.%20One%20argument,to%20illustrate%20this%3A
    parser.add_argument('--object_pool_name', type=str, default='Union', help="Target object to be grasped. Ex: cube")
    parser.add_argument('--asset_root', type=str, default='assets', help="folder path that stores all urdf files")
    parser.add_argument('--object_pool_folder', type=str, default='group_objects/group4_real_objects', help="folder path that stores all urdf files")
    parser.add_argument('--object_textured_mesh_folder', type=str, default='assets/group_objects/group4_real_objects_mesh_downsampled', help="folder path that stores all urdf files")

    parser.add_argument('--rendering', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--realtime', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--quiet', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--panda_urdf_path', type=str, default=f'{code_dir}/robot_related/franka_description/robots/franka_panda.urdf')
    parser.add_argument('--camera_extrinsics_path', type=str, default=f'{code_dir}/assets/rs_camera/camera_extrinsics4.txt')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/FoundationPose/demo_data/mustard0')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/FoundationPose/debug')
    
    args = parser.parse_args()
    
    return args



class PickAndPlace:
    RealObjectsDict = ["Chinese Ceramic Bowl.", "White M Mug.", "Blue Tape.", "Blue Pepsi.", 
                       "Transparent Wine Glass Cup.", "Transparent Water Glass Cup.", "Pink Spray.", 
                       "Yellow Mustard Bottle.", "Red Pepper Powder Container.", "Blue Dish Wash Bottle.", "Spam Can.", "Yellow Domino Sugar Box."]
    def __init__(self, args) -> None:
        self.args = args
        set_logging_format(logging.WARNING)

        self.update_objects_database()
        self.pandaeef2handbase = self.get_pandaeef2handbase(args.panda_urdf_path)
        self.pandahand2eef = pu.invert_transform(self.pandaeef2handbase)
        self.pandabase2camera = self.get_pandabase2camera(args.camera_extrinsics_path)
        self.obj_est = ObjectEstimator(args)
        self.fp_ctrl = FrankaPandaCtrl()

        if self.args.rendering:
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
                o3d_mesh = o3d.io.read_triangle_mesh(obj_mesh_path) # only for rendering
                # to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
                extents = mesh.bounding_box.extents
                center2bboxcenter = mesh.bounding_box.centroid
                to_origin = trimesh.transformations.translation_matrix(-center2bboxcenter)

                bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

                self.obj_uni_names_dataset.update(
                    {obj_uni_name: 
                                {
                                    "urdf_path": obj_urdf_path, 
                                    "label": obj_label,
                                    "mesh": mesh,
                                    "o3d_mesh": o3d_mesh,
                                    "to_origin": to_origin,
                                    "bbox": bbox,
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
            pandabase2camera = np.array([list(map(float, line.split())) for line in lines[:]]).reshape(4, 4)
        pandabase2camera = pu.matrix2pose2d(pandabase2camera)
        return pandabase2camera


    def register_target_object(self, targetUniName):
        # Get the target object's URDF path
        target_obj_db = self.obj_uni_names_dataset[targetUniName]
        target_semantic_label = target_obj_db["label"]["semantic_label"]
        target_mesh = target_obj_db["mesh"]
        target_to_origin = target_obj_db["to_origin"]
        target_bbox = target_obj_db["bbox"]
        # import ipdb; ipdb.set_trace()

        self.obj_est.update_est_target(target_semantic_label, target_mesh, target_to_origin, target_bbox, UniName=targetUniName)
        self.cur_targetUniName = targetUniName
        target_center2handbase = self.obj_uni_names_dataset[targetUniName]["label"]["Center2GripperBase_pose"] # [x, y, z, roll, pitch, yaw]
        self.target_center2handbase = [target_center2handbase[0], pu.get_quaternion_from_euler(target_center2handbase[1])]
        self.hand2pre_hand = [[0., 0., -0.1], [0., 0., 0., 1.]] # z-axis is pointing to the center of the frame; should be negative
        self.pre_target_center2handbase = pu.multiply_multi_transforms(self.target_center2handbase, self.hand2pre_hand)


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
            
            # Create a text label at the origin of the coordinate frame
            # text = o3d.geometry.Text3D()
            # text.text = label
            # text.scale = 0.02
            # text.position = pos
            # vis.add_geometry(text)

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
    
    
    def detect_target_object(self, targetUniName, strict=True):
        if self.cur_targetUniName is None or self.cur_targetUniName != targetUniName:
            self.register_target_object(targetUniName)
            logging.info(f"Target object {targetUniName} registered.")

        color_img, target_mask, camera2target_center_mat = self.obj_est.est_obj_pose6d(strict=strict)

        if camera2target_center_mat is not None:
            target_center2handbase = self.target_center2handbase
            pre_target_center2handbase = self.pre_target_center2handbase

            # Concert the Homogeneous matrix to the pose format
            camera2target_center = pu.matrix2pose2d(camera2target_center_mat)
            # If it is tape, we need to make sure the z-axis is pointing up
            # if "tape" in self.cur_targetUniName.lower():
            #     if camera2target_center[1][0] > 0:
            #         camera2target_center[1] = pu.get_quaternion_from_euler([np.pi, 0, 0])
            #     if camera2target_center[1][1] > 0:
            #         camera2target_center[1] = pu.get_quaternion_from_euler([0, np.pi, 0])
            #     if camera2target_center[1][2] > 0:
            #         camera2target_center[1] = pu.get_quaternion_from_euler([0, 0, np.pi])
            
            # Visualize the grasp pose at the eef link
            pandabase2target_center = pu.multiply_multi_transforms(self.pandabase2camera, camera2target_center)
            camera2handbase_pre_grasp = pu.multiply_multi_transforms(camera2target_center, pre_target_center2handbase, p.invertTransform(*self.pandaeef2handbase))
            camera2handbase_grasp = pu.multiply_multi_transforms(camera2target_center, target_center2handbase, p.invertTransform(*self.pandaeef2handbase))
            camera2handbase_grasp_mat = pu.pose2d2matrix(camera2handbase_grasp)
            vis_img = self.obj_est.draw_posed_3d_box(color_img, camera2target_center_mat)
            vis_img = self.obj_est.draw_xyz_axis(color_img, camera2handbase_grasp_mat)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

            # Compute the grasp and pre-grasp pose in the robot base frame
            pandabase2handbase_pre_grasp = pu.multiply_multi_transforms(self.pandabase2camera, camera2handbase_pre_grasp)
            pandabase2handbase_grasp = pu.multiply_multi_transforms(self.pandabase2camera, camera2handbase_grasp)

            if self.args.rendering:
                panda_joints_state = self.fp_ctrl.get_joint_state()
                self.pb_visualizer.set_panda_joints(panda_joints_state)
                self.pb_visualizer.set_panda_gripper([0.04, 0.04]) # open the gripper
                self.pb_visualizer.set_obj_pose(targetUniName, pandabase2target_center)
                self.camera_axes_ids = self.pb_visualizer.draw_camera_axis(self.pandabase2camera, old_axis_ids=self.camera_axes_ids)

            labeled_color_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            return labeled_color_img, pandabase2target_center, pandabase2handbase_pre_grasp, pandabase2handbase_grasp
        return color_img, None, None, None
    

    def update_target2hand_pose(self, targetUniName, strict=True):
        if self.cur_targetUniName is None or self.cur_targetUniName != targetUniName:
            self.register_target_object(targetUniName)
            logging.info(f"Target object {targetUniName} registered.")

        color_img, target_mask, camera2target_center_mat = self.obj_est.est_obj_pose6d(strict=strict)

        if camera2target_center_mat is not None:
            camera2target_center = pu.matrix2pose2d(camera2target_center_mat)
            # Visualize the grasp pose at the eef link
            pandabase2target_center = pu.multiply_multi_transforms(self.pandabase2camera, camera2target_center)
            pandabase2eef = pu.split_7d(self.fp_ctrl.get_eef_pose())
            new_target_center2handbase = pu.multiply_multi_transforms(pu.invert_transform(pandabase2target_center), pandabase2eef)
            # z-axis is pointing to the center of the frame; should be negative
            new_pre_target_center2handbase = pu.multiply_multi_transforms(new_target_center2handbase, self.hand2pre_hand)
            return new_target_center2handbase, new_pre_target_center2handbase
        return None, None


    def approach_grasp(self, pose2d_lst: list):
        # Probably need BRRT to do the motion planning in pybullet
        logging.info(f"Trying to pick {self.cur_targetUniName}.")
        pose7d_lst = [pu.merge_pose_2d(pose2d) for pose2d in pose2d_lst]
        self.fp_ctrl.plan_eef_cartesian_path(pose7d_lst)


    # Utils
    def render_target_goal_pose(self, targetUniName, goal_camera2target_center):
        target_mesh = deepcopy(self.obj_uni_names_dataset[targetUniName]["o3d_mesh"])

        # Create a transformation matrix for the mesh
        # Example: Rotate 45 degrees around the z-axis and translate by (0.5, 0.5, 0)
        transformation_matrix = pu.pose2d2matrix(goal_camera2target_center)

        # Apply the transformation to the mesh
        target_mesh.transform(transformation_matrix)

        # Render the mesh; all transformations are in the camera frame
        renderer = o3d.visualization.rendering.OffscreenRenderer(self.o3d_cam_intrinsic.width, self.o3d_cam_intrinsic.height)
        renderer.setup_camera(self.o3d_cam_intrinsic, np.eye(4)) # open3d uese camera-to-world pose
        renderer.scene.set_background([1.0, 1.0, 1.0, 0.])

        # Add mesh to scene for rendering
        renderer.scene.add_geometry("mesh", target_mesh, o3d.visualization.rendering.MaterialRecord())  # Adding default material

        # Create an image using the renderer
        mesh_img = renderer.render_to_image()
        # clean the renderer
        renderer.scene.remove_geometry("mesh")
        # Convert Open3D Image to numpy array
        mesh_img_np = np.asarray(mesh_img)

        return mesh_img_np


    def get_raw_rgbd_frame(self):
        # Get the raw color image from the camera
        color, depth = self.obj_est.get_raw_rgbd_frame()
        return color, depth

    
    def __del__(self):
        if hasattr(self, 'fp_ctrl'):
            self.fp_ctrl.disconnect()



if __name__ == "__main__":
    # TODO: Tape is symmetry, we need to manually make sure the prediction axis is up
    # Wine Glass is not very accurate in the prediction. Make sure the z-axis is up
    args = parse_args()
    pap = PickAndPlace(args)
    fp_ctrl = pap.fp_ctrl
    objectUniNames = list(pap.obj_uni_names_dataset.keys())
    target_object = objectUniNames[8]
    height_offset = 0.25; pandabase2handbase_pre_grasp = None; goal_pose_img = None
    with keyboard.Events() as events:
        while True:
            color_img, depth_img = pap.get_raw_rgbd_frame()
            if goal_pose_img is not None:
                vis_img = combine_images(color_img, goal_pose_img)
            else:
                vis_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("Main Visualization", vis_img)
            cv2.waitKey(1)
            key = None
            event = events.get(0.0001)
            if event is not None:
                if isinstance(event, events.Press) and hasattr(event.key, 'char'):
                    key = event.key.char
            if key == "q":
                while pandabase2handbase_pre_grasp is None:
                    # Stage 0: Detect the target object
                    pose_annotated_color_img, pandabase2target_center, pandabase2handbase_pre_grasp, pandabase2handbase_grasp = pap.detect_target_object(target_object)
                if pandabase2handbase_pre_grasp is not None:
                    bgr_pose_annotated_img = cv2.cvtColor(pose_annotated_color_img, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Main Visualization", bgr_pose_annotated_img)
                    cv2.waitKey(2000)
                    # Stage 1: Approach the object
                    pap.approach_grasp([pandabase2handbase_pre_grasp, pandabase2handbase_grasp])
                    fake_pandabase2target_center = pandabase2target_center
                    # Stage 2: Grasp the object
                    fp_ctrl.close_gripper()
                    # Stage 3: Lift the object to the certain pose (We need to find a predefined pose to do this) and update the new target_center2handbase pose
                    fp_ctrl.small_lift(cur_eef_pose=pu.merge_pose_2d(pandabase2handbase_grasp), height=height_offset) # There is get_eef_pose function in the small_lift, might need to change it
                    # new_target_center2handbase, new_pre_target_center2handbase = pap.update_target2hand_pose(target_object)
                    new_target_center2handbase, new_pre_target_center2handbase = pap.target_center2handbase, pap.pre_target_center2handbase
                    goal_pose_img = None
            
            if key == "w":
                # Stage 4: Approach the target location
                # Get and render the target goal pose
                # camera2target_center should be Retrieved from the model
                fake_camera2target_center = pu.multiply_multi_transforms(pu.invert_transform(pap.pandabase2camera), fake_pandabase2target_center)
                goal_pose_img = pap.render_target_goal_pose(target_object, fake_camera2target_center)
                # Compute the goal pose in the robot base frame
                # Same height as the current pose to there
                goal_pandabase2handbase_pre_grasp = pu.multiply_multi_transforms(fake_pandabase2target_center, new_pre_target_center2handbase, p.invertTransform(*pap.pandaeef2handbase))
                goal_pandabase2handbase_grasp = pu.multiply_multi_transforms(fake_pandabase2target_center, new_target_center2handbase, p.invertTransform(*pap.pandaeef2handbase))
                goal_pandabase2handbase_pre_pre_grasp = [list(goal_pandabase2handbase_pre_grasp[0]), list(goal_pandabase2handbase_pre_grasp[1])]
                goal_pandabase2handbase_pre_pre_grasp[0][2] += height_offset # 20cm above the target location
                pap.approach_grasp([goal_pandabase2handbase_pre_pre_grasp, goal_pandabase2handbase_pre_grasp, goal_pandabase2handbase_grasp])
                # Stage 5: Release the object
                fp_ctrl.open_gripper()
                fp_ctrl.small_lift(cur_eef_pose=pu.merge_pose_2d(goal_pandabase2handbase_grasp), height=height_offset)
                # Stage 6: Home the arm
                # fp_ctrl.home_arm()

                pandabase2handbase_pre_grasp = None