from typing import Dict, Any, Tuple
import time, datetime
from copy import deepcopy

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym_utils import *
from isaacgym.torch_utils import tf_inverse, tf_apply, quat_mul, quat_apply, tf_combine, normalize

import math
import os
import numpy as np
import torch
from Robot import Franka
from domain_randomizer import DomainRandomizer
from llm_query import GPT
from post_corrector import PostCorrector
from scene_importer import SceneImporter, DinoSAM

import matplotlib.pyplot as plt
import cv2

try:
    from pynput import keyboard
except ImportError: 
    print("*** Warning: pynput can not be used on the server ***")

try:
    from robot_gripper import Robot
except ImportError as e:
    print(f"*** Fail to import Robot class {e}***")


DeltaP = 0.01 # 1 cm for prismatic movement
DeltaR = 10/180 * np.pi # 15 degree for rotation movement
JointP = 0.002
ControlFreq = 30 # 6Hz control
IntervalSteps = int(1/DT / ControlFreq) # 10 steps for one control / # Interval steps should larger than or equal 2
Min_delta_x, Max_delta_x = -0.25, 0. # Try not give 0 in any dimension
Min_delta_y, Max_delta_y = -0.15, 0.15
Min_delta_z, Max_delta_z = 0., 0.3
Min_eef_angle, Max_eef_angle = 0, 2*np.pi - DeltaR # [0, 345] degree

# set random seed
np.random.seed(42)
torch.set_printoptions(precision=2, sci_mode=False)


class RoboSensaiEnv:    
    def __init__(self, args) -> None:
        # random generator
        self.rng = np.random.default_rng(234)
        # Add custom arguments
        self.args = args
        self.num_envs = self.args.num_envs
        self.device = self.args.sim_device if self.args.use_gpu_pipeline else 'cpu'
        self.rendering = self.args.rendering
        self.use_gpu_pipeline = self.args.use_gpu_pipeline

        # Sim2Real Arguments
        self.real = self.args.real if hasattr(self.args, "real") else False
        self.robot = Robot(device=self.device) if self.real else None
        self.red_finger = self.robot.finger1 if self.real else None # finger 19
        self.green_finger = self.robot.finger3 if self.real else None # finger 24
        self.gripper_height = self.robot.gripper_height if self.real else 0.12
        self.real_env_id = 0
        # Adjust interval steps
        if hasattr(self.args, "interval_steps"): self.interval_steps = self.args.interval_steps
        elif self.real: self.interval_steps = 2
        else: self.interval_steps = IntervalSteps

        # Assets files
        self.asset_root = "assets"
        self.urdf_folder = "objects/ycb_objects_origin_at_center_vhacd/urdfs"
        self.mesh_folder = "objects/ycb_objects_origin_at_center_vhacd/meshes"
        self.xml_folder = "objects/ycb_objects_origin_at_center_vhacd/xmls"
        self.envs = []
        self.object_name = [name.split(".")[0] for name in os.listdir(os.path.join(self.asset_root, self.urdf_folder)) if name.endswith(".urdf")]
        # self.object_name = ["cube"]
        self.object_name_index = {}
        for i, name in enumerate(self.object_name):
            self.object_name_index[name] = i
        
        self.visualize_pc = self.args.visualize_pc if hasattr(self.args, "visualize_pc") else False
        self.num_pcs = self.args.num_pcs if hasattr(self.args, "num_pcs") else 50000
        
        # Mics
        self.use_gpt = self.args.use_gpt if hasattr(self.args, "use_gpt") else False
        self.auto_reset = True
        self.maximum_steps = 50
        self.maximum_detect_force = 100
        self.reach_success_thred = 0.05
        self.reach_success_steps = 5
        self.obj_fall_threshold = -0.15
        self.pos_w = self.args.pos_weight if hasattr(self.args, "pos_weight") else 1.
        self.act_w = self.args.act_weight if hasattr(self.args, "act_weight") else 0.

        # Visualization Configs

        # Create Sim and Camera
        self._configure_sim()
        # Create all assets
        self._configure_asset()
        # Create robot arm
        self._configure_robot()
        # Create Scene importer
        self.scene_importer = SceneImporter()
        # Create DinoSAM
        self.dinosam = DinoSAM()
        # Create domain randomizer
        self.d_randomizer = DomainRandomizer(device=self.device)
        # Create GPT 
        self.gpt = GPT()
        # Create post corrector
        self.post_corrector = PostCorrector(self, device=self.device)
        
        # Model Configs
        self.hidden_size = self.args.hidden_size if hasattr(self.args, "hidden_size") else 256
        self.num_hidden_layers = self.args.num_hidden_layers if hasattr(self.args, "num_hidden_layers") else 5
        self.time_seq_obs = self.args.time_seq_obs if hasattr(self.args, "time_seq_obs") else 1

        #####################################################################
        ###=========================Actions & Observations================###
        #####################################################################
        # Observation dimension
        self.image_width, self.image_height = 640, 360
        self.rgb_image_dim = (self.image_height, self.image_width, 3)
        self.depth_image_dim = (self.image_height, self.image_width, 1)
        self.proprioception = self.robot.franka_num_dofs + 7 + 6 # 7 DOF + 7D eef pose + 6D eef velocity 
        self.single_img_obs_dim = (self.num_envs, self.rgb_image_dim[2]+self.depth_image_dim[2], self.image_height, self.image_width)
        self.single_proprioception_dim = (self.num_envs, self.proprioception)

        self.obs_seq_len = 1 # How to deal with image sequence?

        # Action dimension; How to handle the case that there is no joint action can reach a certain pose?
        self.action_range = [0, 1, 2]
        self.action_dim = (self.num_envs, 6 + 1) # 6D delta actions + grip action

        self.act_seq_len = 1

        # Create environments and allocate buffers
        self.create_envs()

        self._link_states_tensor()

        self.allocate_buffers()
        
        # Set hand work space

        # Goal Relative Pose Range

        # Reset all
        self.reset()

        # All environments simulation 1 s to ensure the reset successfully
        sim_sec(self.sim, 1)


    def _configure_sim(self):
        # configure sim
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = DT
        self.sim_params.substeps = min(2, max(1, int(240 * DT))) # 1<= substeps <=2
        self.sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.rest_offset = 0.000
        self.sim_params.physx.contact_offset = 0.005
        self.sim_params.physx.friction_offset_threshold = 0.001
        self.sim_params.physx.friction_correlation_distance = 0.0005
        self.sim_params.physx.num_threads = self.args.num_threads
        self.sim_params.physx.use_gpu = self.args.use_gpu
        # self.sim_params.physx.max_gpu_contact_pairs = self.sim_params.physx.max_gpu_contact_pairs*2
        # Must CC All substeps! Otherwise gym sometimes give 0 contact info!
        self.sim_params.physx.contact_collection = gymapi.CC_ALL_SUBSTEPS
        
        # create sim
        self.sim = gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, gymapi.SIM_PHYSX, self.sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")
    

    def _configure_viewer(self):
        self.viewer = gym.create_viewer(self.sim, gymapi.CameraProperties())
        
        if self.viewer is None:
            raise Exception("Failed to create viewer")
        
        table_pose = self.get_asset_pose('table')
        position = table_pose.p
        x, y, z = position.x, position.y, position.z
        cam_pos = gymapi.Vec3(x+0.8, y, z+1.2)
        cam_target = gymapi.Vec3(x-1, y, -0.5)
        middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
        gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)
    

    def _configure_camera(self):
        # point camera at middle env
        table_pose = self.get_asset_pose('table')
        position = table_pose.p
        x, y, z = position.x, position.y, position.z
        cam_transform = gymapi.Transform()
        # cam_transform.p = gymapi.Vec3(x+0.8, y, z+1.2)
        cam_transform.p = gymapi.Vec3(x+0.8, y, 0.5)
        cam_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(0.0))
        cam_trans, cam_ori  = transform2list(cam_transform)
        cam_trans, cam_ori = self.to_torch(cam_trans), self.to_torch(cam_ori)
        isaacCam2RealCam_quat = quat_conjugate(quat_from_euler_xyz(*torch.tensor([np.pi/2, -np.pi/2., 0.]))).to(self.device)
        isaacCam2RealCam_trans = torch.zeros(3).to(self.device)
        # Create camera
        self.camera_handles = []
        self.camera_tensors = []
        self.camera_view_quats = []
        self.camera_view_trans = []
        self.camera_proj_matrixs = []

        camera_properties = gymapi.CameraProperties()
        camera_properties.width = self.image_width
        camera_properties.height = self.image_height
        camera_properties.enable_tensors = False
        for i in range(self.num_envs):
            camera_handle = gym.create_camera_sensor(self.envs[i], camera_properties)                                     
            gym.set_camera_transform(camera_handle, self.envs[i], cam_transform)
            World2RealCam_quat, World2RealCam_trans = tf_combine(cam_ori, cam_trans, 
                                                                isaacCam2RealCam_quat,
                                                                isaacCam2RealCam_trans)
            # cam_view = np.array(gym.get_camera_view_matrix(self.sim, self.envs[i], camera_handle))
            cam_proj = np.array(gym.get_camera_proj_matrix(self.sim, self.envs[i], camera_handle))
            self.camera_handles.append(camera_handle)
            self.camera_view_quats.append(World2RealCam_quat)
            self.camera_view_trans.append(World2RealCam_trans)
            self.camera_proj_matrixs.append(cam_proj)

            # _rgb_tensor = gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], camera_handle, gymapi.IMAGE_COLOR)
            # env_rgb_tensor = gymtorch.wrap_tensor(_rgb_tensor)
            # _depth_tensor = gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], camera_handle, gymapi.IMAGE_DEPTH)
            # env_depth_tensor = gymtorch.wrap_tensor(_depth_tensor)
            # self.camera_tensors.append((env_rgb_tensor, env_depth_tensor))

        self.camera_view_quats = torch.stack(self.camera_view_quats, dim=0)
        self.camera_view_trans = torch.stack(self.camera_view_trans, dim=0)
        self.camera_proj_matrixs = self.to_torch(self.camera_proj_matrixs)

        self.visualize_camera_axis(cam_trans, cam_ori)

    
    def _configure_ground(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        gym.add_ground(self.sim, plane_params)


    def _configure_asset(self, scene_dict=None):
        # TODO: We should have a scene_dict to record the scene states
        self.scene_asset = {}
        self.scene_obj_bbox = self.object_name_index.copy()
        #####################################################################
        ###=========================Fixed Objects=======================###
        #####################################################################
        table_dims = [0.4, 0.8, 0.4]
        table_asset = self.create_box_asset(table_dims, fix_base_link=True)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.5, 0.0, table_dims[2]/2)
        self.scene_asset.update({"table": [table_asset, table_pose]})
        self.scene_obj_bbox.update({"table": table_dims})

        # Bin asset
        # bin_asset = self.create_target_asset(target_asset_file=os.path.join(self.urdf_folder, "fixed", "bin")+".urdf", 
        #                                      convex_dec=False, fix_base_link=True)
        # bin_dims = [0.1, 0.1, 0.1]
        # bin_pose = gymapi.Transform()
        # bin_pose.p = gymapi.Vec3(table_pose.p.x, -0.3, table_dims[2])
        # bin_pose.r = gymapi.Quat.from_euler_zyx(0, 0, -np.pi/2)
        # self.scene_asset.update({"bin": [bin_asset, bin_pose]})
        # self.scene_obj_bbox.update({"bin": bin_dims})

        #####################################################################
        ###=========================Movable Objects=======================###
        #####################################################################
        self.obj_assets = self.object_name_index.copy()
        self.obj_urdf = self.object_name_index.copy()
        self.obj_default_scaling = self.object_name_index.copy()
        for name in self.obj_assets:
            obj_asset = self.create_target_asset(target_asset_file=os.path.join(self.urdf_folder, name)+".urdf", convex_dec=True)
            obj_pose = deepcopy(table_pose)
            self.scene_asset[name] = [obj_asset, obj_pose]

            urdfstate = getUrdfStates(os.path.join(self.asset_root, self.urdf_folder, name)+".urdf")
            org_scale = deepcopy(urdfstate[0][1].scale) if hasattr(urdfstate[0][1], "scale") else [1., 1., 1.]
            self.scene_obj_bbox[name] = getMeshBbox(os.path.join(self.asset_root, self.mesh_folder, name, 'collision')+'.obj', scale=org_scale)
            self.obj_urdf[name] = urdfstate
            self.obj_default_scaling[name] = org_scale

        # Should follow pre-loading scene_dict to add or delete object
        self.all_ids = {
                            "root": {
                                "envs": [], 
                                "robot": []
                            },
                            "rb":{
                                "robot_links": [], 
                                "gripper": []
                            }
                        }
        for key in self.scene_asset.keys():
            self.all_ids["root"][key] = []
            self.all_ids["rb"][key] = []

        # pc asset (Only for visualization, not included in the scene_asset)
        if self.visualize_pc:
            self.pc_asset = self.create_GM_asset(radius=0.005, fix_base_link=True, disable_gravity=True)
            self.all_ids["root"]["pc"] = []
            self.pc_handles = []


    def _configure_robot(self, robot_name="franka"):
        self.robot = Franka(sim=self.sim, asset_root=self.asset_root, device=self.device)
        self.scene_asset.update({"robot": [self.robot.franka_asset, self.robot.franka_pose]})


    def create_envs(self):
        # add ground
        self._configure_ground()

        # env grid configure
        self.num_envs = self.args.num_envs
        self.num_per_row = int(math.sqrt(self.num_envs))
        spacing = 0.75
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        print("Creating %d environments" % self.num_envs)

        # prepare area
        prepare_area = gymapi.Transform()
        prepare_area.p = gymapi.Vec3(0., 1000., 0.5)
        self.prepare_area = transform2list(prepare_area)
        self.prepare_area = [self.to_torch(self.prepare_area[0]), self.to_torch(self.prepare_area[1])]

        # object position dict
        self.default_scene_dict = {}

        for i in range(self.num_envs):
            # create env
            env = gym.create_env(self.sim, env_lower, env_upper, self.num_per_row)
            self.envs.append(env)

            for obj_name in self.scene_asset.keys():
                if obj_name in ["robot"]: continue
                obj_asset, obj_pose = self.scene_asset[obj_name]
                obj_handle = gym.create_actor(env, obj_asset, obj_pose, obj_name, i, 0)
                obj_root_id = gym.get_actor_index(env, obj_handle, gymapi.DOMAIN_SIM)
                obj_rb_id = gym.get_actor_rigid_body_index(env, obj_handle, 0, gymapi.DOMAIN_SIM)
                self.all_ids["root"][obj_name].append(obj_root_id)
                self.all_ids["rb"][obj_name].append(obj_rb_id)

            # add robot
            franka_asset, franka_pose = self.scene_asset["robot"]
            franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 2)
            franka_id = gym.get_actor_index(env, franka_handle, gymapi.DOMAIN_SIM)
            self.all_ids["root"]["robot"].append(franka_id)
            for link_name in self.robot.franka_link_dict:
                link_id = gym.find_actor_rigid_body_index(env, franka_handle, link_name, gymapi.DOMAIN_SIM)
                self.all_ids["rb"]["robot_links"].append([i, link_id])
            # set dof properties
            gym.set_actor_dof_properties(env, franka_handle, self.robot.franka_dof_props)
            # set initial dof states
            gym.set_actor_dof_states(env, franka_handle, self.robot.default_dof_state, gymapi.STATE_ALL)
            # set initial position targets
            gym.set_actor_dof_position_targets(env, franka_handle, self.robot.default_dof_state["pos"])
            # get inital hand pose
            hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_gripper_mid")
            hand_pose = gym.get_rigid_transform(env, hand_handle)
            hand_id = gym.find_actor_rigid_body_index(env, franka_handle, "panda_gripper_mid", gymapi.DOMAIN_SIM)
            self.all_ids["rb"]["gripper"].append(hand_id)

            # add pc to visualize points cloud
            if self.visualize_pc:
                # Add pc
                temp_pc_ids = []; temp_pc_handles = []
                gm_pose_temp = gymapi.Transform()
                gm_pos_full = self.rng.uniform([1000.]*3, [1001.]*3, size=(self.num_pcs, 3))
                for k in range(self.num_pcs):
                    gm_pose_temp.p = gymapi.Vec3(*gm_pos_full[k, :])
                    gm_handle = gym.create_actor(env, self.pc_asset, gm_pose_temp, "pc", i, 3)
                    gm_id = gym.get_actor_index(env, gm_handle, gymapi.DOMAIN_SIM)
                    temp_pc_ids.append(gm_id); temp_pc_handles.append(gm_handle)
                self.all_ids["root"]["pc"].append(temp_pc_ids)
                self.pc_handles.append(temp_pc_handles)
        
        ## Record default scene setup
        for obj_name in self.scene_asset.keys():
            obj_asset, obj_pose = self.scene_asset[obj_name]
            scaling = 1 if obj_name not in ['robot', 'gripper'] else None
            obj_pos, obj_ori = transform2list(obj_pose)
            self.default_scene_dict[obj_name] = [self.to_torch(obj_pos), self.to_torch(obj_ori), 1]

        ## Transform all ids to tensors
        for obj_root_name, obj_root_ids in self.all_ids["root"].items():
            self.all_ids["root"][obj_root_name] = self.to_torch(obj_root_ids, dtype=torch.long)
        for obj_rb_name, obj_rb_ids in self.all_ids["rb"].items():
            self.all_ids["rb"][obj_rb_name] = self.to_torch(obj_rb_ids, dtype=torch.long)

        # Configure camera point
        if self.rendering:
            self._configure_viewer()
        self._configure_camera()

        # ==== prepare tensors =====
        # from now on, we will use the tensor API that can run on CPU or GPU
        gym.prepare_sim(self.sim)


    def _link_states_tensor(self):
        # get root state tensor
        _root_tensor = gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(_root_tensor)

        # get rigid body state tensor
        _rb_states = gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)
        
        # get dof state tensor
        if self.robot.franka_num_dofs > 0:
            self.num_dofs_sim = gym.get_sim_dof_count(self.sim)
            _dof_states = gym.acquire_dof_state_tensor(self.sim)
            self.dof_states = gymtorch.wrap_tensor(_dof_states)
            self.dof_pos = self.dof_states[:, 0].view(self.num_envs, -1)[:, :self.robot.franka_num_dofs].contiguous()
            self.dof_vel = self.dof_states[:, 1].view(self.num_envs, -1)[:, :self.robot.franka_num_dofs].contiguous()
            self.goal_dof_pos = self.dof_pos.clone().detach()
            
        # get net-force state tensor / Force direction is in the local frame! / Normal Force + Tangential Force
        # How to split them using normal direction?
        _force_tensor = gym.acquire_net_contact_force_tensor(self.sim)
        self.force_tensor = gymtorch.wrap_tensor(_force_tensor)

        # get force sensor tensor
        _force_sensor_tensor = gym.acquire_force_sensor_tensor(self.sim)
        self.force_sensor_tensor = gymtorch.wrap_tensor(_force_sensor_tensor)

        # get jacobian tensor for ik control
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        # jacobian entries corresponding to franka hand
        self.j_eef = jacobian[:, self.robot.franka_hand_index - 1, :, :7]
    

    def allocate_buffers(self):
        # allocate buffers
        self.single_img_obs_buf = torch.zeros(
            self.single_img_obs_dim, dtype=torch.float32, device=self.device)
        self.single_proprioception_buf = torch.zeros(
            self.single_proprioception_dim, dtype=torch.float32, device=self.device)
        self.observation_dict = {
            'image': self.single_img_obs_buf,
            'proprioception': self.single_proprioception_buf
        }
        self.rewards_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.steps_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.success_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        # self.hand_pose_6D_buf = self.hand_initial_pose_6D.repeat((self.num_envs, 1))
        
        # computation buffers (intermidiate variable no need to reset)
        self.delta_actions_6D = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self.step_info = {}
        self.reach_success_checker = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_world_reset_poses_full = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)
        self.move_obj_world_reset_poses_full = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)
        self.prev_target_hand_world_pos_dis_full = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # self.last_observation = torch.zeros((self.num_envs, self.single_observation_dims), dtype=torch.float32, device=self.device)
        # self.step_actions = torch.zeros((self.interval_steps, self.num_envs, 7), dtype=torch.float32, device=self.device)
        # self.lift_num_steps = int(self.lift_height / DeltaP * 10)
        # self.lift_action_step = self.lift_height / self.lift_num_steps

    
    def reset_buffers(self, reset_ids):
        # reset the buffer
        self.observation_dict['image'][reset_ids] = 0.
        self.observation_dict['proprioception'][reset_ids] = 0.
        self.rewards_buf[reset_ids] = 0.
        self.steps_buf[reset_ids] = 0
        self.reset_buf[reset_ids] = 0
        self.success_buf[reset_ids] = 0
        self.reach_success_checker[reset_ids] = 0

        # Update initial observation buffer
        hand_world_poses, _ = get_pose(self.rb_states, self.all_ids["rb"]["gripper"][reset_ids])
        target_world_poses, _ = get_pose(self.rb_states, self.all_ids["rb"][self.object_name[0]][reset_ids])
        move_obj_world_poses, _ = get_pose(self.rb_states, self.all_ids["rb"][self.object_name[0]][reset_ids])
        self.prev_target_hand_world_pos_dis_full[reset_ids] = torch.norm(hand_world_poses[:, :3] - target_world_poses[:, :3], p=2, dim=1)
        self.target_world_reset_poses_full[reset_ids] = target_world_poses.clone()
        self.move_obj_world_reset_poses_full[reset_ids] = move_obj_world_poses.clone()
        # Goal dof_pos as the initial default dof_pos!
        self.goal_dof_pos[reset_ids] = self.dof_pos[reset_ids]

        # self.nocontact_buf[reset_ids] = 0
        # self.env_stages[reset_ids] = 0
        # self.hand_pose_6D_buf[reset_ids] = self.hand_initial_pose_6D


    def refresh_request_tensors(self):
        # refresh tensors
        gym.refresh_actor_root_state_tensor(self.sim)
        gym.refresh_rigid_body_state_tensor(self.sim)
        gym.refresh_dof_state_tensor(self.sim)
        gym.refresh_net_contact_force_tensor(self.sim)
        gym.refresh_jacobian_tensors(self.sim)
        gym.refresh_mass_matrix_tensors(self.sim)
        # gym.refresh_force_sensor_tensor(self.sim)
    

    def update_viewer(self):
        # update viewer
        gym.step_graphics(self.sim)
        gym.sync_frame_time(self.sim)
        gym.render_all_camera_sensors(self.sim)
        if self.rendering: gym.draw_viewer(self.viewer, self.sim, False)


    def stepsimulation(self): # TODO: not fetch results at each time
        gym.simulate(self.sim)
        gym.fetch_results(self.sim, True)
        self.refresh_request_tensors()
        self.update_viewer()

    
    def random_actions(self):
        return torch.tensor(self.rng.choice(self.action_range, self.action_shape), dtype=torch.float32, device=self.device)


    def control_ik(self, dpose):
        damping = 0.05
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u


    def convert_actions(self, index_actions): # convert int relative actions to real absolute positions
        # Convert model actions (0, 1, 2) to real action
        delta_actions = index_actions[:, :6] - 1 # from (0, 1, 2) to (-1, 0, 1)
        # convert to real actions in the environment
        self.delta_actions_6D[:, :3] = delta_actions[:, :3] * DeltaP
        self.delta_actions_6D[:, 3:] = delta_actions[:, 3:] * DeltaR

        gripper_pose, _ = get_pose(self.rb_states, self.all_ids["rb"]['gripper'])
        self.gripper_goal_pos = gripper_pose[:, :3] + self.delta_actions_6D[:, :3]
        self.gripper_goal_quat = quat_mul(gripper_pose[:, 3:], quat_from_euler(self.delta_actions_6D[:, 3:]))
        self.gripper_act_Flag = index_actions[:, 6]

        # # Check moving action is within the moving space; TODO: Orientation check is not correct for 3D rotation (We need query IK to see whether it is a valid move?) 
        # hand_goal_pose_6D_buf = self.hand_pose_6D_buf.clone()
        # hand_goal_pose_6D_buf[:, :3] = self.hand_pose_6D_buf[:, :3] + self.delta_actions_6D[:, :3] if self.use_world_frame_obs \
        #                                else tf_apply(hand_cur_poses[:, 3:], hand_cur_poses[:, :3], self.delta_actions_6D[:, :3])
        # hand_goal_pose_6D_buf[:, 3:] += self.delta_actions_6D[:, 3:]
        # hand_goal_pose_6D_buf = hand_goal_pose_6D_buf.round(decimals=3) # Pytorch weird problem? Why tf_apply affects the other dimension 
        # goal_out_bound_idxs = ((hand_goal_pose_6D_buf > self.hand_move_bound_high) + (hand_goal_pose_6D_buf < self.hand_move_bound_low)).any(dim=1)
        # # reject out bound movement / mark reject actions to be -1 (maybe we do not need that)
        # self.delta_actions_6D[goal_out_bound_idxs, :] = 0.; self.raw_actions[goal_out_bound_idxs, :6] = -1.
        # # update hand_pose_6D_buf
        # self.hand_pose_6D_buf[:, :3] = self.hand_pose_6D_buf[:, :3] + self.delta_actions_6D[:, :3] if self.use_world_frame_obs \
        #                                else tf_apply(hand_cur_poses[:, 3:], hand_cur_poses[:, :3], self.delta_actions_6D[:, :3])
        # self.hand_pose_6D_buf[:, 3:] += self.delta_actions_6D[:, 3:]

        # steps_delta_actions = self.steps_tensor_6D * self.delta_actions_6D
        # steps_delta_pos, steps_delta_quat = \
        #     steps_delta_actions[:, :, 0:3], quat_from_euler(steps_delta_actions[:, :, 3:6])


    def update_franka_goal_dof(self):
        # compute gripper current position and orientation error
        gripper_cur_pose, _ = get_pose(self.rb_states, self.all_ids["rb"]['gripper'])
        pos_error = self.gripper_goal_pos - gripper_cur_pose[:, :3]
        ori_error = orientation_error(gripper_cur_pose[:, 3:], self.gripper_goal_quat)
        dpose = torch.cat([pos_error, ori_error], dim=1).unsqueeze(-1)

        # Deploy control based on type
        self.goal_dof_pos[:, :7] = self.dof_pos.squeeze(-1)[:, :7] + self.control_ik(dpose)

        # Gripper actions depend on distance between hand and box
        close_gripper = self.gripper_act_Flag == 2
        release_gripper = self.gripper_act_Flag == 0
        self.goal_dof_pos[close_gripper, 7:9] = self.to_torch([0.04, 0.04])
        self.goal_dof_pos[release_gripper, 7:9] = self.to_torch([0., 0.])

        set_dof(gym, self.sim, self.dof_pos, self.dof_vel, torch.arange(self.num_envs, device=self.device), self.goal_dof_pos)
        
        
    def pre_physics_step(self, actions):
        # check environments that need to be reset
        reset_ids = self.reset_buf.nonzero().squeeze(-1)
        if len(reset_ids) > 0: self.reset(reset_ids)
        # clamp actions and memorize it
        action_tensor = torch.clamp(actions, min(self.action_range), max(self.action_range)).to(self.device)
        self.raw_actions = action_tensor.clone()
        # DO NOT INPUT self.raw_actions in convert! It is in-place function!!
        self.convert_actions(action_tensor)

    
    def post_physics_step(self):
        self.steps_buf += 1
        self.compute_observations()
        self.compute_reward()
        # Priorly reset observation buffer for terminated environment to avoid closing the gripper at the first state
        # Why not reset envs here? Because we need done signal and all other reward information to record, we only need to reset obs for next initial action.
        reset_ids = self.reset_buf.nonzero().squeeze(-1)
        if len(reset_ids)>0: 
            self.observation_dict['image'][reset_ids] = 0.
            self.observation_dict['proprioception'][reset_ids] = 0.


    def query_scene(self):
        queried_scene_dict = {"prefixed":{}, # prefix is fix_base_link obj
                              "movable": {},
                              "unused": {}}

        # Choose from this_setup_obj
        if self.use_gpt: initial_chosen_objs = self.gpt.query_scene_objects(self.object_name)
        elif hasattr(self, "selected_obj_name"): initial_chosen_objs = self.selected_obj_name.copy()
        else: initial_chosen_objs = self.object_name

        prefixed_objs = ["table"]
        movable_objs = []
        unused_objs = self.object_name.copy()
        for obj_name in initial_chosen_objs:
            if obj_name not in self.object_name:
                print(f"This obj: {obj_name} is not in the available objects")
            else:
                movable_objs.append(obj_name)
                unused_objs.remove(obj_name)
                
        for name in prefixed_objs:
            queried_scene_dict["prefixed"][name] = [self.scene_obj_bbox[name], self.default_scene_dict[name]]
        
        for name in movable_objs:
            queried_scene_dict["movable"][name] = [self.scene_obj_bbox[name], None]

        for name in unused_objs: # Need to set their collision group
            queried_scene_dict["unused"][name] = [self.scene_obj_bbox[name], [*self.prepare_area, 1]]
            
        return queried_scene_dict


    def reset(self, reset_ids=None, post_correct=True, correct_Flags=[False, True]):
        if reset_ids is None:
            reset_ids = torch.arange(self.num_envs, device=self.device)
            self.cur_scene_dict = [deepcopy(self.default_scene_dict) for _ in range(self.num_envs)]
            self.prev_scene_dict = [deepcopy(self.default_scene_dict) for _ in range(self.num_envs)]

        # Query from LLM model to get self.cur_scene_dict
        body_root_ids = []; positions=[]; orientations=[]; linvels=[]; angvels=[]; scalings=[]
        body_env_ids = []; move_body_rb_ids = []; move_body_names = []; move_body_poss=[]; move_body_oris=[]; move_body_linvels=[]; move_body_angvels=[]
        # queried_scene_dict = self.questioner(give_input, reset_ids, self.prev_scene_dict, prev_diagnose)
        queried_scene_dict = self.query_scene()
        queried_scene_dicts = self.d_randomizer.fill_obj_poses([deepcopy(queried_scene_dict) for _ in range(self.num_envs)])

        for env_id in reset_ids:
            self.prev_scene_dict[env_id] = self.cur_scene_dict[env_id]
            self.cur_scene_dict[env_id] = queried_scene_dicts[env_id]
            iterate_scene_dict = {**self.cur_scene_dict[env_id]["prefixed"], **self.cur_scene_dict[env_id]["movable"], **self.cur_scene_dict[env_id]["unused"]}

            for obj_name, obj_status in iterate_scene_dict.items():
                body_root_ids.append(self.all_ids["root"][obj_name][env_id])
                obj_pose_info = obj_status[1] # obj_status: [obj_bbox, [obj_pos, obj_ori, obj_scaling]]
                positions.append(obj_pose_info[0])
                orientations.append(obj_pose_info[1])
                scalings.append(obj_pose_info[2])
                linvels.append([0., 0., 0.])
                angvels.append([0., 0., 0.])

            for move_obj_name, move_obj_status in self.cur_scene_dict[env_id]["movable"].items():
                body_env_ids.append(env_id)
                move_body_rb_ids.append(self.all_ids["rb"][move_obj_name][env_id])
                move_body_names.append(move_obj_name)
                move_obj_pose_info = move_obj_status[1] # obj_status: [obj_bbox, [obj_pos, obj_ori, obj_scaling]]
                move_body_poss.append(move_obj_pose_info[0])
                move_body_oris.append(move_obj_pose_info[1])
                move_body_linvels.append([0., 0., 0.])
                move_body_angvels.append([0., 0., 0.])

        body_root_ids = self.to_torch(body_root_ids, dtype=torch.long)
        positions = torch.stack(positions).to(self.device) # positions are list of tensors
        orientations = torch.stack(orientations).to(self.device)
        linvels = self.to_torch(linvels)
        angvels = self.to_torch(angvels)

        body_env_ids = self.to_torch(body_env_ids, dtype=torch.long)
        move_body_rb_ids = self.to_torch(move_body_rb_ids, dtype=torch.long)
        move_body_names = np.array(move_body_names)
        move_body_poss = torch.stack(move_body_poss).to(self.device)
        move_body_oris = torch.stack(move_body_oris).to(self.device)
        move_body_linvels = self.to_torch(move_body_linvels)
        move_body_angvels = self.to_torch(move_body_angvels)

        self.move_env_body_rb_ids = torch.stack([body_env_ids, move_body_rb_ids], dim=1)
        
        # Reset all objects
        if self.visualize_pc and hasattr(self, "pc_pos") and hasattr(self, "pc_color"):
            body_root_ids_list = [body_root_ids, self.all_ids["root"]["pc"][reset_ids].flatten()]
            positions_list = [self.prepare_area[0].repeat(len(body_root_ids), 1), self.pc_pos[reset_ids].squeeze(0)]
            orientations_list = [self.prepare_area[1].repeat(len(body_root_ids), 1), None]
            linvels_list = [linvels, None]
            angvels_list = [angvels, None]
            set_pose(gym, self.sim, self.root_tensor, body_root_ids_list, positions_list, orientations_list, linvels_list, angvels_list)
        else:
            set_pose(gym, self.sim, self.root_tensor, body_root_ids, positions, orientations, linvels, angvels)
            

        # Reset Robot Arm
        set_dof(gym, self.sim, self.dof_pos, self.dof_vel, reset_ids, 
                self.robot.default_dof_pos_tensor.repeat(len(reset_ids), 1), 
                self.robot.default_dof_vel_tensor.repeat(len(reset_ids), 1), 
                lower_limits=self.robot.franka_lower_limits, upper_limits=self.robot.franka_upper_limits)
        control_dof(gym, self.sim, self.dof_pos, reset_ids, self.robot.default_dof_pos_tensor.repeat(len(reset_ids), 1),
                    lower_limits=self.robot.franka_lower_limits, upper_limits=self.robot.franka_upper_limits)
        
        self.stepsimulation()

        # Post correct the scene
        if post_correct:
            self.failed_env_ids, self.check_table = self.post_corrector.handed_check_realiablity([body_env_ids, move_body_rb_ids, move_body_names, \
                                                                                              move_body_poss, move_body_oris, move_body_linvels, move_body_angvels], \
                                                                                              self.cur_scene_dict, force_check=correct_Flags[0], vel_check=correct_Flags[1],
                                                                                              gen_readable_table=False, verbose=False)

        # reset the buffer
        self.reset_buffers(reset_ids)

        return self.observation_dict


    def compute_observations(self):
        self.compute_observations_gpu()

    
    def compute_reward(self):
        self.compute_reward_reach()


    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        return self.step_arrange(actions)


    def compute_observations_gpu(self):
        # In the future, we can use a for loop to loop all fingers and compute each finger's observation then cat them together.
        # Maybe need a faster way to compute observations 
        # Fetch again to make sure the results be updated
        gym.fetch_results(self.sim, True)
        self.refresh_request_tensors()

        hand_world_poses, hand_world_vel = get_pose(self.rb_states, self.all_ids["rb"]["gripper"])
        rgb_images = self.to_torch(get_envs_images(self.sim, self.envs, self.camera_handles, gymapi.IMAGE_COLOR))
        depth_images = self.to_torch(get_envs_images(self.sim, self.envs, self.camera_handles, gymapi.IMAGE_DEPTH))
        depth_images = torch.clamp(depth_images, min=-10.0, max=10.0)
        
        if False and self.add_random_noise:
            pos_random_noise = torch_rand_float(*self.to_torch([-self.contact_noise_v, self.contact_noise_v]), shape=(self.num_envs, 3))
            force_random_noise = torch_rand_float(*self.to_torch([-self.force_noise_v, self.force_noise_v]), shape=(self.num_envs, 3))
            hand_world_poses[:, 0:3] += pos_random_noise
            rgb_images[:, :, :, 0:3] += pos_random_noise.unsqueeze(1).unsqueeze(1)
            
        # Start to concatenate all observations
        image_tensor = torch.cat([rgb_images, depth_images.unsqueeze(-1)], dim=3)
        self.single_img_obs_buf[:, :, :, :] = image_tensor.permute(0, 3, 1, 2) # (N, C, H, W)
        self.single_proprioception_buf[:, :self.robot.franka_num_dofs] = self.dof_pos
        self.single_proprioception_buf[:, self.robot.franka_num_dofs:self.robot.franka_num_dofs+7] = hand_world_poses
        self.single_proprioception_buf[:, self.robot.franka_num_dofs+7:] = hand_world_vel

        # Update observation buffer | Use LSTM+Linear or only use Linear (default Linear)
        # if self.time_seq_obs or self.use_transformer:
        #     self.observations_buf[:, :-1, :] = self.observations_buf[:, 1:, :].detach().clone() if self.num_envs==1 \
        #                                        else self.observations_buf[:, 1:, :] # faster popleft
        #     self.observations_buf[:, -1, :] = self.last_observation

        self.observation_dict.update({
            "image": self.single_img_obs_buf,
            "proprioception": self.single_proprioception_buf,
        })

        # Visualization

        if self.rendering: self.visualize_image_observation(env_id=0)


    def compute_reward_reach(self):

        target_world_cur_poses, _ = get_pose(self.rb_states, self.all_ids["rb"][self.object_name[0]])
        hand_world_cur_poses, _ = get_pose(self.rb_states, self.all_ids["rb"]["gripper"])
        all_moveable_obj_poses, _ = get_pose(self.rb_states, self.move_env_body_rb_ids[:, 1])
        robot_contact_force = get_force(self.force_tensor, self.all_ids["rb"]["robot_links"][:, 1])

        # TODO: check all targets are in the cabinet

        # compute l2 distance for [x, y] poses (need to scale them for better performancfe?) 
        # We need to use euler angle to compute orientation otherwise it is super unstable -> No, quaternion convert to euler might be very numerical unstable? 
        delta_pos_norm_full = torch.norm(hand_world_cur_poses[:, :3] - target_world_cur_poses[:, :3], p=2, dim=1)

        # We need a second order difference reward for this because we do not have cube pose state!
        pos_eps = 0.1
        pos_reward = self.pos_w * (1.0 / (delta_pos_norm_full + pos_eps) - 1.0 / (self.prev_target_hand_world_pos_dis_full + pos_eps))

        self.prev_target_hand_world_pos_dis_full[:] = delta_pos_norm_full.detach().clone() 

        action_penalty = self.act_w * torch.sum(self.raw_actions[:, :3].abs(), dim=1)
        
        rewards = pos_reward + action_penalty
        self.step_info.update({'pos_reward': pos_reward, 'act_penalty': action_penalty})

        # --- Compute reset indexes --- #
        penalty_r = 0.; success_r = 800.
        # terminated episodes that reached the max steps
        out_of_time_index = self.steps_buf >= self.maximum_steps
        # if torch.any(out_of_time_index): print("Out of time")
        # terminated episodes that the Target falls down to the ground
        target_fall_idxs = (target_world_cur_poses[:, 2] - self.target_world_reset_poses_full[:, 2]) < self.obj_fall_threshold
        # if torch.any(target_fall_idxs): print("Target Fall")
        # terminated episodes that insert into the table
        robot_collision_link_idxs = torch.any(robot_contact_force >= self.maximum_detect_force, dim=-1).squeeze(dim=-1)
        robot_collision_env_ids = self.all_ids["rb"]["robot_links"][robot_collision_link_idxs, 0].unique()
        # if len(robot_collision_env_ids) > 0: print("Hand insertion")
        # Success
        reach_succ = delta_pos_norm_full < self.reach_success_thred
        # if torch.any(reach_succ): print("Reach success")
        self.reach_success_checker[reach_succ] += 1
        success_idxs = self.reach_success_checker >= self.reach_success_steps

        self.success_buf[success_idxs] = 1
        rewards[out_of_time_index] -= penalty_r
        rewards[target_fall_idxs] -= penalty_r
        rewards[robot_collision_env_ids] -= penalty_r
        rewards[success_idxs] += success_r

        if self.auto_reset:
            self.reset_buf[robot_collision_env_ids] = 1
            self.reset_buf[target_fall_idxs] = 1
            self.reset_buf[out_of_time_index] = 1
            self.reset_buf[success_idxs] = 1

        # Update rewards
        self.rewards_buf[:] = rewards


    def step_arrange(self, actions: torch.Tensor):
        """Step the physics of the environment.
        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """
        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame; right now we step all robot arms at the same time
        for i in range(self.interval_steps):
            self.update_franka_goal_dof()
            self.stepsimulation()

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        return self.observation_dict, self.rewards_buf, self.reset_buf, self.step_info


    # def step_P2G(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
    #     """Step the physics of the environment.
    #     Args:
    #         actions: actions to apply
    #     Returns:
    #         Observations, rewards, resets, info
    #         Observations are dict of observations (currently only one member called 'obs')
    #     """

    #     # apply actions
    #     self.pre_physics_step(actions)

    #     # Act the terminal action episode
    #     # Terminal actions Question: we need to move this heuristic control outside the training loop
    #     moving_env_ids = (self.env_stages == 0).nonzero(as_tuple=False).squeeze(dim=-1)
    #     start_closing_env_ids = (self.raw_actions[:, -1] == 2).nonzero(as_tuple=False).squeeze(dim=-1)
    #     lifting_env_ids = (self.env_stages == 2).nonzero(as_tuple=False).squeeze(dim=-1)
    #     self.env_stages[start_closing_env_ids] = 1
    #     # Lifting envs should keep their stages as 2 (overwrite the previous start_closing_env_ids)
    #     self.env_stages[lifting_env_ids] = 2

    #     # step physics and render each frame
    #     for i in range(self.interval_steps):
    #         set_actor_ids = []
    #         set_actor_positions = []
    #         set_actor_orientations = []
            
    #         # Finger movement control (It is not necessary to write len(..._env_ids)>0 but just to improve readability)
    #         if len(moving_env_ids) > 0 and self.step_actions is not None:
    #             set_actor_ids.append(self.hand_idxs[moving_env_ids])
    #             set_actor_positions.append(self.step_actions[i][moving_env_ids, :3])
    #             set_actor_orientations.append(self.step_actions[i][moving_env_ids, 3:])
            
    #         closing_gripper_env_ids = (self.env_stages == 1).nonzero(as_tuple=False).squeeze(dim=-1)
    #         # Close gripper check
    #         if len(closing_gripper_env_ids) > 0:
    #             finger_red_force = get_force(self.force_tensor, self.finger_red_idxs[closing_gripper_env_ids]).norm(dim=1)
    #             finger_green_force = get_force(self.force_tensor, self.finger_green_idxs[closing_gripper_env_ids]).norm(dim=1)
    #             if self.real and self.real_env_id in closing_gripper_env_ids:
    #                 real_env_closing_idx = (closing_gripper_env_ids==self.real_env_id).nonzero(as_tuple=False).squeeze(dim=-1)
    #                 finger_red_force[real_env_closing_idx] = self.robot.get_force(self.red_finger)
    #                 finger_green_force[real_env_closing_idx] = self.robot.get_force(self.green_finger)

    #             force_satisfy_idxs = (finger_red_force > self.force_threshold) * \
    #                                  (finger_green_force > self.force_threshold)
    #             self.close_gripper_force_checker[closing_gripper_env_ids[force_satisfy_idxs]] += 1
    #             done_close_gripper_idxs = self.close_gripper_force_checker >= self.force_check_nums
    #             self.env_stages[done_close_gripper_idxs] = 2

    #             force_failure_idxs = ~force_satisfy_idxs
    #             claw_actions = torch.ones((len(closing_gripper_env_ids), self.hand_num_dofs), dtype=torch.float32, device=self.device) * JointP
    #             # Mask out actor that is already statisfied with the force check, stop continuing to close gripper
    #             claw_actions[force_satisfy_idxs, :] = 0.
    #             claw_target_pos = self.dof_pos[closing_gripper_env_ids, :] + claw_actions

    #             if False: # Close gripper but find not pass the force check
    #                 fail_grasp_obj_idxs = closing_gripper_env_ids[self.close_gripper_force_checker[closing_gripper_env_ids[force_failure_idxs]] > 0]
    #                 self.env_stages[fail_grasp_obj_idxs] = 0; self.close_gripper_force_checker[fail_grasp_obj_idxs] = 0
    #                 control_dof(gym, self.sim, self.dof_pos, fail_grasp_obj_idxs, self.default_dof_pos_tensor.expand(len(fail_grasp_obj_idxs), -1))

    #         # Lift action
    #         lifting_env_ids = (self.env_stages == 2).nonzero(as_tuple=False).squeeze(dim=-1)
    #         if len(lifting_env_ids) > 0:
    #             lifting_hand_idxs = self.hand_idxs[lifting_env_ids]
    #             hand_poses, _ = get_pose(self.root_tensor, lifting_hand_idxs)
    #             hand_pos, hand_ori = hand_poses[:, :3], hand_poses[:, 3:]
    #             hand_pos[:, 2] = hand_pos[:, 2] + self.lift_action_step
    #             set_actor_ids.append(lifting_hand_idxs); set_actor_positions.append(hand_pos); set_actor_orientations.append(hand_ori)

    #         # Set actor together here (Otherwise GPU pipeline will only consider the last set_pose function)
    #         if len(set_actor_ids) > 0:
    #             if self.real:
    #                 if self.real_env_id in moving_env_ids:
    #                     real_env_moving_idx = (moving_env_ids==self.real_env_id).nonzero(as_tuple=False).squeeze(dim=-1)
    #                     self.robot.set_pose_gripper(self.step_actions[i][self.real_env_id])
    #                 elif self.real_env_id in lifting_env_ids:
    #                     real_env_lifting_idx = (lifting_env_ids==self.real_env_id).nonzero(as_tuple=False).squeeze(dim=-1)
    #                     self.robot.set_pose_gripper(torch.cat([hand_pos[real_env_lifting_idx, :], hand_ori[real_env_lifting_idx, :]]))
                    
    #                 gripper_pose = self.robot.get_pose_gripper().squeeze(dim=0)
    #                 set_actor_ids.append(self.hand_idxs[self.real_env_id].unsqueeze(dim=0)); set_actor_positions.append(gripper_pose[:3]); set_actor_orientations.append(gripper_pose[3:])
    #             set_pose(gym, self.sim, self.root_tensor, set_actor_ids, set_actor_positions, set_actor_orientations)
            
    #         # Control actor DOF here together
    #         if len(closing_gripper_env_ids) > 0:
    #             if self.real:
    #                 if self.real_env_id in closing_gripper_env_ids:
    #                     real_env_closing_idx = (closing_gripper_env_ids==self.real_env_id).nonzero(as_tuple=False).squeeze(dim=-1)
    #                     self.robot.move_fin_pos(claw_target_pos[real_env_closing_idx, :])
    #                 claw_pos = self.robot.get_fin_pos()
    #                 claw_target_pos[real_env_closing_idx, :] = claw_pos
    #             control_dof(gym, self.sim, self.dof_pos, closing_gripper_env_ids, claw_target_pos, 
    #                 lower_limits=self.hand_lower_limits, upper_limits=self.hand_upper_limits)

    #         self.stepsimulation()

    #         # Check if lifting is done
    #         if len(lifting_env_ids) > 0:
    #             lifting_hand_idxs = self.hand_idxs[lifting_env_ids]
    #             hand_poses, _ = get_pose(self.root_tensor, lifting_hand_idxs)
    #             if self.real:
    #                 if self.real_env_id in lifting_env_ids:
    #                     real_env_lifting_idx = (lifting_env_ids==self.real_env_id).nonzero(as_tuple=False).squeeze(dim=-1)
    #                     hand_poses[real_env_lifting_idx, :] = self.robot.get_pose_gripper()

    #             hand_pos, hand_ori = hand_poses[:, :3], hand_poses[:, 3:]
    #             lifting_end_env_ids = lifting_env_ids[hand_pos[:, 2] - self.hand_initial_position[2] >= self.lift_height]
    #             self.env_stages[lifting_end_env_ids] = 3

    #     # compute observations, rewards, resets, ...
    #     self.post_physics_step()

    #     return self.observations_buf, self.rewards_buf, self.reset_buf, self.step_info
    

    def get_real_robot_states(self): # THIS NEEDS TO BE MODIFIED
        force = self.robot.get_force(self.green_finger)
        finger_pose = self.robot.get_pose_gripper()
        contact_info = self.robot.get_contact_points_single(self.green_finger)
        return force, finger_pose, contact_info
    

    def replay_real_trajctory(self):
        file_name = 'sim2real/06-20_17_40.pth'
        trajectory_buff = torch.load(file_name)
        finger_pose, force, contact_info = trajectory_buff["finger_pose"], trajectory_buff["force"], trajectory_buff["contact_info"]
        steps = len(trajectory_buff["force"])
        for i in range(steps):
            print(f"One step, {finger_pose[i][:3]}")
            root_positions, root_orientations = self.to_torch(finger_pose[i][:3]), self.to_torch(finger_pose[i][3:])
            set_pose(gym, self.sim, self.root_tensor, self.hand_idxs, root_positions, root_orientations)
            self.visualize_contact_force(contact_info_buffer=self.to_torch(contact_info[i].reshape(1, -1)))
            for j in range(200):
                self.stepsimulation()
            

    def step_manual(self):
        self.auto_reset = False;

        if self.visualize_pc:
            from PIL import Image
            test_img = Image.open("assets/image_dataset/scratch/test4.jpg").convert("RGB").resize((384, 384), Image.BILINEAR)
            _, index_masks, obj_phrase = self.dinosam.get_masks_labels_from_image(test_img)
            self.pc_pos, self.pc_color, self.pc_masks = self.scene_importer.get_pc_from_rgb([test_img], 
                                                                            intrinsic_matrix=self.scene_importer.intrinsic_matrix, 
                                                                            extrinsic_matrix=(self.camera_view_quats, self.camera_view_trans),
                                                                            masks=index_masks, 
                                                                            downsample=self.num_pcs)
            obj_bboxes, obj_center_poses = self.scene_importer.get_obj_pos_bbox_batch(self.pc_pos, self.pc_masks)

            # Change color for the points cloud
            for i in range(self.num_envs):
                env = self.envs[i]
                for j, pc_id in enumerate(self.all_ids["root"]["pc"][i, :]):
                    color = self.pc_color[i, j, :]
                    pc_handle = self.pc_handles[i][j]
                    gym.set_rigid_body_color(env, pc_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*color))
            
            # Draw the bounding box
            for i in range(self.num_envs):
                env = self.envs[i]
                obj_bbox = obj_bboxes[i, ...]
                obj_center_pose = obj_center_poses[i, ...]
                for j in range(obj_center_pose.shape[0]):
                    bbox_lines = self.compute_bounding_box_lines(obj_bbox[j])
                    self.visualize_bounding_box(env, bbox_lines, pose=None, refresh=False)
                    self.freeze_sim()

            self.bbox_lines = bbox_lines

        with keyboard.Events() as events:
            while (not hasattr(self, "viewer") or not gym.query_viewer_has_closed(self.viewer)):
                key = None
                event = events.get(0.0001)
                if event is not None:
                    if isinstance(event, events.Press) and hasattr(event.key, 'char'):
                        key = event.key.char
                if key == 'r': 
                    self.reset()
                    if len(self.failed_env_ids) > 0:
                        key = self.freeze_sim()
                        if key == 'f': self.failed_env_ids = []

                self.actions_tensor = torch.ones(self.action_dim, dtype=torch.float32, device=self.device)
                if key == 's': self.actions_tensor[:, 0] += 1
                if key == 'w': self.actions_tensor[:, 0] -= 1
                if key == 'd': self.actions_tensor[:, 1] += 1
                if key == 'a': self.actions_tensor[:, 1] -= 1
                if key == 'q': self.actions_tensor[:, 2] += 1
                if key == 'e': self.actions_tensor[:, 2] -= 1
                if key == 'u': self.actions_tensor[:, 3] += 1
                if key == 'i': self.actions_tensor[:, 3] -= 1
                if key == 'j': self.actions_tensor[:, 5] += 1
                if key == 'k': self.actions_tensor[:, 5] -= 1
                if key == 'm': self.actions_tensor[:, 6] += 1
                if key == ',': self.actions_tensor[:, 6] -= 1

                self.step(self.actions_tensor)

    
    def freeze_sim(self):
        if not hasattr(self, "viewer"): self._configure_viewer()
        previous_rendering = self.rendering
        self.rendering = True
        with keyboard.Events() as events:
            while not gym.query_viewer_has_closed(self.viewer):
                key = None
                event = events.get(0.0001)
                if event is not None:
                    if isinstance(event, events.Press) and hasattr(event.key, 'char'):
                        key = event.key.char
                if key == 'f': 
                    self.rendering = previous_rendering
                    return key
                if key == 'z': 
                    self.stepsimulation()
                else:
                    # step the viewer
                    self.update_viewer()
                
                rgb_images = self.to_torch(get_envs_images(self.sim, self.envs, self.camera_handles, gymapi.IMAGE_COLOR))
                depth_images = self.to_torch(get_envs_images(self.sim, self.envs, self.camera_handles, gymapi.IMAGE_DEPTH))
                depth_images = torch.clamp(depth_images, min=-10.0, max=10.0)

                image_tensor = torch.cat([rgb_images, depth_images.unsqueeze(-1)], dim=3).permute(0, 3, 1, 2) # (N, C, H, W)
                self.observation_dict.update({"image": image_tensor})

                # self.rgbd_images = get_envs_images_tensor(self.sim, self.camera_tensors)

                # self.visualize_image_observation(env_id=0)
    

    def visualize_image_observation(self, env_id=0):
        rgbd_image = self.observation_dict["image"]
        # Uint8 is necessary for the visualization of cv2!!!
        rgb_image = rgbd_image[env_id, :3, :, :].permute(1, 2, 0).to(torch.uint8) # (C, H, W) -> (H, W, C)
        depth_image = rgbd_image[env_id, 3, :, :]
        rgb_image = rgb_image.cpu().numpy()
        depth_image = depth_image.cpu().numpy()
        cv2_rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2_depth_image = np_scale(depth_image, 0, 255).astype(np.uint8)
        cv2.imshow('RGB', cv2.resize(cv2_rgb_image, (800, 400)))
        cv2.imshow('Alpha', cv2.resize(cv2_depth_image, (800, 400)))
        cv2.waitKey(1)

    
    def post_corrector_test(self, test_range=list(range(1, 19, 2)), save_path="results/post_corrector", check_force_diff=False, rendering=False):
        seed = 123456; iterations = 1000; total_trials = self.num_envs * iterations
        check_books = {"Both": (True, True), "Vel_only": (False, True), "Force_only": (True, False)}
        failure_rate_record = {}
        for num_obj in test_range:
            self.selected_obj_name = self.object_name[:num_obj]
            reset_failure_iters = {check_name: [] for check_name in check_books.keys()}

            for check_name, Flags in check_books.items():
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                
                failure_num = 0
                for i in range(iterations):
                    self.reset(correct_Flags=Flags)
                    failure_num += len(self.failed_env_ids)
                    if len(self.failed_env_ids) > 0: 
                        reset_failure_iters[check_name].append(i)
                        
                    if check_force_diff and check_name == "Force_only":
                        if len(self.failed_env_ids) > 0 and i not in reset_failure_iters["Both"]:
                            print(f"Force checker announces failure but others not at {i}")
                            print(self.check_table)
                            if rendering: self.freeze_sim()
                        elif len(self.failed_env_ids) == 0 and i in reset_failure_iters["Both"]:
                            print(f"Force checker announces success but others not at {i}")
                            print(self.check_table)
                            if rendering: self.freeze_sim()

                failure_rate = failure_num / total_trials
                print(f"Failure rate for {check_name} is {failure_rate}")

                if num_obj not in failure_rate_record.keys():
                    failure_rate_record[num_obj] = {}
                failure_rate_record[num_obj].update({check_name: failure_rate})
            
        torch.save(failure_rate_record, f"{save_path}/failure_rate_record.pth")


    def step_callibration(self, num_steps=20): # THIS NEEDS TO BE MODIFIED
        if self.real: assert self.num_envs == 2, "Number of envs should be equal to 2 for step callibration"

        self.auto_reset = False
        sim_env_id = self.real_env_id + 1 if self.real else 0
        action_waypoints = torch.ones((num_steps, ) + self.action_shape, dtype=torch.float, device=self.device)
        action_waypoints[:, :, 0] += 1 # y-axis negative direction movement
        sim_finger_contact_box, rob_finger_contact_box = [], []
        sim_finger_pos_box, rob_finger_pos_box = [], []
        sim_rob_finger_pose_diff_box = []

        time_stamp = '_'.join(str(datetime.datetime.now())[5:16].split())
        save_folder = f'sim2real/callibration/{time_stamp}'; 
        if not os.path.exists(save_folder): os.makedirs(save_folder, exist_ok=True)
        save_path = f"{save_folder}/{time_stamp}.pth"

        for i, action_tensor in enumerate(action_waypoints):
            observations_buf, rewards_buf, reset_buf, step_info = self.step(action_tensor)
            sim_finger_pose, _ = get_pose(self.root_tensor, self.hand_idxs[sim_env_id]); sim_finger_pose = sim_finger_pose.squeeze(dim=0)
            sim_contact_info = observations_buf[sim_env_id, -1, :6]
            if not self.use_world_frame_obs:
                sim_contact_info[:3] = tf_apply(sim_finger_pose[3:], sim_finger_pose[:3], sim_contact_info[:3])
                sim_contact_info[3:] = quat_apply(sim_finger_pose[3:], sim_contact_info[3:])
            sim_finger_pos_box.append(sim_finger_pose.tolist())
            sim_finger_contact_box.append(sim_contact_info.tolist())
            
            if self.real:
                rob_finger_pose  = self.robot.get_pose_gripper().squeeze(dim=0)
                rob_contact_info = observations_buf[self.real_env_id, -1, :6]
                finger_pos_diff = torch.norm(sim_finger_pose[:3]-rob_finger_pose[:3]) # Manually fix the quaternion +-1 problem
                finger_ori_diff = torch.min(torch.norm(sim_finger_pose[3:]-rob_finger_pose[3:]), torch.norm(sim_finger_pose[3:]-quat_conjugate(rob_finger_pose[3:])))
                finger_pose_diff = finger_pos_diff + finger_ori_diff

                if not self.use_world_frame_obs:
                    rob_contact_info[:3] = tf_apply(rob_finger_pose[3:], rob_finger_pose[:3], rob_contact_info[:3])
                    rob_contact_info[3:] = quat_apply(rob_finger_pose[3:], rob_contact_info[3:])
                rob_finger_pos_box.append(rob_finger_pose.tolist())
                rob_finger_contact_box.append(rob_contact_info.tolist())
                sim_rob_finger_pose_diff_box.append(finger_pose_diff.item())
        
        sim_finger_pos_box = self.to_torch(sim_finger_pos_box)
        sim_finger_contact_box = self.to_torch(sim_finger_contact_box)
        if self.real:
            rob_finger_pos_box = self.to_torch(rob_finger_pos_box)
            rob_finger_contact_box = self.to_torch(rob_finger_contact_box)
            sim_rob_finger_pose_diff_box = self.to_torch(sim_rob_finger_pose_diff_box)
        else:
            rob_finger_pos_box, rob_finger_contact_box, sim_rob_finger_pose_diff_box = None, None, None
        
        callibration_vis(sim_finger_pos_box, rob_finger_pos_box, sim_finger_contact_box, rob_finger_contact_box, sim_rob_finger_pose_diff_box, save_folder)

        callibration_dict = {'sim_finger_pos_box': sim_finger_pos_box,
                             'rob_finger_pos_box': rob_finger_pos_box,
                             'sim_finger_contact_box': sim_finger_contact_box,
                             'rob_finger_contact_box': rob_finger_contact_box,
                             'sim_rob_finger_pose_diff_box': sim_rob_finger_pose_diff_box}
        print(f"Save Trajectory to {save_path}")
        torch.save(callibration_dict, save_path)


    def get_asset_pose(self, name):
        return self.scene_asset[name][1]


    def visualize_contact_force(self, contact_info_buffer_world_frame, refresh=False): # Should write as each finger version
        if refresh: gym.clear_lines(self.viewer)
        normal_coef = 10
        lines_color = self.to_torch([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float32)
        # Concert local contact position to global

        for i, contact_info in enumerate(contact_info_buffer_world_frame):
            env = self.envs[i]
            force_marker_id_r, force_direct_id_r = self.force_marker_idxs_r[i].unsqueeze(dim=0), self.force_direct_idxs_r[i].unsqueeze(dim=0)
            force_marker_id_g, force_direct_id_g = self.force_marker_idxs_g[i].unsqueeze(dim=0), self.force_direct_idxs_g[i].unsqueeze(dim=0)
            red_contact_info, green_contact_info = contact_info[0:6], contact_info[6:12]
            red_contact_position_world, red_contact_force = red_contact_info[0:3], red_contact_info[3:6]
            green_contact_position_world, green_contact_force = green_contact_info[0:3], green_contact_info[3:6]
            

            # Add the contact location dot and the force direction
            translation = self.force_direct_dimension / 2
            if (red_contact_force == 0.).all():
                red_contact_position_world = self.prepare_area_position
                force_direct_position_r = self.prepare_area_position
                quat_v1_v2_r = self.prepare_area_orientation
            else:
                quat_v1_v2_r = quat_bet_vects(translation, red_contact_force)
                force_direct_position_r = tf_apply(quat_v1_v2_r, red_contact_position_world, translation)
            if (green_contact_force == 0.).all():
                green_contact_position_world = self.prepare_area_position
                force_direct_position_g = self.prepare_area_position
                quat_v1_v2_g = self.prepare_area_orientation
            else:
                quat_v1_v2_g = quat_bet_vects(translation, green_contact_force)
                force_direct_position_g = tf_apply(quat_v1_v2_g, green_contact_position_world, translation)
            # set_pose(gym, self.sim, self.root_tensor, force_direct_id_g, force_direct_position, quat_v1_v2)
            set_pose(gym, self.sim, self.root_tensor,
                    [force_marker_id_r, force_direct_id_r, force_marker_id_g, force_direct_id_g], 
                    [red_contact_position_world, force_direct_position_r, 
                     green_contact_position_world, force_direct_position_g],
                    [self.prepare_area_orientation, quat_v1_v2_r,
                     self.prepare_area_orientation, quat_v1_v2_g])
            # Old version to draw contact line
            # normalize_contact_force_green = green_contact_force / (green_contact_force.abs().max() * normal_coef)
            # green_line = torch.cat([green_contact_position_world, green_contact_position_world + normalize_contact_force_green])
            # gym.add_lines(self.viewer, env, 1, green_line, lines_color)


    def visualize_hand_axis(self, refresh=True):
        if refresh: gym.clear_lines(self.viewer)
        xyzAxisColor = self.to_torch([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float32)
        xyzAxis = self.to_torch([[0.1, 0., 0.], [0., 0.1, 0.], [0., 0., 0.1]], dtype=torch.float32)
        finger_red_world_poses, _ = get_pose(self.rb_states, self.finger_red_idxs)
        finger_green_world_poses, _ = get_pose(self.rb_states, self.finger_green_idxs)
        for i, env in enumerate(self.envs):
            red_position_world, red_orientation_world = finger_red_world_poses[i, :3], finger_red_world_poses[i, 3:]
            green_position_world, green_orientation_world = finger_green_world_poses[i, :3], finger_green_world_poses[i, 3:]
            red_xyz_axis_Wor = tf_apply(red_orientation_world, red_position_world, xyzAxis)
            green_xyz_axis_Wor = tf_apply(green_orientation_world, green_position_world, xyzAxis)
            red_xyz_axis_lines = torch.cat([red_position_world.expand_as(red_xyz_axis_Wor), red_xyz_axis_Wor], dim=1)
            green_xyz_axis_lines = torch.cat([green_position_world.expand_as(green_xyz_axis_Wor), green_xyz_axis_Wor], dim=1)
            gym.add_lines(self.viewer, env, 6, torch.cat([red_xyz_axis_lines, green_xyz_axis_lines], dim=0), xyzAxisColor.repeat((2, 1)))


    def visualize_camera_axis(self, cam_trans, cam_ori):
        xyzAxisColor = self.to_torch([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float32)
        xyzAxis = self.to_torch([[0.1, 0., 0.], [0., 0.1, 0.], [0., 0., 0.1]], dtype=torch.float32)
        for i, env in enumerate(self.envs):
            xyz_axis_Wor = tf_apply(cam_ori, cam_trans, xyzAxis)
            xyz_axis_lines = torch.cat([cam_trans.expand_as(xyz_axis_Wor), xyz_axis_Wor], dim=1)
            gym.add_lines(self.viewer, env, 3, xyz_axis_lines.cpu().numpy(), xyzAxisColor.cpu().numpy()) # use_gpu_pipeline will require .cpu.numpy()


    def visualize_filter_area(self, refresh=True):
        if refresh: gym.clear_lines(self.viewer)
        red_lines_color = self.to_torch([1., 0., 0.], dtype=torch.float32)
        green_lines_color = self.to_torch([0., 1., 0.], dtype=torch.float32)
        finger_red_world_poses, _ = get_pose(self.rb_states, self.finger_red_idxs)
        finger_green_world_poses, _ = get_pose(self.rb_states, self.finger_green_idxs)
        for i, env in enumerate(self.envs):
            red_position_world, red_orientation_world = finger_red_world_poses[i, :3], finger_red_world_poses[i, 3:]
            green_position_world, green_orientation_world = finger_green_world_poses[i, :3], finger_green_world_poses[i, 3:]
            red_lines_box = self.compute_bounding_box_lines(env, self.filter_area_start_end)
            red_lines_world_box = tf_apply(red_orientation_world, red_position_world, red_lines_box.reshape(-1, 3)).reshape(-1, 6)
            gym.add_lines(self.viewer, env, len(red_lines_world_box), red_lines_world_box, red_lines_color.repeat((len(red_lines_world_box), 1)))
            green_lines_box = self.compute_bounding_box_lines(env, self.filter_area_start_end)
            green_lines_world_box = tf_apply(green_orientation_world, green_position_world, green_lines_box.reshape(-1, 3)).reshape(-1, 6)
            gym.add_lines(self.viewer, env, len(green_lines_world_box), green_lines_world_box, green_lines_color.repeat((len(green_lines_world_box), 1)))


    def visualize_bounding_box(self, env, bbox_lines, pose=None, refresh=True):
        if refresh: gym.clear_lines(self.viewer)
        lines_color = self.to_torch([1., 0., 0.], dtype=torch.float32)
        
        if pose is None: 
            lines_world_box = bbox_lines.reshape(-1, 6)
        else: # Transform the bbox (bbox is in the local frame)
            position_world, orientation_world = pose
            lines_world_box = tf_apply(orientation_world, position_world, bbox_lines).reshape(-1, 6)
        gym.add_lines(self.viewer, env, len(lines_world_box), lines_world_box.cpu().numpy(), lines_color.repeat((len(lines_world_box), 1)).cpu().numpy())


    def compute_bounding_box_lines(self, start_end_coordinate: torch.Tensor):
        start_cor, end_cor = start_end_coordinate
        x1, y1, z1 = start_cor
        x2, y2, z2 = end_cor
        corner_points = self.to_torch([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
        up_points = torch.cat([corner_points, torch.ones((4, 1), device=self.device)*z1], dim=1)
        bot_points = torch.cat([corner_points, torch.ones((4, 1), device=self.device)*z2], dim=1)
        lines_box = []
        for i in range(len(up_points)-1):
            lines_box.append(torch.cat([up_points[i], up_points[i+1]]))
            lines_box.append(torch.cat([bot_points[i], bot_points[i+1]]))
            lines_box.append(torch.cat([up_points[i], bot_points[i]]))
        lines_box.append(torch.cat([up_points[0], up_points[-1]]))
        lines_box.append(torch.cat([bot_points[0], bot_points[-1]]))
        lines_box.append(torch.cat([up_points[-1], bot_points[-1]]))
        return torch.cat(lines_box, dim=0).reshape(-1, 3)


    def create_target_asset(self, target_asset_file, fix_base_link=False, disable_gravity=False, density=None, use_default_cube=False, convex_dec=False, target_dim=[0.05]*3):
        if use_default_cube:
            return self.create_box_asset(dimension=target_dim, fix_base_link=fix_base_link, disable_gravity=disable_gravity, density=density)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = fix_base_link
        asset_options.disable_gravity = disable_gravity
        if density is not None: asset_options.density = density
        if convex_dec: 
            asset_options.convex_decomposition_from_submeshes = True
            asset_options.vhacd_enabled = True
        return gym.load_asset(self.sim, self.asset_root, target_asset_file, asset_options)


    def create_box_asset(self, dimension=[0.5, 0.5, 0.5], fix_base_link=False, disable_gravity=False, density=None):
        box_dimension = gymapi.Vec3(*dimension)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = fix_base_link
        asset_options.disable_gravity = disable_gravity
        if density is not None: asset_options.density = density
        return gym.create_box(self.sim, box_dimension.x, box_dimension.y, box_dimension.z, asset_options)
    

    def create_GM_asset(self, radius=0.02, type_name='sphere', fix_base_link=False, disable_gravity=False, density=None):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = fix_base_link
        asset_options.disable_gravity = disable_gravity
        if density is not None: asset_options.density = density
        if type_name =='sphere': return gym.create_sphere(self.sim, radius, asset_options)
        elif type_name =='polybead': 
            return gym.load_asset(self.sim, self.asset_root, os.path.join(self.urdf_folder, "polybead")+".urdf", asset_options)
        else: return gym.create_box(self.sim, radius, radius, radius, asset_options)


    def close(self):
        if self.rendering: gym.destroy_viewer(self.viewer)
        gym.destroy_sim(self.sim)


    def to_torch(self, x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=self.device, requires_grad=False)


if __name__ == '__main__':
    args = gymutil.parse_arguments(description="Isaac Gym for Sandem")
    args.graphics_device_id = 0 # might need to change in different computer
    args.object_name = 'cube'
    args.task = 'P2G'
    args.use_gpu_pipeline = True
    args.rendering = True
    args.time_seq_obs = False
    args.use_transformer = True
    args.sequence_len = 5
    args.num_envs = 1
    args.draw_contact = True
    args.filter_contact = True
    args.include_target_obs = False
    args.use_world_frame_obs = False
    args.random_target_init = False
    args.random_goal = True
    args.random_target = False
    args.include_finger_vel = False
    args.use_abstract_contact_obs = False
    args.use_contact_torque = False
    args.use_2D_contact = False
    args.pos_weight = 1.
    args.ori_weight = 0.
    args.act_weight = 0.
    args.real = False
    args.visualize_pc = True
    args.num_pcs = 20000


    args.use_gpt = False

    env = RoboSensaiEnv(args)
    # env.post_corrector_test(rendering=False)
    env.step_manual()
    env.close()