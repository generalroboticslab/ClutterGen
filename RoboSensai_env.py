from typing import Dict, Any, Tuple
import time, datetime
from copy import deepcopy

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym_utils import *
from isaacgym.torch_utils import tf_inverse, tf_apply, quat_mul, quat_apply, tf_combine

import math
import os
import numpy as np
import torch
from Robot import Franka
from domain_randomizer import DomainRandomizer
from llm_query import GPT
from post_corrector import PostCorrector

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
ControlFreq = 6 # 6Hz control
IntervalSteps = int(1/DT / ControlFreq) # 10 steps for one control / # Interval steps should larger than or equal 2
Min_delta_x, Max_delta_x = -0.25, 0. # Try not give 0 in any dimension
Min_delta_y, Max_delta_y = -0.15, 0.15
Min_delta_z, Max_delta_z = 0., 0.3
Min_eef_angle, Max_eef_angle = 0, 2*np.pi - DeltaR # [0, 345] degree

# set random seed
np.random.seed(42)
torch.set_printoptions(precision=8, sci_mode=False)


class HandemEnv:    
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
        self.object_name = ['cube_arrow', 'cuboid', 'bowl']
        self.object_name_index = {}
        for i, name in enumerate(self.object_name):
            self.object_name_index[name] = i
        
        self.add_gms = self.args.add_gms if hasattr(self.args, "add_gms") else False
        self.add_sides_shelf = self.args.add_sides_shelf if hasattr(self.args, "add_sides_shelf") else False
        self.num_gms = self.args.num_gms if hasattr(self.args, "num_gms") else 1000
        

        # Visualization Configs

        # Model Configs

        # Actions & Observations

        # Action dimension

        # Observation dimension
        
        # Single_observation_dimension (image)

        # Create Sim and Camera
        self._configure_sim()
        # Create all assets
        self._configure_asset()
        # Create robot arm
        self._configure_robot()
        # Create domain randomizer
        self.d_randomizer = DomainRandomizer(device=self.device)
        # Create GPT 
        self.gpt = GPT()
        # Create post corrector
        self.post_corrector = PostCorrector(self, device=self.device)
        # Create environments and allocate buffers
        self.create_envs()
        
        self._link_states_tensor()

        # All environments simulation 1 s to ensure the reset successfully
        sim_sec(self.sim, 1)

        # self.allocate_buffers()
        
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
        self.sim_params.physx.contact_offset = 0.001
        self.sim_params.physx.friction_offset_threshold = 0.001
        self.sim_params.physx.friction_correlation_distance = 0.0005
        self.sim_params.physx.num_threads = self.args.num_threads
        self.sim_params.physx.use_gpu = self.args.use_gpu
        # Must CC All substeps! Otherwise gym sometimes give 0 contact info!
        self.sim_params.physx.contact_collection = gymapi.CC_ALL_SUBSTEPS
        # self.sim_params.physx.friction_offset_threshold = 0.001
        # self.sim_params.physx.friction_correlation_distance = 0.0005
        
        # create sim
        self.sim = gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, gymapi.SIM_PHYSX, self.sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")
    

    def _configure_viewer(self):
        self.viewer = gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")
    

    def _configure_camera(self):
        # point camera at middle env
        table_pose = self.get_asset_pose('table')
        position = table_pose.p
        x, y, z = position.x, position.y, position.z
        cam_pos = gymapi.Vec3(x+0.8, y, z+1.2)
        cam_target = gymapi.Vec3(x-1, y, -0.5)
        middle_env = self.all_ids['envs'][self.num_envs // 2 + self.num_per_row // 2]
        gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)
    

    def _configure_ground(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        gym.add_ground(self.sim, plane_params)


    def _configure_asset(self, scene_dict=None):
        # TODO: We should have a scene_dict to record the scene states
        self.scene_asset = {}
        table_dims = [0.6, 1.0, 0.4]
        table_asset = self.create_box_asset(table_dims, fix_base_link=True)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.5, 0.0, table_dims[2]/2)

        self.obj_assets = self.object_name_index.copy()
        self.obj_bbox = self.object_name_index.copy()
        for name in self.obj_assets:
            obj_asset = self.create_target_asset(target_asset_file=os.path.join(self.urdf_folder, name)+".urdf")
            obj_pose = deepcopy(table_pose)
            self.scene_asset[name] = [obj_asset, obj_pose]
            self.obj_bbox[name] = getMeshBbox(os.path.join(self.asset_root, self.mesh_folder, name, 'collision')+'.obj')

        # Manual Update here, later will need to merge in the main loop
        self.scene_asset.update(
            {"table": [table_asset, table_pose]}
        )

        self.obj_bbox.update({
            "table": table_dims,
        })

        # Should follow pre-loading scene_dict to add or delete object
        self.all_ids = {"envs": [], "robot": [], "gripper": []}
        for key in self.scene_asset.keys():
            self.all_ids[key] = []


    def _configure_robot(self, robot_name="franka"):
        self.robot = Franka(sim=self.sim, asset_root=self.asset_root)
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
        self.prepare_area = gymapi.Transform()
        self.prepare_area.p.x = 0.
        self.prepare_area.p.y = 100.
        self.prepare_area.p.z = 0.2
        self.prepare_area = transform2list(self.prepare_area)

        # object position dict
        self.default_scene_dict = {}

        for i in range(self.num_envs):
            # create env
            env = gym.create_env(self.sim, env_lower, env_upper, self.num_per_row)
            self.all_ids['envs'].append(env)

            for obj_name in self.scene_asset.keys():
                if obj_name in ['robot']: continue
                obj_asset, obj_pose = self.scene_asset[obj_name]
                obj_handle = gym.create_actor(env, obj_asset, obj_pose, obj_name, i, 0)
                obj_id = gym.get_actor_index(env, obj_handle, gymapi.DOMAIN_SIM)
                self.all_ids[obj_name].append(obj_id)

            # add robot
            franka_asset, franka_pose = self.scene_asset["robot"]
            franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 2)
            franka_id = gym.get_actor_index(env, franka_handle, gymapi.DOMAIN_SIM)
            self.all_ids['robot'].append(franka_id)
            # set dof properties
            gym.set_actor_dof_properties(env, franka_handle, self.robot.franka_dof_props)
            # set initial dof states
            gym.set_actor_dof_states(env, franka_handle, self.robot.default_dof_state, gymapi.STATE_ALL)
            # set initial position targets
            gym.set_actor_dof_position_targets(env, franka_handle, self.robot.default_dof_state["pos"])
            # get inital hand pose
            hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
            hand_pose = gym.get_rigid_transform(env, hand_handle)
            hand_id = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
            self.all_ids['gripper'].append(hand_id)
        

        for obj_name in self.scene_asset.keys():
            obj_asset, obj_pose = self.scene_asset[obj_name]
            scaling = 1 if obj_name not in ['robot', 'gripper'] else None
            self.default_scene_dict[obj_name] = [*transform2list(obj_pose), 1]


        self.movable_obj_name = {}
        for obj_name, obj_ids in self.all_ids.items():
            if obj_name == "envs": continue # envs are objects not ids
            self.all_ids[obj_name] = self.to_torch(obj_ids, dtype=torch.long)
            if obj_name not in ["envs", "gripper"]:
                self.movable_obj_name[obj_name] = []

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
            self.dof_pos = self.dof_states[:, 0].view(self.num_envs, self.robot.franka_num_dofs).contiguous()
            self.dof_vel = self.dof_states[:, 1].view(self.num_envs, self.robot.franka_num_dofs).contiguous()
            
        # get net-force state tensor / Force direction is in the local frame! / Normal Force + Tangential Force
        # How to split them using normal direction?
        _force_tensor = gym.acquire_net_contact_force_tensor(self.sim)
        self.force_tensor = gymtorch.wrap_tensor(_force_tensor)

        # get force sensor tensor
        _force_sensor_tensor = gym.acquire_force_sensor_tensor(self.sim)
        self.force_sensor_tensor = gymtorch.wrap_tensor(_force_sensor_tensor)

        # action interpolation tensor, interpolation value to be the same shape as control tensor
        step_value = torch.linspace(0, 1, steps=self.interval_steps).view(self.interval_steps, 1, 1)
        self.steps_tensor_6D = step_value.expand(self.interval_steps, self.num_envs, 6).to(self.device)
    

    def allocate_buffers(self):
        # allocate buffers
        self.observations_buf = torch.zeros(
            self.observation_shape, device=self.device, dtype=torch.float)
        self.rewards_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.steps_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.success_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.nocontact_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.hand_pose_6D_buf = self.hand_initial_pose_6D.repeat((self.num_envs, 1))
        
        # computation buffers (intermidiate variable no need to reset)
        self.last_observation = torch.zeros((self.num_envs, self.single_observation_dims), dtype=torch.float32, device=self.device)
        self.delta_actions_6D = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self.step_actions = torch.zeros((self.interval_steps, self.num_envs, 7), dtype=torch.float32, device=self.device)
        self.lift_num_steps = int(self.lift_height / DeltaP * 10)
        self.lift_action_step = self.lift_height / self.lift_num_steps

    
    def reset_buffers(self, reset_ids):
        # reset the buffer
        self.observations_buf[reset_ids] = 0. # Automatically expand to change the whole row to be zero
        self.rewards_buf[reset_ids] = 0.
        self.steps_buf[reset_ids] = 0
        self.reset_buf[reset_ids] = 0
        self.success_buf[reset_ids] = 0
        self.nocontact_buf[reset_ids] = 0
        self.close_gripper_force_checker[reset_ids] = 0
        self.env_stages[reset_ids] = 0
        self.hand_pose_6D_buf[reset_ids] = self.hand_initial_pose_6D


    def refresh_request_tensors(self):
        # refresh tensors
        gym.refresh_actor_root_state_tensor(self.sim)
        gym.refresh_rigid_body_state_tensor(self.sim)
        gym.refresh_dof_state_tensor(self.sim)
        gym.refresh_net_contact_force_tensor(self.sim)
        # gym.refresh_force_sensor_tensor(self.sim)
        # gym.refresh_jacobian_tensors(sim)
        # gym.refresh_mass_matrix_tensors(sim)
    

    def update_viewer(self):
        # update viewer
        gym.step_graphics(self.sim)
        gym.draw_viewer(self.viewer, self.sim, False)
        gym.sync_frame_time(self.sim)


    def stepsimulation(self): # TODO: not fetch results at each time
        gym.simulate(self.sim)
        gym.fetch_results(self.sim, True)
        self.refresh_request_tensors()
        if self.rendering: self.update_viewer()

    
    def random_actions(self):
        return torch.tensor(self.rng.choice(self.action_range, self.action_shape), dtype=torch.float32, device=self.device)


    def convert_actions(self, index_actions): # convert int relative actions to real absolute positions
        # Convert model actions (0, 1, 2) to real action
        delta_actions = index_actions[:, :6] - 1 # from (0, 1, 2) to (-1, 0, 1)
        # convert to real actions in the environment
        self.delta_actions_6D[:, :3] = delta_actions[:, :3] * DeltaP
        self.delta_actions_6D[:, 5] = delta_actions[:, 5] * DeltaR

        # self.delta_actions_6D[:, :3] = delta_actions * DeltaP
        # self.delta_actions_6D[:, 3:] = delta_actions * DeltaR

        hand_cur_poses, _ = get_pose(self.root_tensor, self.hand_idxs)
        if self.real: hand_cur_poses[self.real_env_id, :] = self.robot.get_pose_gripper() # hand base pose is same as finger pose in the simulation
        hand_cur_positions, hand_cur_quats = hand_cur_poses[:, :3], hand_cur_poses[:, 3:7]

        # Check moving action is within the moving space; TODO: Orientation check is not correct for 3D rotation (We need query IK to see whether it is a valid move?) 
        hand_goal_pose_6D_buf = self.hand_pose_6D_buf.clone()
        hand_goal_pose_6D_buf[:, :3] = self.hand_pose_6D_buf[:, :3] + self.delta_actions_6D[:, :3] if self.use_world_frame_obs \
                                       else tf_apply(hand_cur_poses[:, 3:], hand_cur_poses[:, :3], self.delta_actions_6D[:, :3])
        hand_goal_pose_6D_buf[:, 3:] += self.delta_actions_6D[:, 3:]
        hand_goal_pose_6D_buf = hand_goal_pose_6D_buf.round(decimals=3) # Pytorch weird problem? Why tf_apply affects the other dimension 
        goal_out_bound_idxs = ((hand_goal_pose_6D_buf > self.hand_move_bound_high) + (hand_goal_pose_6D_buf < self.hand_move_bound_low)).any(dim=1)
        # reject out bound movement / mark reject actions to be -1 (maybe we do not need that)
        self.delta_actions_6D[goal_out_bound_idxs, :] = 0.; self.raw_actions[goal_out_bound_idxs, :6] = -1.
        # update hand_pose_6D_buf
        self.hand_pose_6D_buf[:, :3] = self.hand_pose_6D_buf[:, :3] + self.delta_actions_6D[:, :3] if self.use_world_frame_obs \
                                       else tf_apply(hand_cur_poses[:, 3:], hand_cur_poses[:, :3], self.delta_actions_6D[:, :3])
        self.hand_pose_6D_buf[:, 3:] += self.delta_actions_6D[:, 3:]

        steps_delta_actions = self.steps_tensor_6D * self.delta_actions_6D
        steps_delta_pos, steps_delta_quat = \
            steps_delta_actions[:, :, 0:3], quat_from_euler(steps_delta_actions[:, :, 3:6])
        
        steps_hand_cur_pos = hand_cur_positions.expand_as(steps_delta_pos)
        steps_hand_cur_quat = hand_cur_quats.expand_as(steps_delta_quat)
        
        if self.use_world_frame_obs:
            steps_hand_goal_pos = steps_delta_pos + steps_hand_cur_pos
            steps_hand_goal_quat = quat_mul(steps_delta_quat, steps_hand_cur_quat)
        else:
            steps_hand_goal_pos = tf_apply(steps_hand_cur_quat, steps_hand_cur_pos, steps_delta_pos)
            steps_hand_goal_quat = quat_mul(steps_hand_cur_quat, steps_delta_quat)
        # print(steps_hand_goal_pos[-1, :])
        # Quaternion multiplication is reverse order as rotation matrix multiplication! R1*R2 -> q2*q1???? 
        # It seems that this quat_mul in torch_utils has fixed this rotation prob where R1*R2 -> q1*q2
        self.step_actions[:, :, :3] = steps_hand_goal_pos; self.step_actions[:, :, 3:] = steps_hand_goal_quat
        
        
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
        if len(reset_ids)>0: self.observations_buf[reset_ids] = 0.


    def query_scene(self):
        queried_scene_dict = {"prefixed":{}, # prefix is fix_base_link obj
                              "movable": {},
                              "unused": {}}

        # Choose from this_setup_obj
        prefixed_objs = ["table"]
        # gptChosen_objs = self.gpt.query_scene_objects(self.object_name)
        gptChosen_objs = self.object_name
        movable_objs = []
        unused_objs = self.object_name.copy()
        for obj_name in gptChosen_objs:
            if obj_name not in self.object_name:
                print(f"This obj: {obj_name} is not in the available objects")
            else:
                movable_objs.append(obj_name)
                unused_objs.remove(obj_name)
                
        for name in prefixed_objs:
            queried_scene_dict["prefixed"][name] = [self.obj_bbox[name], self.default_scene_dict[name]]
        
        for name in movable_objs:
            queried_scene_dict["movable"][name] = [self.obj_bbox[name], None]

        for name in unused_objs: # Need to set their collision group
            queried_scene_dict["unused"][name] = [self.obj_bbox[name], [*self.prepare_area, 1]]
            
        return queried_scene_dict


    def reset(self, reset_ids=None):
        if reset_ids is None:
            reset_ids = torch.arange(self.num_envs, device=self.device)
            self.cur_scene_dict = [deepcopy(self.default_scene_dict) for _ in range(self.num_envs)]
            self.prev_scene_dict = [deepcopy(self.default_scene_dict) for _ in range(self.num_envs)]

        # Query from LLM model to get self.cur_scene_dict
        env_ids = []; body_ids = []; body_names = []; positions=[]; orientations=[]; linvels=[]; angvels=[]; scalings=[]
        # queried_scene_dict = self.questioner(give_input, reset_ids, self.prev_scene_dict, prev_diagnose)
        queried_scene_dict = self.query_scene()
        queried_scene_dicts = self.d_randomizer.fill_obj_poses([deepcopy(queried_scene_dict) for _ in range(self.num_envs)])

        for env_id in reset_ids:
            self.prev_scene_dict[env_id] = self.cur_scene_dict[env_id]
            self.cur_scene_dict[env_id] = queried_scene_dicts[env_id]
            iterate_scene_dict = {**self.cur_scene_dict[env_id]["prefixed"], **self.cur_scene_dict[env_id]["movable"], **self.cur_scene_dict[env_id]["unused"]}

            for obj_name, obj_status in iterate_scene_dict.items():
                env_ids.append(env_id)
                body_ids.append(self.all_ids[obj_name][env_id])
                body_names.append(obj_name)
                obj_pose_info = obj_status[1] # obj_status: [obj_bbox, [obj_pos, obj_ori, obj_scaling]]
                positions.append(obj_pose_info[0])
                orientations.append(obj_pose_info[1])
                scalings.append(obj_pose_info[2])
                linvels.append([0., 0., 0.])
                angvels.append([0., 0., 0.])

        env_ids = self.to_torch(env_ids, dtype=torch.long)
        body_ids = self.to_torch(body_ids, dtype=torch.long)
        body_names = np.array(body_names)
        positions = self.to_torch(positions)
        orientations = self.to_torch(orientations)
        linvels = self.to_torch(linvels)
        angvels = self.to_torch(angvels)
        
        set_pose(gym, self.sim, self.root_tensor, body_ids, positions, orientations, linvels, angvels)
        
        self.stepsimulation()
        failed_env_ids = self.post_corrector.handed_check_realiablity([env_ids, body_ids, body_names, positions, orientations, linvels, angvels], self.cur_scene_dict)


    # def reset(self, reset_ids=None):
        # if reset_ids is None: # reset all envs
        #     reset_ids = torch.arange(self.num_envs, device=self.device)
        #     # setup a full pose table for reward computation
        #     self.target_initial_poses_full = torch.cat([self.target_initial_position, self.target_initial_orientation]).repeat(self.num_envs, 1)
        #     self.target_world_reset_poses_full = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
        #     self.target_world_goal_poses_full = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
        #     self.target_relative_goal_poses_full = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
        #     self.prev_target_goal_world_pos_dis_full = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        #     self.prev_target_goal_world_ori_dis_full = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        #     self.prev_target_hand_world_pos_dis_full = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        #     self.prev_fingers_poses = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)
        #     # set an initial env_stages for step_manual; The value range is [0, 1, 2, 3] -> moving, closing gripper, lifting, end
        #     self.env_stages = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        #     self.close_gripper_force_checker = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # # Update target indexes buffer to randomize the kind of target
        # if self.random_target:
        #     objects_class = torch.randint(0, self.object_idxs_full.shape[1], size=(len(reset_ids), ))
        #     self.target_idxs[reset_ids] = self.object_idxs_full[reset_ids, objects_class]
        #     self.goal_target_idxs[reset_ids] = self.goal_object_idxs_full[reset_ids, objects_class]
        #     self.object_class_idxs[reset_ids] = objects_class

        # # Target Pose
        # if self.random_target_init:
        #     target_relative_reset_positions = torch_rand_float(*self.rand_rela_init_position, shape=(len(reset_ids), 3))
        #     target_relative_reset_eulers = torch_rand_float(*self.rand_rela_init_rotation_angle, shape=(len(reset_ids), 3))
        # else: 
        #     target_relative_reset_positions = torch.zeros((len(reset_ids), 3), device=self.device, dtype=torch.float32)
        #     target_relative_reset_eulers = torch.zeros((len(reset_ids), 3), device=self.device, dtype=torch.float32)
        # target_relative_reset_orientations = quat_from_euler(target_relative_reset_eulers)
        # # need to fix the object initial reset z position
        # self.target_initial_poses_full[reset_ids, 2] = self.table_height + self.obj_height[self.object_class_idxs[reset_ids]] / 2 + 1e-4  # Fix the mesh bounding box not perfect prob
        # target_world_init_positions, target_world_init_orientations = \
        #     self.target_initial_poses_full[reset_ids, :3], self.target_initial_poses_full[reset_ids, 3:]

        # target_world_reset_positions = target_world_init_positions + target_relative_reset_positions
        # target_world_reset_orientations = quat_mul(target_world_init_orientations, target_relative_reset_orientations)
        # if self.real: # Remove real environment object;
        #     target_world_reset_positions[self.real_env_id, :] = self.prepare_area_position
        #     target_world_reset_orientations[self.real_env_id, :] = self.prepare_area_orientation

        # # Goal Pose
        # if self.random_goal:
        #     target_relative_goal_positions = torch_rand_float(*self.rand_rela_goal_position, shape=(len(reset_ids), 3))
        #     target_relative_goal_eulers = torch_rand_float(*self.rand_rela_goal_rotation_angle, shape=(len(reset_ids), 3))
        # else: 
        #     target_relative_goal_positions = torch.zeros((len(reset_ids), 3), device=self.device, dtype=torch.float32)
        #     target_relative_goal_eulers = torch.zeros((len(reset_ids), 3), device=self.device, dtype=torch.float32)
        # target_relative_goal_orientations = quat_from_euler(target_relative_goal_eulers)
        
        # if self.task=="P":
        #     target_world_goal_positions = target_world_reset_positions + target_relative_goal_positions
        #     target_world_goal_orientations = quat_mul(target_world_reset_orientations, target_relative_goal_orientations)
        #     # Keep the reset target_world_goal_positions and orientations
        #     if self.real:
        #         target_world_goal_positions[self.real_env_id, :] = self.target_initial_position + target_relative_goal_positions[self.real_env_id, :]
        #         target_world_goal_orientations[self.real_env_id, :] = quat_mul(self.target_initial_orientation, target_relative_goal_orientations[self.real_env_id, :])
        # elif self.task=='P2G':
        #     target_world_goal_positions = self.prepare_area_position.repeat(len(reset_ids), 1)
        #     target_world_goal_orientations = self.prepare_area_orientation.repeat(len(reset_ids), 1)
        
        # # Prepare Pose
        # prepare_world_positions = self.prepare_area_position.repeat(len(reset_ids)*self.object_idxs_full.shape[1], 1)
        # prepare_world_orientations = self.prepare_area_orientation.repeat(len(reset_ids)*self.object_idxs_full.shape[1], 1)

        # # Hand Pose
        # hand_world_positions = self.hand_initial_position.repeat(len(reset_ids), 1)
        # hand_world_orientations = self.hand_initial_orientation.repeat(len(reset_ids), 1)

        # #Reset Granular Media
        # if self.add_gms:
        #     gm_height = 0.03
        #     gm_range = 0.1
        #     gm_positions =  self.to_torch(np.random.uniform(low=[-self.shelf_dim[0]/2 * 0.9 + self.table_pose.p.x, -self.shelf_dim[1]/2 * 0.9 + self.table_pose.p.y, self.table_dim[2]+gm_height], \
        #                                 high=[self.shelf_dim[0]/2 * 0.9 + self.table_pose.p.x, self.shelf_dim[1]/2 * 0.9 + self.table_pose.p.y, self.table_dim[2]+gm_height+gm_range], \
        #                         size=(len(reset_ids), self.num_gms,3)))
        #     gm_orientations = self.to_torch(np.ones((len(reset_ids), self.num_gms,4)))

        # if self.add_gms:
        #     set_pose(gym, self.sim, self.root_tensor, 
        #          [self.object_idxs_full[reset_ids, :].flatten(), self.goal_object_idxs_full[reset_ids, :].flatten(), self.target_idxs[reset_ids], self.goal_target_idxs[reset_ids], self.hand_idxs[reset_ids],self.gm_ids],
        #          positions=[prepare_world_positions, prepare_world_positions, target_world_reset_positions, target_world_goal_positions, hand_world_positions,gm_positions],
        #          orientations=[prepare_world_orientations, prepare_world_orientations, target_world_reset_orientations, target_world_goal_orientations, hand_world_orientations,gm_orientations],
        #          linvels=[torch.zeros_like(prepare_world_positions), torch.zeros_like(prepare_world_positions), torch.zeros_like(target_world_reset_positions), torch.zeros_like(target_world_goal_positions), torch.zeros_like(hand_world_positions),torch.zeros_like(gm_positions)],
        #          angvels=[torch.zeros_like(prepare_world_positions), torch.zeros_like(prepare_world_positions), torch.zeros_like(target_world_reset_positions), torch.zeros_like(target_world_goal_positions), torch.zeros_like(hand_world_positions),torch.zeros_like(gm_positions)])
        # else:
        #     set_pose(gym, self.sim, self.root_tensor, 
        #          [self.object_idxs_full[reset_ids, :].flatten(), self.goal_object_idxs_full[reset_ids, :].flatten(), self.target_idxs[reset_ids], self.goal_target_idxs[reset_ids], self.hand_idxs[reset_ids]],
        #          positions=[prepare_world_positions, prepare_world_positions, target_world_reset_positions, target_world_goal_positions, hand_world_positions],
        #          orientations=[prepare_world_orientations, prepare_world_orientations, target_world_reset_orientations, target_world_goal_orientations, hand_world_orientations],
        #          linvels=[torch.zeros_like(prepare_world_positions), torch.zeros_like(prepare_world_positions), torch.zeros_like(target_world_reset_positions), torch.zeros_like(target_world_goal_positions), torch.zeros_like(hand_world_positions)],
        #          angvels=[torch.zeros_like(prepare_world_positions), torch.zeros_like(prepare_world_positions), torch.zeros_like(target_world_reset_positions), torch.zeros_like(target_world_goal_positions), torch.zeros_like(hand_world_positions)])
         
        # # reset hand DOF + control DOF 
        # set_dof(gym, self.sim, self.dof_pos, self.dof_vel, self.env_idxs[reset_ids], 
        #         self.default_dof_pos_tensor.expand(len(reset_ids), -1), self.default_dof_vel_tensor.expand(len(reset_ids), -1), 
        #         lower_limits=self.hand_lower_limits, upper_limits=self.hand_upper_limits)
        # control_dof(gym, self.sim, self.dof_pos, self.env_idxs[reset_ids], self.default_dof_pos_tensor.expand(len(reset_ids), -1))

        # if self.real:
        #     self.robot.set_pose_gripper(torch.cat([hand_world_positions[self.real_env_id], hand_world_orientations[self.real_env_id]]))
        #     self.robot.release()

        # # Query the target position and green, red finger position
        # if self.use_gpu_pipeline: self.stepsimulation() # Using gpu_pipeline requires one stepsimulation to reset and refreshtensor
        # else: self.refresh_request_tensors()
        # target_world_cur_poses, _ = get_pose(self.root_tensor, self.target_idxs[reset_ids])
        # hand_poses, _ = get_pose(self.root_tensor, self.hand_idxs[reset_ids])
        # finger_red_poses, _ = get_pose(self.rb_states, self.finger_red_idxs[reset_ids])
        # finger_green_poses, _ = get_pose(self.rb_states, self.finger_green_idxs[reset_ids])
        # # update the full table
        # self.target_world_reset_poses_full[reset_ids, :3] = target_world_reset_positions
        # self.target_world_reset_poses_full[reset_ids, 3:] = target_world_reset_orientations
        # self.prev_fingers_poses[reset_ids, 0:3] = finger_red_poses[:, :3]
        # self.prev_fingers_poses[reset_ids, 3:6] = finger_green_poses[:, :3]
        # # update goal
        # self.target_world_goal_poses_full[reset_ids, :3] = target_world_goal_positions
        # self.target_world_goal_poses_full[reset_ids, 3:] = target_world_goal_orientations
        # self.target_relative_goal_poses_full[reset_ids, :3] = target_relative_goal_positions
        # self.target_relative_goal_poses_full[reset_ids, 3:] = target_relative_goal_orientations
        # # update the initial distance table
        # self.prev_target_goal_world_pos_dis_full[reset_ids] = torch.norm(target_world_goal_positions[:, :2] - target_world_reset_positions[:, :2], p=2, dim=1)
        # self.prev_target_goal_world_ori_dis_full[reset_ids] = torch.norm(target_world_goal_orientations[:, 3:] - target_world_reset_orientations[:, 3:], p=2, dim=1)
        # self.prev_target_hand_world_pos_dis_full[reset_ids] = torch.norm(target_world_reset_positions[:, :2] - hand_poses[:, :2], p=2, dim=1)

        # # reset the buffer
        # self.reset_buffers(reset_ids)

        # return self.observations_buf


    def compute_observations(self):
        if self.use_gpu_pipeline: self.compute_observations_gpu()
        else: self.compute_observations_cpu()

    
    def compute_reward(self):
        if self.task=="P": return self.compute_reward_P()
        elif self.task=="P2G": return self.compute_reward_P2G()


    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.task=="P": return self.step_P(actions)
        elif self.task=="P2G": return self.step_P2G(actions)


    def compute_observations_cpu(self):
        # In the future, we can use a for loop to loop all fingers and compute each finger's observation then cat them together.
        # Maybe need a faster way to compute observations 
        # Fetch again to make sure the results be updated
        gym.fetch_results(self.sim, True)
        self.refresh_request_tensors()
        
        # Full contacts might save time but the value of return is weired (looks like meaningless numbers) and still need to pick up the contact force! 
        # self.full_contacts = gym.get_rigid_contacts(self.sim)

        target_world_poses, _ = get_pose(self.root_tensor, self.target_idxs)
        hand_world_poses, _ = get_pose(self.root_tensor, self.hand_idxs)
        finger_red_world_poses, _ = get_pose(self.rb_states, self.finger_red_idxs)
        finger_green_world_poses, _ = get_pose(self.rb_states, self.finger_green_idxs)
        finger_red_force = get_force(self.force_tensor, self.finger_red_idxs)
        finger_green_force = get_force(self.force_tensor, self.finger_green_idxs)
        # finger_red_sensor_force = get_force(self.force_sensor_tensor, self.finger_red_force_idxs)
        # finger_green_sensor_force = get_force(self.force_sensor_tensor, self.finger_green_force_idxs)
        # cube_sensor_force = get_force(self.force_sensor_tensor, self.cube_force_idxs)
        
        # Query all contact information env by env (isaac gym's problem!)
        contact_info_buffer = []
        for i, env in enumerate(self.envs):
            # contact[0] is contact position, contact[1] is normal direction
            red_contact, green_contact = get_contact_points(env, [self.env_finger_red_idx, self.env_finger_green_idx], filter_mask=self.filter_area_start_end)
            if red_contact is None: 
                red_contact = [[0.] * 3, [0.] * 3]; finger_red_force[i] = 0.
            if green_contact is None: 
                green_contact = [[0.] * 3, [0.] * 3]; finger_green_force[i] = 0.
            # (3D point position + net contact force) * 2 fingers
            red_contact_info = red_contact[0] + finger_red_force[i].tolist() # 3D point position + net contact force
            green_contact_info = green_contact[0] + finger_green_force[i].tolist() # 3D point position + normal direction
            contact_info = sum([red_contact_info, green_contact_info], [])
            contact_info_buffer.append(contact_info)
        
        contact_info_buffer = self.to_torch(contact_info_buffer, dtype=torch.float32)
        
        if self.real: # overwrite the real-env obs to real observation
            finger_red_world_poses[self.real_env_id, :] = self.robot.get_pose_finger(self.red_finger)
            finger_green_world_poses[self.real_env_id, :] = self.robot.get_pose_finger(self.green_finger)
            rob_finger_red_force_scalar = self.robot.get_force(self.red_finger)
            rob_finger_green_force_scalar = self.robot.get_force(self.green_finger)
            contact_info_buffer[self.real_env_id, 0:6] = self.robot.get_contact_points_single(self.red_finger)
            contact_info_buffer[self.real_env_id, 6:12] = self.robot.get_contact_points_single(self.green_finger)
            # Convert real contact force info to simulation frame (world) to match the isaac gym contact info query format
            contact_info_buffer[self.real_env_id, 3:6] = quat_apply(finger_red_world_poses[self.real_env_id, 3:], contact_info_buffer[self.real_env_id, 3:6]*rob_finger_red_force_scalar)
            contact_info_buffer[self.real_env_id, 9:12] = quat_apply(finger_green_world_poses[self.real_env_id, 3:], contact_info_buffer[self.real_env_id, 9:12]*rob_finger_green_force_scalar)
            # print(f"Finger force: {finger_green_force}")

        if self.use_2D_contact: # Zero out Z dimension component and normalize again
            contact_info_buffer[:, [2, 5, 8, 11]] = 0.
        if not self.use_contact_force: # use the normal direction
            contact_info_buffer[:, 3:6] = normalize(contact_info_buffer[:, 3:6])
            contact_info_buffer[:, 9:12] = normalize(contact_info_buffer[:, 9:12])

        non_contact_idxs = (contact_info_buffer == 0).all(dim=1)
        self.nocontact_buf[non_contact_idxs] += 1
        self.nocontact_buf[~non_contact_idxs] = 0 # Has contact: reset nocontact_buffer
        
        if self.draw_contact: # Save world frame contact info for visualization
            contact_info_buffer_world_frame = contact_info_buffer.clone()
            contact_info_buffer_world_frame[:, 0:3] = tf_apply(finger_red_world_poses[:, 3:], finger_red_world_poses[:, :3], contact_info_buffer_world_frame[:, 0:3])
            contact_info_buffer_world_frame[:, 6:9] = tf_apply(finger_green_world_poses[:, 3:], finger_green_world_poses[:, :3], contact_info_buffer_world_frame[:, 6:9])

        # Transform contact points to world frame if needed / Compute observations that needed
        if self.use_world_frame_obs:
            world_to_red_finger_trans, world_to_red_finger_ori = finger_red_world_poses[:, :3], finger_red_world_poses[:, 3:]
            world_to_green_finger_trans, world_to_green_finger_ori = finger_green_world_poses[:, :3], finger_green_world_poses[:, 3:]
            contact_info_buffer[:, 0:3] = tf_apply(world_to_red_finger_ori, world_to_red_finger_trans, contact_info_buffer[:, 0:3])
            contact_info_buffer[:, 6:9] = tf_apply(world_to_green_finger_ori, world_to_green_finger_trans, contact_info_buffer[:, 6:9])
            if self.use_contact_torque: # cross product T = r x F (r is the distance vector from contact point to the center)
                contact_info_buffer[:, 0:3] = torch.cross(contact_info_buffer[:, 0:3] - world_to_red_finger_trans, contact_info_buffer[:, 3:6])
                contact_info_buffer[:, 6:9] = torch.cross(contact_info_buffer[:, 6:9] - world_to_green_finger_trans, contact_info_buffer[:, 9:12])
        # Transform all other information to finger local frame (default) 
        else: 
            finger_to_world_red_ori, finger_to_world_red_trans = tf_inverse(finger_red_world_poses[:, 3:], finger_red_world_poses[:, :3])
            finger_to_world_green_ori, finger_to_world_green_trans = tf_inverse(finger_green_world_poses[:, 3:], finger_green_world_poses[:, :3])
            # Net force direction to the finger frame; these are vector not point, the translation will be canceled out!
            # We need to give the world frame goal pose rather than relative pose (it will be considered as world pose)
            contact_info_buffer[:, 3:6] = quat_apply(finger_to_world_red_ori, contact_info_buffer[:, 3:6])
            contact_info_buffer[:, 9:12] = quat_apply(finger_to_world_green_ori, contact_info_buffer[:, 9:12])
            if self.use_contact_torque:
                contact_info_buffer[:, 0:3] = torch.cross(contact_info_buffer[:, 0:3], contact_info_buffer[:, 3:6])
                contact_info_buffer[:, 6:9] = torch.cross(contact_info_buffer[:, 6:9], contact_info_buffer[:, 9:12])
            if self.include_target_obs:
                finger_red_to_target_poses = \
                    torch.cat(tf_combine(finger_to_world_red_ori, finger_to_world_red_trans, target_world_poses[:, 3:], target_world_poses[:, :3]), dim=1)
                finger_green_to_target_poses = \
                    torch.cat(tf_combine(finger_to_world_green_ori, finger_to_world_green_trans, target_world_poses[:, 3:], target_world_poses[:, :3]), dim=1)

        if self.include_finger_vel:
            red_finger_vel = self.prev_fingers_poses[:, :3] - finger_red_world_poses[:, :3]
            green_finger_vel = self.prev_fingers_poses[:, 3:6] - finger_green_world_poses[:, :3]
            self.prev_fingers_poses[:, :3], self.prev_fingers_poses[:, 3:6] = finger_red_world_poses[:, :3], finger_green_world_poses[:, :3]
            if not self.use_world_frame_obs:
                red_finger_vel = quat_apply(finger_to_world_red_ori, red_finger_vel)
                green_finger_vel = quat_apply(finger_to_world_green_ori, green_finger_vel)
        
        if self.use_relative_goal:
            goal_poses_in_hand_frame_full = self.target_relative_goal_poses_full.clone()
        else:
            hand_to_world_ori, hand_to_world_trans = tf_inverse(hand_world_poses[:, 3:], hand_world_poses[:, :3])
            goal_poses_in_hand_frame_full \
                = torch.cat(tf_combine(hand_to_world_ori, hand_to_world_trans, self.target_world_goal_poses_full[:, 3:], self.target_world_goal_poses_full[:, :3]), dim=1)

        if self.add_random_noise:
            pos_random_noise = torch_rand_float(*self.to_torch([-self.contact_noise_v, self.contact_noise_v]), shape=(self.num_envs, 3))
            force_random_noise = torch_rand_float(*self.to_torch([-self.force_noise_v, self.force_noise_v]), shape=(self.num_envs, 3))
            contact_info_buffer[:, 0:3] += pos_random_noise
            contact_info_buffer[:, 6:9] += pos_random_noise
            contact_info_buffer[:, 3:6] += force_random_noise if self.use_contact_force else normalize(force_random_noise)
            contact_info_buffer[:, 9:12] += force_random_noise if self.use_contact_force else normalize(force_random_noise)
            
        # Start to concatenate all observations
        self.last_observation = torch.cat((contact_info_buffer, self.raw_actions), dim=1) # minimum observation
        if self.task=="P":
            self.last_observation = torch.cat((self.last_observation, goal_poses_in_hand_frame_full), dim=1)

        if self.use_world_frame_obs:
            self.last_observation = torch.cat((self.last_observation, finger_red_world_poses, finger_green_world_poses), dim=1)
        
        if self.include_target_obs:
            if self.use_world_frame_obs:
                self.last_observation = torch.cat((self.last_observation, target_world_poses), dim=1)
            else:
                self.last_observation = torch.cat((self.last_observation, finger_green_to_target_poses), dim=1) 
        if self.include_finger_vel:
            self.last_observation = torch.cat((self.last_observation, green_finger_vel), dim=1)

        # Update observation buffer | Use LSTM+Linear or only use Linear (default Linear)
        if self.use_lstm or self.use_transformer:
            self.observations_buf[:, :-1, :] = self.observations_buf[:, 1:, :].detach().clone() if self.num_envs==1 \
                                               else self.observations_buf[:, 1:, :] # faster popleft
            self.observations_buf[:, -1, :] = self.last_observation
        else:
            self.observations_buf[:, :-self.single_observation_dims] = self.observations_buf[:, self.single_observation_dims:].detach().clone() if self.num_envs==1 \
                                                                       else self.observations_buf[:, self.single_observation_dims:] # popleft one observation
            self.observations_buf[:, -self.single_observation_dims:] = self.last_observation

        # Visualization
        if self.rendering:
            gym.clear_lines(self.viewer)
            if self.draw_contact: self.visualize_contact_force(contact_info_buffer_world_frame, refresh=False)
            
            if self.filter_contact: self.visualize_filter_area(refresh=False)
            self.visualize_hand_axis(refresh=False)
            self.update_viewer()

    
    def compute_observations_gpu(self):
        # Maybe need a faster way to compute observations 
        # Fetch again to make sure the results be updated
        gym.fetch_results(self.sim, True)
        self.refresh_request_tensors()

        target_world_poses, _ = get_pose(self.root_tensor, self.target_idxs)
        finger_green_world_poses, _ = get_pose(self.rb_states, self.finger_green_idxs)
        finger_green_force = get_force(self.force_tensor, self.finger_green_idxs)
        contact_info_buffer = finger_green_force
        
        if self.real: # overwrite the real-env obs to real observation
            finger_green_world_poses[self.real_env_id, :] = self.robot.get_pose_finger(self.green_finger)
            rob_finger_green_force_scalar = self.robot.get_force(self.green_finger)
            contact_info_buffer[self.real_env_id, 0:3] = self.robot.get_contact_points_single(self.green_finger)[3:6] * rob_finger_green_force_scalar
            # Convert real contact force info to simulation frame (world) to match the isaac gym contact info query format
            contact_info_buffer[self.real_env_id, 0:3] = quat_apply(finger_green_world_poses[self.real_env_id, 3:], contact_info_buffer[self.real_env_id, 0:3])
            finger_green_force[self.real_env_id, :] = contact_info_buffer[self.real_env_id, 0:3]
            # print(f"Finger force: {finger_green_force}")

        if self.use_2D_contact: # Zero out Z dimension component and normalize again
            contact_info_buffer[:, 2], finger_green_force[:, 2] = 0., 0.
        if not self.use_contact_force: # use the normal direction
            contact_info_buffer[:, 0:3] = normalize(contact_info_buffer[:, 0:3])

        non_contact_idxs = (contact_info_buffer == 0).all(dim=1)
        self.nocontact_buf[non_contact_idxs] += 1
        self.nocontact_buf[~non_contact_idxs] = 0 # Has contact: reset nocontact_buffer
        
        if not self.use_world_frame_obs:
            # Transform all other information to finger local frame (default) 
            finger_to_world_green_ori, finger_to_world_green_trans = tf_inverse(finger_green_world_poses[:, 3:], finger_green_world_poses[:, :3])
            contact_info_buffer[:, 0:3] = quat_apply(finger_to_world_green_ori, contact_info_buffer[:, 0:3])
            if self.include_target_obs:
                finger_green_to_target_poses = target_world_poses.clone()
                finger_green_to_target_poses[:, :3], finger_green_to_target_poses[:, 3:] = \
                    tf_combine(finger_to_world_green_ori, finger_to_world_green_trans, target_world_poses[:, 3:], target_world_poses[:, :3])

        if self.use_abstract_contact_obs:
            green_finger_get_contact = (~torch.isclose(contact_info_buffer[:, 3:6], torch.zeros_like(contact_info_buffer[:, 3:6]))).any(dim=1)
            green_finger_contact_angle = torch.arctan2(contact_info_buffer[:, 4], contact_info_buffer[:, 3])
            contact_info_buffer = torch.stack((green_finger_get_contact, green_finger_contact_angle), dim=1)

        if self.include_finger_vel:
            green_finger_vel = self.prev_fingers_poses[:, 3:6] - finger_green_world_poses[:, :3]
            self.prev_fingers_poses[:, 3:6] = finger_green_world_poses[:, :3]
            if not self.use_world_frame_obs:
                green_finger_vel = quat_apply(finger_to_world_green_ori, green_finger_vel)
        
        if self.use_relative_goal:
            goal_poses_in_hand_frame_full = self.target_relative_goal_poses_full.clone()
        else:
            goal_poses_in_hand_frame_full = torch.zeros_like(finger_green_world_poses)
            goal_poses_in_hand_frame_full[:, :3], goal_poses_in_hand_frame_full[:, 3:] \
                = tf_combine(finger_to_world_green_ori, finger_to_world_green_trans, self.target_world_goal_poses_full[:, 3:], self.target_world_goal_poses_full[:, :3])

        if self.add_random_noise:
            force_random_noise = torch_rand_float(*self.to_torch([-self.force_noise_v, self.force_noise_v]), shape=(self.num_envs, 3))
            contact_info_buffer[:, :3] += force_random_noise if self.use_contact_force else normalize(force_random_noise)

        # Start to concatenate observations
        self.last_observation = torch.cat((contact_info_buffer, self.raw_actions, goal_poses_in_hand_frame_full), dim=1) # minimum observation
        if self.use_world_frame_obs:
            self.last_observation = torch.cat((self.last_observation, finger_green_world_poses), dim=1)
        
        if self.include_target_obs:
            if self.use_world_frame_obs:
                self.last_observation = torch.cat((self.last_observation, target_world_poses), dim=1)
            else:
                self.last_observation = torch.cat((self.last_observation, finger_green_to_target_poses), dim=1)
        
        if self.include_finger_vel:
            self.last_observation = torch.cat((self.last_observation, green_finger_vel), dim=1)

        # Update observation buffer | Use LSTM+Linear or only use Linear (default Linear)
        if self.use_lstm or self.use_transformer:
            self.observations_buf[:, :-1, :] = self.observations_buf[:, 1:, :].detach().clone() if self.num_envs==1 \
                                               else self.observations_buf[:, 1:, :] # faster popleft
            self.observations_buf[:, -1, :] = self.last_observation
        else:
            self.observations_buf[:, :-self.single_observation_dims] = self.observations_buf[:, self.single_observation_dims:].detach().clone() if self.num_envs==1 \
                                                                       else self.observations_buf[:, self.single_observation_dims:] # popleft one observation
            self.observations_buf[:, -self.single_observation_dims:] = self.last_observation


    def compute_reward_P(self):
        target_world_cur_poses, _ = get_pose(self.root_tensor, self.target_idxs)

        # compute l2 distance for [x, y] poses (need to scale them for better performancfe?) 
        # We need to use euler angle to compute orientation otherwise it is super unstable -> No, quaternion convert to euler might be very numerical unstable? 
        delta_pos_norm_full = torch.norm(self.target_world_goal_poses_full[:, :2] - target_world_cur_poses[:, :2], p=2, dim=1)
        delta_ori_norm_full = torch.norm((self.target_world_goal_poses_full[:, 3:] - target_world_cur_poses[:, 3:]), p=2, dim=1)
        # delta_ori_norm_full = torch.norm(quat_mul(target_world_cur_poses[:, 3:], quat_conjugate(self.target_world_reset_poses_full[:, 3:]))[:, :3], p=2, dim=1)
        # delta_ori_norm_full = torch.asin(torch.clamp(delta_ori_norm_full, max=1))

        # We need a second order difference reward for this because we do not have cube pose state!
        pos_eps = 0.1; ori_eps = 0.1;
        pos_reward = self.pos_w * (1.0 / (delta_pos_norm_full + pos_eps) - 1.0 / (self.prev_target_goal_world_pos_dis_full + pos_eps))
        ori_reward = self.ori_w * (1.0 / (delta_ori_norm_full + ori_eps) - 1.0 / (self.prev_target_goal_world_ori_dis_full + ori_eps))

        self.prev_target_goal_world_pos_dis_full[:] = delta_pos_norm_full.detach().clone() 
        self.prev_target_goal_world_ori_dis_full[:] = delta_ori_norm_full.detach().clone()

        action_penalty = self.act_w * torch.sum(self.raw_actions[:, :3].abs(), dim=1)
        
        rewards = pos_reward + ori_reward + action_penalty
        self.step_info.update({'pos_reward': pos_reward, 'act_penalty': action_penalty, 'step_env_idx': self.env_stages==0})

        # --- Compute reset indexes --- #
        penalty_r = 0.; success_r = 40.
        # terminated episodes that reached the max steps
        out_of_time_index = self.steps_buf >= self.maximum_steps
        # terminated episodes that reached the max no-contact steps
        lose_track_index = self.nocontact_buf >= self.maximum_no_contact
        # terminated episodes that the Target falls down to the ground
        target_fall_idxs = (target_world_cur_poses[:, 2] - self.target_world_reset_poses_full[:, 2]) < self.obj_fall_threshold
        if torch.any(target_fall_idxs): print("Target is knocked down!")
        # terminated episodes that insert into the table
        hand_insertion_idxs = torch.any(self.last_observation.abs() >= self.maximum_detect_force, dim=-1).squeeze(dim=-1)
        if torch.any(hand_insertion_idxs): print("Hand insertion!")
        # Success
        success_idxs = delta_pos_norm_full < self.pose_diff_threshold[0]
        if self.ori_w > 0: success_idxs *= delta_ori_norm_full < self.pose_diff_threshold[1]
        
        self.success_buf[success_idxs] = 1
        rewards[out_of_time_index] -= penalty_r
        rewards[lose_track_index] -= penalty_r
        rewards[target_fall_idxs] -= penalty_r
        rewards[hand_insertion_idxs] -= penalty_r
        rewards[success_idxs] += success_r

        if self.auto_reset:
            self.reset_buf[hand_insertion_idxs] = 1
            self.reset_buf[target_fall_idxs] = 1
            self.reset_buf[lose_track_index] = 1
            self.reset_buf[out_of_time_index] = 1
            self.reset_buf[success_idxs] = 1

        # Update rewards
        self.rewards_buf[:] = rewards


    def compute_reward_P2G(self):
        target_world_cur_poses, _ = get_pose(self.root_tensor, self.target_idxs)
        hand_world_cur_poses, _ = get_pose(self.root_tensor, self.hand_idxs)

        # compute l2 distance for [x, y] poses (need to scale them for better performancfe?) 
        # We need to use euler angle to compute orientation otherwise it is super unstable -> No, quaternion convert to euler might be very numerical unstable? 
        delta_pos_norm_full = torch.norm(hand_world_cur_poses[:, :2] - target_world_cur_poses[:, :2], p=2, dim=1)

        # We need a second order difference reward for this because we do not have cube pose state!
        pos_eps = 0.1; ori_eps = 0.1;
        pos_reward = self.pos_w * (1.0 / (delta_pos_norm_full + pos_eps) - 1.0 / (self.prev_target_hand_world_pos_dis_full + pos_eps))

        self.prev_target_hand_world_pos_dis_full[:] = delta_pos_norm_full.detach().clone() 

        action_penalty = self.act_w * torch.sum(self.raw_actions[:, :3].abs(), dim=1)
        
        rewards = pos_reward + action_penalty
        self.step_info.update({'pos_reward': pos_reward, 'act_penalty': action_penalty, 'step_env_idx': (self.env_stages==0)+(self.env_stages==3)})

        # --- Compute reset indexes --- #
        penalty_r = 0.; success_r = 800.
        # terminated episodes that reached the max steps
        out_of_time_index = self.steps_buf >= self.maximum_steps
        # if torch.any(out_of_time_index): print("Out of time")
        # terminated episodes that reached the max no-contact steps
        lose_track_index = self.nocontact_buf >= self.maximum_no_contact
        # if torch.any(lose_track_index): print("Lose track")
        # terminated episodes that the Target falls down to the ground
        target_fall_idxs = (target_world_cur_poses[:, 2] < self.target_world_reset_poses_full[:, 2]) < self.obj_fall_threshold
        # if torch.any(target_fall_idxs): print("Target Fall")
        # terminated episodes that insert into the table
        hand_insertion_idxs = torch.any(self.last_observation.abs() >= self.maximum_detect_force, dim=-1).squeeze(dim=-1)
        # if torch.any(hand_insertion_idxs): print("Hand insertion")
        # Success
        terminated_idxs = (self.env_stages == 3).nonzero().squeeze(dim=-1)
        success_idxs = terminated_idxs[(target_world_cur_poses[terminated_idxs, 2] - self.target_world_reset_poses_full[terminated_idxs, 2]) > self.lift_success_thre]
        # Close gripper but fail
        close_fail_idxs = terminated_idxs[(target_world_cur_poses[terminated_idxs, 2] - self.target_world_reset_poses_full[terminated_idxs, 2]) <= self.lift_success_thre]

        self.success_buf[success_idxs] = 1
        rewards[out_of_time_index] -= penalty_r
        rewards[lose_track_index] -= penalty_r
        rewards[target_fall_idxs] -= penalty_r
        rewards[hand_insertion_idxs] -= penalty_r
        rewards[close_fail_idxs] -= penalty_r
        rewards[success_idxs] += success_r

        if self.auto_reset:
            self.reset_buf[hand_insertion_idxs] = 1
            self.reset_buf[target_fall_idxs] = 1
            self.reset_buf[lose_track_index] = 1
            self.reset_buf[out_of_time_index] = 1
            self.reset_buf[close_fail_idxs] = 1
            self.reset_buf[success_idxs] = 1

        # Update rewards
        self.rewards_buf[:] = rewards


    def step_P(self, actions: torch.Tensor):
        """Step the physics of the environment.
        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """
        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        for i in range(self.interval_steps):
            if self.real:
                self.robot.set_pose_gripper(self.step_actions[i][self.real_env_id, :])
                robot_gripper_pose = self.robot.get_pose_gripper()
                self.step_actions[i][self.real_env_id, :] = robot_gripper_pose
            
            set_pose(gym, self.sim, self.root_tensor, self.hand_idxs, \
                     self.step_actions[i][:, :3], self.step_actions[i][:, 3:])

            self.stepsimulation()

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        return self.observations_buf, self.rewards_buf, self.reset_buf, self.step_info


    def step_P2G(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.
        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # apply actions
        self.pre_physics_step(actions)

        # Act the terminal action episode
        # Terminal actions Question: we need to move this heuristic control outside the training loop
        moving_env_ids = (self.env_stages == 0).nonzero(as_tuple=False).squeeze(dim=-1)
        start_closing_env_ids = (self.raw_actions[:, -1] == 2).nonzero(as_tuple=False).squeeze(dim=-1)
        lifting_env_ids = (self.env_stages == 2).nonzero(as_tuple=False).squeeze(dim=-1)
        self.env_stages[start_closing_env_ids] = 1
        # Lifting envs should keep their stages as 2 (overwrite the previous start_closing_env_ids)
        self.env_stages[lifting_env_ids] = 2

        # step physics and render each frame
        for i in range(self.interval_steps):
            set_actor_ids = []
            set_actor_positions = []
            set_actor_orientations = []
            
            # Finger movement control (It is not necessary to write len(..._env_ids)>0 but just to improve readability)
            if len(moving_env_ids) > 0 and self.step_actions is not None:
                set_actor_ids.append(self.hand_idxs[moving_env_ids])
                set_actor_positions.append(self.step_actions[i][moving_env_ids, :3])
                set_actor_orientations.append(self.step_actions[i][moving_env_ids, 3:])
            
            closing_gripper_env_ids = (self.env_stages == 1).nonzero(as_tuple=False).squeeze(dim=-1)
            # Close gripper check
            if len(closing_gripper_env_ids) > 0:
                finger_red_force = get_force(self.force_tensor, self.finger_red_idxs[closing_gripper_env_ids]).norm(dim=1)
                finger_green_force = get_force(self.force_tensor, self.finger_green_idxs[closing_gripper_env_ids]).norm(dim=1)
                if self.real and self.real_env_id in closing_gripper_env_ids:
                    real_env_closing_idx = (closing_gripper_env_ids==self.real_env_id).nonzero(as_tuple=False).squeeze(dim=-1)
                    finger_red_force[real_env_closing_idx] = self.robot.get_force(self.red_finger)
                    finger_green_force[real_env_closing_idx] = self.robot.get_force(self.green_finger)

                force_satisfy_idxs = (finger_red_force > self.force_threshold) * \
                                     (finger_green_force > self.force_threshold)
                self.close_gripper_force_checker[closing_gripper_env_ids[force_satisfy_idxs]] += 1
                done_close_gripper_idxs = self.close_gripper_force_checker >= self.force_check_nums
                self.env_stages[done_close_gripper_idxs] = 2

                force_failure_idxs = ~force_satisfy_idxs
                claw_actions = torch.ones((len(closing_gripper_env_ids), self.hand_num_dofs), dtype=torch.float32, device=self.device) * JointP
                # Mask out actor that is already statisfied with the force check, stop continuing to close gripper
                claw_actions[force_satisfy_idxs, :] = 0.
                claw_target_pos = self.dof_pos[closing_gripper_env_ids, :] + claw_actions

                if False: # Close gripper but find not pass the force check
                    fail_grasp_obj_idxs = closing_gripper_env_ids[self.close_gripper_force_checker[closing_gripper_env_ids[force_failure_idxs]] > 0]
                    self.env_stages[fail_grasp_obj_idxs] = 0; self.close_gripper_force_checker[fail_grasp_obj_idxs] = 0
                    control_dof(gym, self.sim, self.dof_pos, fail_grasp_obj_idxs, self.default_dof_pos_tensor.expand(len(fail_grasp_obj_idxs), -1))

            # Lift action
            lifting_env_ids = (self.env_stages == 2).nonzero(as_tuple=False).squeeze(dim=-1)
            if len(lifting_env_ids) > 0:
                lifting_hand_idxs = self.hand_idxs[lifting_env_ids]
                hand_poses, _ = get_pose(self.root_tensor, lifting_hand_idxs)
                hand_pos, hand_ori = hand_poses[:, :3], hand_poses[:, 3:]
                hand_pos[:, 2] = hand_pos[:, 2] + self.lift_action_step
                set_actor_ids.append(lifting_hand_idxs); set_actor_positions.append(hand_pos); set_actor_orientations.append(hand_ori)

            # Set actor together here (Otherwise GPU pipeline will only consider the last set_pose function)
            if len(set_actor_ids) > 0:
                if self.real:
                    if self.real_env_id in moving_env_ids:
                        real_env_moving_idx = (moving_env_ids==self.real_env_id).nonzero(as_tuple=False).squeeze(dim=-1)
                        self.robot.set_pose_gripper(self.step_actions[i][self.real_env_id])
                    elif self.real_env_id in lifting_env_ids:
                        real_env_lifting_idx = (lifting_env_ids==self.real_env_id).nonzero(as_tuple=False).squeeze(dim=-1)
                        self.robot.set_pose_gripper(torch.cat([hand_pos[real_env_lifting_idx, :], hand_ori[real_env_lifting_idx, :]]))
                    
                    gripper_pose = self.robot.get_pose_gripper().squeeze(dim=0)
                    set_actor_ids.append(self.hand_idxs[self.real_env_id].unsqueeze(dim=0)); set_actor_positions.append(gripper_pose[:3]); set_actor_orientations.append(gripper_pose[3:])
                set_pose(gym, self.sim, self.root_tensor, set_actor_ids, set_actor_positions, set_actor_orientations)
            
            # Control actor DOF here together
            if len(closing_gripper_env_ids) > 0:
                if self.real:
                    if self.real_env_id in closing_gripper_env_ids:
                        real_env_closing_idx = (closing_gripper_env_ids==self.real_env_id).nonzero(as_tuple=False).squeeze(dim=-1)
                        self.robot.move_fin_pos(claw_target_pos[real_env_closing_idx, :])
                    claw_pos = self.robot.get_fin_pos()
                    claw_target_pos[real_env_closing_idx, :] = claw_pos
                control_dof(gym, self.sim, self.dof_pos, closing_gripper_env_ids, claw_target_pos, 
                    lower_limits=self.hand_lower_limits, upper_limits=self.hand_upper_limits)

            self.stepsimulation()

            # Check if lifting is done
            if len(lifting_env_ids) > 0:
                lifting_hand_idxs = self.hand_idxs[lifting_env_ids]
                hand_poses, _ = get_pose(self.root_tensor, lifting_hand_idxs)
                if self.real:
                    if self.real_env_id in lifting_env_ids:
                        real_env_lifting_idx = (lifting_env_ids==self.real_env_id).nonzero(as_tuple=False).squeeze(dim=-1)
                        hand_poses[real_env_lifting_idx, :] = self.robot.get_pose_gripper()

                hand_pos, hand_ori = hand_poses[:, :3], hand_poses[:, 3:]
                lifting_end_env_ids = lifting_env_ids[hand_pos[:, 2] - self.hand_initial_position[2] >= self.lift_height]
                self.env_stages[lifting_end_env_ids] = 3

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        return self.observations_buf, self.rewards_buf, self.reset_buf, self.step_info
    

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
        with keyboard.Events() as events:
            while not gym.query_viewer_has_closed(self.viewer):
                key = None
                event = events.get(0.0001)
                if event is not None:
                    if isinstance(event, events.Press) and hasattr(event.key, 'char'):
                        key = event.key.char
                if key == 'r': self.reset()

                # step the physics
                self.stepsimulation()


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


    def compute_bounding_box_lines(self, env, start_end_coordinate: torch.Tensor):
        start_cor, end_cor = start_end_coordinate
        x1, y1, z1 = start_cor
        x2, y2, z2 = end_cor
        corner_points = self.to_torch([(x1, y1),(x1, y2), (x2, y2), (x2, y1)])
        up_points = torch.cat([corner_points, torch.ones((4, 1))*z1], dim=1)
        bot_points = torch.cat([corner_points, torch.ones((4, 1))*z2], dim=1)
        lines_box = []
        for i in range(len(up_points)-1):
            lines_box.append(torch.cat([up_points[i], up_points[i+1]]))
            lines_box.append(torch.cat([bot_points[i], bot_points[i+1]]))
            lines_box.append(torch.cat([up_points[i], bot_points[i]]))
        lines_box.append(torch.cat([up_points[0], up_points[-1]]))
        lines_box.append(torch.cat([bot_points[0], bot_points[-1]]))
        lines_box.append(torch.cat([up_points[-1], bot_points[-1]]))
        return torch.cat(lines_box, dim=0)


    def create_target_asset(self, target_asset_file, fix_base_link=False, disable_gravity=False, density=None, use_default_cube=False, target_dim=[0.05]*3):
        if use_default_cube:
            return self.create_box_asset(dimension=target_dim, fix_base_link=fix_base_link, disable_gravity=disable_gravity, density=density)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = fix_base_link
        asset_options.disable_gravity = disable_gravity
        if density is not None: asset_options.density = density
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
        else: return gym.create_box(self.sim, radius, radius, radius, asset_options)


    def close(self):
        if self.rendering: gym.destroy_viewer(self.viewer)
        gym.destroy_sim(self.sim)


    def to_torch(self, x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=self.device, requires_grad=False)


if __name__ == '__main__':
    args = gymutil.parse_arguments(description="Isaac Gym for Sandem")
    args.graphics_device_id = 1 # might need to change in different computer
    args.object_name = 'cube'
    args.task = 'P2G'
    args.use_gpu_pipeline = False
    args.rendering = True
    args.use_lstm = False
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

    env = HandemEnv(args)
    env.step_manual()
    env.close()