from utils import read_json, dict2list, get_on_bbox, get_in_bbox, pc_random_downsample
from torch_utils import tf_apply
import time
import os
import numpy as np
from math import ceil
import torch
import open3d as o3d
from copy import deepcopy

# Pybullet things
import pybullet as p
import time
import pybullet_data
import pybullet_utils as pu
try:
    from pynput import keyboard
except ImportError: 
    print("*** Warning: pynput can not be used on the server ***")

# PointNet
from PointNet_Model.pointnet2_cls_ssg import get_model

# Visualization
from tabulate import tabulate


class RoboSensaiBullet:
    ON = 0
    IN = 1

    def __init__(self, args=None) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.numpy_dtype = np.float16 if (hasattr(self.args, "use_bf16") and self.args.use_bf16) else np.float32
        self.tensor_dtype = torch.bfloat16 if (hasattr(self.args, "use_bf16") and self.args.use_bf16) else torch.float32
        self.rng = np.random.default_rng(args.seed if hasattr(args, "seed") else None)
        self._init_simulator()
        self.update_objects_database()
        self.load_world(num_objects=self.args.num_pool_objs, 
                        random=self.args.random_select_pool, 
                        default_scaling=self.args.default_scaling if hasattr(self.args, "default_scaling") else 1.)
        self.reset()


    def _init_simulator(self):
        connect_type = p.GUI if self.args.rendering else p.DIRECT
        self.client_id = p.connect(connect_type)#or p.DIRECT for non-graphical version
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if not self.args.debug:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=self.client_id)
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)


    def update_objects_database(self):
        # self.obj_categories = read_json(f'{self.args.asset_root}/{self.args.object_pool_folder}/obj_categories.json')
        # self.test_obj_categories = read_json(f'{self.args.asset_root}/{self.args.object_pool_folder}/test_obj_categories.json')
        
        # Right now we only have two stages, object can not appear in both stages! We need to figure out how to deal with this problem
        #   0: "Table"
        self.obj_categories = {
            1: {
                "microwave": 1,
                "bowl": 1,
                "tomato_soup_can": 1,
                "master_chef_can": 1,
                "pitcher_base": 1,
                "cuboid": 1,
                "cube_arrow": 1,
                "mug": 3,
                "bleach_cleanser": 1,
                "cracker_box": 1,
                "sugar_box": 1,
                # "lamp": 1,
                # "storage_furniture": 1,

                "ball": 1,
                "cube": 1,
                "mustard_bottle": 1,
                "potted_meat_can": 1,
                "pentagram": 1,
                "banana": 1,
            },
            2: {

            }
        }

        # Preference of placing objects
        self.obj_scaling = {
            "microwave": 0.2,
        }

        self.valid_place_relation = {
            "plane": self.ON,
            "table": self.ON,
            # "microwave": self.IN,
            # "bowl": self.IN,
            # "mug": self.IN,
            # "cuboid": self.ON,
        }

        obj_names, obj_values = dict2list(self.obj_categories)
        self.obj_back_pool_name = np.array(obj_names)
        self.obj_back_pool_indexes_nums = np.array(obj_values)


    def _load_default_scene(self):
        self.default_scene_name = "table"
        self.default_scene_name_id = {}
        # Plane
        self.planeId = self.loadURDF("plane.urdf", 
                                     basePosition=[0, 0, 0.], 
                                     baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), 
                                     useFixedBase=True)
        # self.default_scene_name_id["plane"] = self.planeId
        self.plane_pose = pu.get_body_pose(self.planeId, client_id=self.client_id)
        plane_qr_region_half_extents = [1., 1., 1.]
        plane_qr_region_pos = [0., 0., 0.]
        plane_qr_region_ori = p.getQuaternionFromEuler([0., 0., 0.])
        planeHalfExtents = [*plane_qr_region_half_extents[:2], 0.]
        self.plane_qr_region_np = self.to_numpy([*plane_qr_region_pos, *plane_qr_region_ori, *plane_qr_region_half_extents])

        # Walls
        # self.wallHalfExtents = [1., 0.05, 0.75]
        # self.wallxId = pu.create_box_body(position=[-1, 0., self.wallHalfExtents[2]], orientation=p.getQuaternionFromEuler([0., 0., np.pi/2]),
        #                                   halfExtents=self.wallHalfExtents, rgba_color=[1, 1, 1, 1], mass=0, client_id=self.client_id)
        # self.wallyId = pu.create_box_body(position=[0., -1, self.wallHalfExtents[2]], orientation=p.getQuaternionFromEuler([0., 0., 0.]),
        #                                     halfExtents=self.wallHalfExtents, rgba_color=[1, 1, 1, 1], client_id=self.client_id)
        # Table
        tableHalfExtents = [0.3, 0.4, 0.2]
        self.tableId = pu.create_box_body(position=[0., 0., tableHalfExtents[2]], orientation=p.getQuaternionFromEuler([0., 0., 0.]),
                                          halfExtents=tableHalfExtents, rgba_color=[1, 1, 1, 1], mass=0., client_id=self.client_id)
        self.table_pose = pu.get_body_pose(self.tableId, client_id=self.client_id)
        self.default_scene_name_id["table"] = self.tableId
        # Default region using table position and table half extents
        table_region_half_extents = [tableHalfExtents[0], tableHalfExtents[1], 0.1]
        # Keep in mind: our queried region is in the object local frame!!
        table_region_pos = [0., 0., tableHalfExtents[2]+table_region_half_extents[2]]
        table_region_ori = p.getQuaternionFromEuler([0., 0., 0.])
        self.table_qr_region = [*table_region_pos, *table_region_ori, *table_region_half_extents]
        
        self.prepare_area = [[-1100., -1100., 0.], [-1000, -1000, 5]]
        init_scene_pc = []
        for name, id in self.default_scene_name_id.items():
            if name == "plane": # Plane needs to use default region to sample
                plane_pc_sample_region = self.to_numpy(plane_qr_region_half_extents[:2]+[0.])
                init_scene_pc.append(self.rng.uniform(-plane_pc_sample_region, plane_pc_sample_region, size=(self.args.max_num_scene_points, 3)))
            else:
                object_pc_world_frame = self.get_obj_pc_from_id(id, num_points=self.args.max_num_scene_points, use_worldpos=False)
                init_scene_pc.append(object_pc_world_frame)
        self.init_scene_pose = self.table_pose if self.default_scene_name == "table" else self.plane_pose
        init_scene_half_extents = tableHalfExtents if self.default_scene_name == "table" else planeHalfExtents
        self.init_scene_bbox = [0., 0., 0., *p.getQuaternionFromEuler([0, 0, 0]), *init_scene_half_extents]
        self.init_scene_pc = np.concatenate(init_scene_pc, axis=0)
        self.init_scene_pc_feature = self.pc_extractor(self.to_torch(self.init_scene_pc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)).squeeze(0).detach().cpu().numpy().astype(self.numpy_dtype)


    def load_objects(self, num_objects=10, random=True, default_scaling=1., init_region=None):
        self.obj_name_id = {}; init_region = init_region if init_region is not None else self.table_qr_region
        self.obj_name_pc = {}; self.obj_name_axes_bbox = {}; self.obj_name_pc_feature = {}; self.obj_stage_name_id = {k:{} for k, v in self.obj_categories.items()}
        self.scene_obj_valid_place_relation = {}; self.obj_name_joint_limits = {}
        if random: # Random choose object categories from the pool and their index
            obj_back_pool_indexes = self.rng.choice(np.arange(len(self.obj_back_pool_name)), num_objects, replace=False)
            objects_act_pool_name = self.obj_back_pool_name[obj_back_pool_indexes]
            objects_act_pool_indexes = self.rng.integers(self.obj_back_pool_indexes_nums[obj_back_pool_indexes])
        else:
            objects_act_pool_name = self.obj_back_pool_name[:num_objects]
            objects_act_pool_indexes = np.array(self.obj_back_pool_indexes_nums, dtype=np.int32)-1 # Index nums start from 1 but index needs to start from 0

        for i, obj_name in enumerate(objects_act_pool_name):
            obj_urdf_file = f"{self.args.asset_root}/{self.args.object_pool_folder}/{obj_name}/{objects_act_pool_indexes[i]}/mobility.urdf"
            assert os.path.exists(obj_urdf_file), f"Object {obj_name} does not exist! Given path: {obj_urdf_file}"

            try: 
                obj_globalScaling = self.specify_obj_scale(obj_name, default_scaling=default_scaling)
                basePosition, baseOrientation = self.rng.uniform(*self.region2axesbbox(init_region)), p.getQuaternionFromEuler([self.rng.uniform(0., np.pi)]*3)
                object_unique_name = f"{obj_name}_{objects_act_pool_indexes[i]}"
                object_id = self.loadURDF(f"{self.args.asset_root}/{self.args.object_pool_folder}/{obj_name}/{objects_act_pool_indexes[i]}/mobility.urdf", 
                                        basePosition=basePosition, baseOrientation=baseOrientation, globalScaling=obj_globalScaling)  # Load an object at position [0, 0, 1]
                obj_mesh_num = pu.get_body_mesh_num(object_id, client_id=self.client_id)
                obj_joints_num = pu.get_num_joints(object_id, client_id=self.client_id)
                self.obj_name_id[object_unique_name] = object_id
                self.obj_name_pc[object_unique_name] = self.get_obj_pc_from_id(object_id, num_points=min(max(256*obj_mesh_num, 1024), self.args.max_num_scene_points), use_worldpos=False)
                self.obj_name_axes_bbox[object_unique_name] = pu.get_obj_axes_aligned_bbox_from_pc(self.obj_name_pc[object_unique_name])
                self.obj_stage_name_id[self.get_obj_stage(obj_name)].update({object_unique_name: object_id})
                # TODO: For object who has self.IN relation, set transparence to 0.5
                if obj_name in self.valid_place_relation.keys():
                    self.scene_obj_valid_place_relation[object_unique_name] = self.valid_place_relation[obj_name]
                if obj_joints_num > 0: # If the object is not movable, we need to set the joints to the lower limit
                    joints_limits = np.array([pu.get_joint_limits(object_id, joint_i, client_id=self.client_id) for joint_i in range(obj_joints_num)])
                    pu.set_joint_positions(object_id, list(range(obj_joints_num)), joints_limits[:, 0], client_id=self.client_id)
                    pu.control_joints(object_id, list(range(obj_joints_num)), joints_limits[:, 0], client_id=self.client_id)
                    self.obj_name_joint_limits[object_unique_name] = joints_limits
            except Exception as e:
                print(f"Failed to load object {obj_name} | Error: {e}")
        
        # Pre-extract the feature for each object and store here 
        with torch.no_grad():
            for object_unique_name, obj_pc in self.obj_name_pc.items(): # Extract the feature for each object
                self.obj_name_pc_feature[object_unique_name] = self.pc_extractor(self.to_torch(obj_pc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)).squeeze(0).detach().cpu().numpy().astype(self.numpy_dtype)
        
        self.num_pool_objs = len(self.obj_name_id)

        self.obj_name_id[self.default_scene_name] = self.default_scene_name_id[self.default_scene_name]
        self.obj_name_pc[self.default_scene_name] = self.init_scene_pc
        self.obj_name_axes_bbox[self.default_scene_name] = self.init_scene_bbox
        self.obj_name_pc_feature[self.default_scene_name] = self.init_scene_pc_feature
        self.obj_stage_name_id.update({0: {self.default_scene_name: self.obj_name_id[self.default_scene_name]}})
        self.scene_obj_valid_place_relation[self.default_scene_name] = self.valid_place_relation[self.default_scene_name]

    
    def load_world(self, num_objects=10, random=True, default_scaling=0.5):
        self._init_pc_extractor()
        self._init_misc_variables()
        self._init_obs_act_space()
        self._load_default_scene()
        self.load_objects(num_objects=num_objects, random=random, default_scaling=default_scaling)
        self.reset_all = True


    def post_checker(self, verbose=False):
        self.failed_objs = []; headers = ["Type", "Env ID", "Name", "ID", "Value"]

        for obj_name, obj_id in self.obj_name_id.items():
            obj_vel = pu.getObjVelocity(obj_id, client_id=self.client_id)
            if not self.vel_checker(obj_vel):
                self.failed_objs.append(["VEL_FAIL", self.client_id, obj_name, obj_id, obj_vel])

        if verbose: 
            # Generate the table and print it; Needs 0.0013s to generate one table
            self.check_table = tabulate(self.failed_objs, headers, tablefmt="pretty")
            print(self.check_table)


    def step(self, action):
        # stepping == 1 means this action is from the agent and is meaningful
        # Pre-physical step
        if self.info['stepping'] == 1.:
            pose_xyz, pose_quat = self.convert_actions(action)
            pu.set_pose(self.selected_obj_id, (pose_xyz, pose_quat), client_id=self.client_id)
            self.traj_history = [[0.]* (7 + 6)] * self.args.max_traj_history_len  # obj_pos dimension + obj_vel dimension
            self.accm_vel_reward = 0.
            self.his_steps = 0
            self.info['stepping'] = 0.
            self.info['his_steps'] = 0
        
        # stepping == 0 means previous action is still running, we need to wait until the object is stable
        # In-pysical step
        if self.info['stepping'] == 0.:
            self.prev_obj_vel = np.array([0.]*6, dtype=self.numpy_dtype)
            for _ in range(ceil(self.args.max_traj_history_len/self.args.step_divider)):
                self.simstep(1/240)
                obj_pos, obj_quat = pu.get_body_pose(self.selected_obj_id, client_id=self.client_id)
                obj_vel = pu.getObjVelocity(self.selected_obj_id, to_array=True, client_id=self.client_id)
                # Update the trajectory history
                self.traj_history[self.his_steps] = obj_pos + obj_quat + obj_vel.tolist()
                # Accumulate velocity reward
                self.accm_vel_reward += -obj_vel[:].__abs__().sum()
                # Jump out if the object is not moving (in the future, we might need to add acceleration checker)
                self.his_steps += 1
                if (self.vel_checker(obj_vel) and self.acc_checker(self.prev_obj_vel, obj_vel)) \
                    or self.his_steps >= self.args.max_traj_history_len:
                    self.info['stepping'] = 1.
                    self.info['his_steps'] = self.his_steps
                    break
                self.prev_obj_vel = obj_vel
        
        # Post-physical step
        reward = self.compute_reward() if self.info['stepping']==1 else 0. # Must compute reward before observation since we use the velocity to compute reward
        done = self.compute_done() if self.info['stepping']==1 else False
        # Success Visualization here since observation needs to be reset. Only for evaluation!
        if self.args.rendering and self.info['stepping']==1:
            print(f"Obj Name: {self.selected_obj_name} | Stable Steps: {self.his_steps}")
            if done and self.info['success'] == 1:
                print(f"Successfully Place {self.success_obj_num[self.cur_stage]} Objects at the stage {self.cur_stage}!")
                if hasattr(self.args, "eval_result") and self.args.eval_result: time.sleep(3.)

        if self.info['stepping'] == 1.: observation = self.compute_observations() if not done else self.reset() # This point should be considered as the start of the episode!
        else: observation = self.last_observation

        # Reset successfully placed object pose
        if self.info['stepping'] == 1.:
            obj_names, obj_poses = dict2list(self.placed_obj_poses)
            for i, obj_name in enumerate(obj_names):
                pu.set_pose(self.obj_name_id[obj_name], obj_poses[i], client_id=self.client_id)

        return observation, reward, done, self.info


    def reset(self):
        if self.reset_all:
            # Reset Training buffer and update the start observation for the next episode
            self.reset_env()
        self.reset_buffer()
        return self.compute_observations()


    def step_manual(self):
        with keyboard.Events() as events:
            while True:
                key = None
                event = events.get(0.0001)
                if event is not None:
                    if isinstance(event, events.Press) and hasattr(event.key, 'char'):
                        key = event.key.char
                if key == 's':
                    self.simstep()
                    self.post_checker(verbose=True)


    #########################################
    ######### Training FUnctions ############
    #########################################

    def _init_obs_act_space(self):
        # Observation space: [scene_pc_feature, obj_pc_feature, bbox, action, history (hitory_len * (obj_pos + obj_vel)))]
        scene_ft_dim = 1024; obj_ft_dim = 1024; qr_region_dim = 10; action_dim = 6; history_dim = self.args.max_traj_history_len * (6 + 7)
        self.observation_shape = (1, scene_ft_dim + obj_ft_dim + qr_region_dim + action_dim + history_dim)
        # Action space: [x, y, z, roll, pitch, yaw]
        self.action_shape = (1, 6)
        # slice
        self.act_scene_feature_slice = slice(0, scene_ft_dim)
        self.selected_obj_feature_slice = slice(self.act_scene_feature_slice.stop, self.act_scene_feature_slice.stop+obj_ft_dim)
        self.placed_region_slice = slice(self.selected_obj_feature_slice.stop, self.selected_obj_feature_slice.stop+qr_region_dim)
        self.last_action_slice = slice(self.placed_region_slice.stop, self.placed_region_slice.stop+qr_region_dim+action_dim)
        self.traj_history_slice = slice(self.last_action_slice.stop, self.last_action_slice.stop+self.args.max_traj_history_len*(6+7))

    
    def _init_misc_variables(self):
        self.args.vel_threshold = self.args.vel_threshold if hasattr(self.args, "vel_threshold") else [0.005, np.pi/1800] # 0.5cm/s and 0.1 degree/s
        self.args.acc_threshold = self.args.acc_threshold if hasattr(self.args, "acc_threshold") else [0.1, np.pi/180] # 0.1m/s^2 and 1degree/s^2
        self.args.max_num_placing_objs = self.args.max_num_placing_objs if hasattr(self.args, "max_num_placing_objs") else [0, 3, 15]
        self.args.max_traj_history_len = self.args.max_traj_history_len if hasattr(self.args, "max_traj_history_len") else 240
        self.args.step_divider = self.args.step_divider if hasattr(self.args, "step_divider") else 6
        self.args.reward_pobj = self.args.reward_pobj if hasattr(self.args, "reward_pobj") else 10
        self.args.vel_reward_scale = self.args.vel_reward_scale if hasattr(self.args, "vel_reward_scale") else 0.005
        self.args.max_stable_steps = self.args.max_stable_steps if hasattr(self.args, "max_stable_steps") else 50
        self.args.max_trials = self.args.max_trials if hasattr(self.args, "max_trials") else 10
        self.args.specific_scene = self.args.specific_scene if hasattr(self.args, "specific_scene") else None
        self.args.max_num_scene_points = self.args.max_num_scene_points if hasattr(self.args, "max_num_scene_points") else 10240
        self.args.max_stage = len(self.args.max_num_placing_objs) - 1 # We start from stage 0 (table)
        # Buffer does not need to be reset
        self.info = {'success': 0., 'stepping': 1., 'his_steps': 0, 'success_placed_obj_num': 0}

    
    def _init_pc_extractor(self):
        self.pc_extractor = get_model(num_class=40, normal_channel=False).to(self.device) # num_classes is used for loading checkpoint to make sure the model is the same
        self.pc_extractor.load_checkpoint(ckpt_path="PointNet_Model/checkpoints/best_model.pth", evaluate=True, map_location=self.device)


    def reset_buffer(self):
        # Training
        self.moving_steps = 0
        # Observations
        self.traj_history = [[0.]* (7 + 6)] * self.args.max_traj_history_len  # obj_pos dimension + obj_vel dimension
        self.last_raw_action = np.zeros(6, dtype=self.numpy_dtype)
        # Rewards
        self.accm_vel_reward = 0.
        # Misc Info
        self.success_obj_num[self.cur_stage] = 0 # Reset the success obj num for the next stage

    
    def reset_env(self):
        # Place all objects to the prepare area
        for obj_name in self.obj_name_id.keys():
            if obj_name == self.default_scene_name: continue # Don't move plane!!!
            pu.set_pose(body=self.obj_name_id[obj_name], pose=(self.rng.uniform(*self.prepare_area), [0., 0., 0., 1.]), client_id=self.client_id)
        self.reset_all = False
        
        # Object
        assert len(self.args.max_num_placing_objs) == len(self.obj_stage_name_id.keys()), \
            f"Max number of placing objects {len(self.args.max_num_placing_objs)} does not match the number of stages {len(self.obj_stage_name_id.keys())}!"
        self.args.max_num_placing_objs = [min(self.args.max_num_placing_objs[i], len(self.obj_stage_name_id[i])) for i in range(len(self.args.max_num_placing_objs))]

        # Scene
        self.cur_stage = 1
        self.placed_obj_poses = {k:{} for k in self.obj_stage_name_id.keys()}
        self.placed_obj_poses[0][self.default_scene_name] = deepcopy(self.init_scene_pose)
        self.success_obj_num = {k:0 for k in self.obj_stage_name_id.keys()}
        self.update_unplaced_objs()
        self.update_unquery_scenes()
        self.obj_done = True
        self.query_scene_done = True

    
    def update_unplaced_objs(self, stage=None, num_objs=None):
        # You can use unplaced_objs to decide how many objs should be placed on the scene
        stage = self.cur_stage if stage is None else stage
        if stage not in self.obj_stage_name_id.keys(): 
            self.unplaced_objs_name_id = {}; return
        selected_obj_pool = self.obj_stage_name_id[stage]
        if len(selected_obj_pool) == 0:
            self.unplaced_objs_name_id = {}; return
        
        num_objs = min(num_objs, self.args.max_num_placing_objs[stage], len(selected_obj_pool)) if num_objs is not None \
                   else min(self.args.max_num_placing_objs[stage], len(selected_obj_pool))

        if self.args.random_select_placing:
            unplaced_objs_name = self.rng.choice(list(selected_obj_pool.keys()), num_objs, replace=False)
        else:
            unplaced_objs_name = list(selected_obj_pool.keys())[:num_objs]
        self.unplaced_objs_name_id = {obj_name: self.obj_name_id[obj_name] for obj_name in unplaced_objs_name}


    def update_unquery_scenes(self, stage=None):
        stage = max(self.cur_stage-1, 0) if stage is None else stage
        if stage not in self.obj_stage_name_id.keys(): 
            self.unquried_scene_name_id = {}; return
        selected_scene_pool_lst = [scene_name for scene_name in self.placed_obj_poses[stage].keys() \
                                   if scene_name in self.scene_obj_valid_place_relation.keys()]
        if len(selected_scene_pool_lst) == 0:
            self.unquried_scene_name_id = {}; return

        if self.args.random_select_placing:
            unquried_scene_name = self.rng.permutation(selected_scene_pool_lst)
        else:
            unquried_scene_name = selected_scene_pool_lst
        self.unquried_scene_name_id = {obj_name: self.obj_name_id[obj_name] for obj_name in unquried_scene_name}
        
        
    def convert_actions(self, action):
        # action shape is (num_env, action_dim) / action is action logits we need to convert it to (0, 1)
        action = action.squeeze(dim=0).sigmoid().cpu().numpy() # Map action to (0, 1)
        action[3:5] = 0. # No x,y rotation
        self.last_raw_action = action.copy()
        
        # action = [x, y, z, roll, pitch, yaw]
        scene_obj_pose = pu.get_body_pose(self.selected_qr_scene_id, client_id=self.client_id)
        placed_qr_region = self.last_observation[self.placed_region_slice]
        half_extents = placed_qr_region[7:]
        untrans_xyz = action[:3] * (2 * half_extents) - half_extents
        # In the qr_scene baseLink frame
        local_action_xyz = p.multiplyTransforms(placed_qr_region[:3], placed_qr_region[3:7], untrans_xyz, [0., 0., 0., 1.])[0]
        # In the simulator world frame
        step_action_xyz = p.multiplyTransforms(scene_obj_pose[0], scene_obj_pose[1], local_action_xyz, [0., 0., 0., 1.])[0]
        step_action_quat = p.getQuaternionFromEuler((action[3:] * 2*np.pi))

        if self.args.rendering:
            world2qr_region = p.multiplyTransforms(*scene_obj_pose, placed_qr_region[:3], placed_qr_region[3:7])
            if hasattr(self, "last_world2qr_region") and world2qr_region == self.last_world2qr_region: pass
            else:
                if hasattr(self, "region_vis_id"): p.removeBody(self.region_vis_id, physicsClientId=self.client_id)
                self.region_vis_id = pu.draw_box_body(position=world2qr_region[0], orientation=world2qr_region[1],
                                                      halfExtents=half_extents, rgba_color=[1, 0, 0, 0.3], client_id=self.client_id)
                self.last_world2qr_region = world2qr_region

        return step_action_xyz, step_action_quat


    def compute_observations(self):
        # We need object description (index), bbox, reward (later)
        # Choose query object
        if self.obj_done:
            self.obj_done = False
            self.selected_obj_name = self.rng.choice(list(self.unplaced_objs_name_id.keys()))
            self.selected_obj_id = self.obj_name_id[self.selected_obj_name]
            self.selected_obj_pc = self.obj_name_pc[self.selected_obj_name].copy()
            self.selected_obj_pc_feature = self.obj_name_pc_feature[self.selected_obj_name].copy()

        # Choose query scene 
        if self.query_scene_done:
            self.query_scene_done = False
            self.selected_qr_scene_name = self.rng.choice(list(self.unquried_scene_name_id.keys()))
            self.selected_qr_scene_id = self.obj_name_id[self.selected_qr_scene_name]
            self.selected_qr_scene_pc = self.obj_name_pc[self.selected_qr_scene_name].copy()
            self.selected_qr_scene_pc_feature = self.obj_name_pc_feature[self.selected_qr_scene_name].copy()
        
        # Compute query region based on the selected object and scene
        selected_obj_bbox = self.obj_name_axes_bbox[self.selected_obj_name]
        selected_scene_bbox = self.obj_name_axes_bbox[self.selected_qr_scene_name]
        # Plane needs to use default region to sample; needs to set scene_bbox
        # We only have on and in relation for now. On and in are pre-annotated.
        if self.scene_obj_valid_place_relation[self.selected_qr_scene_name] == self.ON:
            max_z_half_extent = selected_obj_bbox[9] # bbox is half extent!
            self.selected_qr_region = get_on_bbox(selected_scene_bbox.copy(), z_half_extend=max_z_half_extent)
        elif self.scene_obj_valid_place_relation[self.selected_qr_scene_name] == self.IN:
            max_z_half_extent = max(selected_obj_bbox[9], selected_scene_bbox[9]) # bbox is half extent!
            self.selected_qr_region = get_in_bbox(selected_scene_bbox.copy(), z_half_extend=max_z_half_extent)

        # Convert history to tensor
        his_traj = self.to_numpy(self.traj_history).flatten()

        self.last_observation = np.concatenate([self.selected_qr_scene_pc_feature, self.selected_obj_pc_feature, 
                                                self.selected_qr_region, self.last_raw_action, his_traj])
        
        return self.last_observation


    def compute_reward(self):
        self.moving_steps += 1
        vel_reward = self.args.vel_reward_scale * self.accm_vel_reward
        if self.his_steps <= self.args.max_stable_steps: 
            # Jump to the next object, object is stable within 10 simulation steps
            self.obj_done = True; self.moving_steps = 0; self.success_obj_num[self.cur_stage] += 1
            self.unplaced_objs_name_id.pop(self.selected_obj_name)
            # Record the successful object pose
            selected_obj_pose = pu.get_body_pose(self.selected_obj_id, client_id=self.client_id)
            self.placed_obj_poses[self.cur_stage][self.selected_obj_name] = selected_obj_pose
            vel_reward += len(self.placed_obj_poses[self.cur_stage]) * self.args.reward_pobj
            # Update the scene observation | transform the selected object point cloud to world frame using the current pose
            scene_obj_pose = pu.get_body_pose(self.selected_qr_scene_id, client_id=self.client_id)
            scene_obj2_selected_obj_pose = p.multiplyTransforms(*p.invertTransform(*scene_obj_pose), selected_obj_pose[0], selected_obj_pose[1])
            transformed_selected_obj_pc = self.to_numpy([p.multiplyTransforms(*scene_obj2_selected_obj_pose, point, [0., 0., 0., 1.])[0] for point in self.selected_obj_pc])
            # transformed_selected_obj_pc = tf_apply(self.to_torch(selected_obj_pose[1]), self.to_torch(selected_obj_pose[0]), self.to_torch(self.selected_obj_pc)).cpu().numpy()
            self.selected_qr_scene_pc = np.concatenate([self.selected_qr_scene_pc, transformed_selected_obj_pc], axis=0)
            self.selected_qr_scene_pc = pc_random_downsample(self.selected_qr_scene_pc, self.args.max_num_scene_points)
            # Run one inference needs ~0.5s!
            with torch.no_grad():
                self.selected_qr_scene_pc_feature = self.pc_extractor(self.to_torch(self.selected_qr_scene_pc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)).squeeze(0).cpu().numpy()

            # pu.visualize_pc(self.selected_qr_scene_pc)
            # pu.visualize_pc(self.selected_obj_pc)

        self.last_reward = vel_reward

        return self.last_reward
        

    def compute_done(self):
        done = False # Not real done, done signal means one training episode is done
        # When all cur stage objects have been placed, move to the next scene, refill all objects
        if len(self.unplaced_objs_name_id)==0 or self.moving_steps >= self.args.max_trials:
            self.unquried_scene_name_id.pop(self.selected_qr_scene_name)
            self.query_scene_done = True # Choose new queried scene in the current stage
            self.update_unplaced_objs()  # Refill all objects
            self.obj_done = True         # Choose new object in the current stage
            self.reset_all = False       # Not real done (reset the query scene and query object)
            done = True
            if self.moving_steps >= self.args.max_trials:# Failed to place the object, reset its pose to the prepare area
                pu.set_pose(body=self.obj_name_id[self.selected_obj_name], pose=(self.rng.uniform(*self.prepare_area), [0., 0., 0., 1.]), client_id=self.client_id)              
            
        # When all scene have been queried, move to the next stage, refill scene and objects
        # If there is no next stage (no next stage placed objs or no next stage queried scene), then reset everything
        if len(self.unquried_scene_name_id)==0:
            self.cur_stage += 1
            self.update_unplaced_objs()  # Update next-stage unplaced objects
            self.update_unquery_scenes() # Update next-stage unqueried scenes
            done = True
            if len(self.unplaced_objs_name_id)==0 or len(self.unquried_scene_name_id)==0:
                self.reset_all = True    # Real done (reset the whole environment)
        
        # Record miscs info
        if done:
            self.info['placed_obj_poses'] = self.placed_obj_poses
            self.info['success_placed_obj_num'] = sum(dict2list(self.success_obj_num)[1])
            self.info['success'] = 0.
            if self.cur_stage > len(self.args.max_num_placing_objs):
                self.info['success'] = all([self.success_obj_num[i] >= self.args.max_num_placing_objs[i] for i in range(self.cur_stage-1)])
        return done
        

    ######################################
    ######### Utils FUnctions ############
    ######################################

    def simstep(self, freeze_time=1/240):
        for i in range(ceil(freeze_time * 240)):
            # # step the robot
            # self.robot.simstep()
            # simstep physics
            p.stepSimulation(physicsClientId=self.client_id)
            if self.args.realtime:
                time.sleep(1.0 / 240.0)


    def loadURDF(self, urdf_path, basePosition=None, baseOrientation=None, globalScaling=1.0, useFixedBase=False):
        basePosition = basePosition if basePosition is not None else [0., 0., 0.]
        baseOrientation = baseOrientation if baseOrientation is not None else p.getQuaternionFromEuler([0., 0., 0.])
        return p.loadURDF(urdf_path, basePosition=basePosition, baseOrientation=baseOrientation, 
                          globalScaling=globalScaling, useFixedBase=useFixedBase, physicsClientId=self.client_id)
    

    def get_obj_pc_from_id(self, obj_id, num_points=1024, use_worldpos=False):
        return pu.get_obj_pc_from_id(obj_id, num_points=num_points, use_worldpos=use_worldpos,
                                      rng=self.rng, client_id=self.client_id).astype(self.numpy_dtype)
    

    def get_obj_stage(self, obj_name):
        for stage, obj_names in self.obj_categories.items():
            if obj_name in obj_names.keys(): return stage
        return None
    

    def vel_checker(self, obj_vel):
        return ((obj_vel[:3].__abs__() < self.args.vel_threshold[0]).all() \
                    and (obj_vel[3:].__abs__() < self.args.vel_threshold[1]).all())


    def acc_checker(self, obj_vel, prev_obj_vel):
        acc = np.abs((obj_vel - prev_obj_vel) / (1/240))
        return (acc[:3] < self.args.acc_threshold[0]).all() \
                    and (acc[3:] < self.args.acc_threshold[1]).all()
    

    def region2axesbbox(self, region):
        ## region: [x, y, z, qx, qy, qz, w, half_x, half_y, half_z]
        region = self.to_numpy(region)
        lower_bound = region[:3] - region[7:]
        upper_bound = region[:3] + region[7:]
        return [lower_bound, upper_bound]
    

    def specify_obj_scale(self, obj_name, default_scaling=1.):
        if obj_name in self.obj_scaling.keys(): return self.obj_scaling[obj_name]
        else: return default_scaling


    def to_torch(self, x, dtype=None):
        dtype = dtype if dtype is not None else self.tensor_dtype
        return torch.tensor(x, dtype=dtype, device=self.device)
    

    def to_numpy(self, x, dtype=None):
        dtype = dtype if dtype is not None else self.numpy_dtype
        return np.array(x, dtype=dtype)
    

    def close(self):
        p.disconnect(physicsClientId=self.client_id)
    

if __name__=="__main__":
    from argparse import ArgumentParser

    args = ArgumentParser()
    args.rendering = True
    args.debug = False
    args.asset_root = "assets"
    args.object_pool_folder = "objects"
    args.num_pool_objs = 100
    args.max_num_placing_objs = [0, 3, 16]
    args.random_select_pool = False
    args.random_select_placing = True
    args.default_scaling = 0.5
    args.realtime = True
    args.force_threshold = 20.
    args.vel_threshold = [0.005, np.pi/1800] # This two threshold need to be tuned
    args.acc_threshold = [0.1, np.pi/180]
    args.eval_result = True
    args.specific_scene = "microwave_0"

    env = RoboSensaiBullet(args)

    all_pc = np.concatenate(list(env.obj_name_pc.values()) + [env.init_scene_pc], axis=0)
    pu.visualize_pc(all_pc)

    while True:
        random_action = (torch.rand((1, 6), device=env.device) * 2 - 1) * 5
        _, _, done, _ = env.step(random_action)
        # print(done)
        # input("Press Enter to continue...")

    # env.step_manual()
