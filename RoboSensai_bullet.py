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
from torch_utils import tf_combine, quat_from_euler
from Blender_script.PybulletRecorder import PyBulletRecorder


class RoboSensaiBullet:
    def __init__(self, args=None) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.numpy_dtype = np.float16 if (hasattr(self.args, "use_bf16") and self.args.use_bf16) else np.float32
        self.tensor_dtype = torch.bfloat16 if (hasattr(self.args, "use_bf16") and self.args.use_bf16) else torch.float32
        self.rng = np.random.default_rng(args.seed if hasattr(args, "seed") else None)
        self._init_simulator()
        self.update_objects_database()
        self.load_world()
        self.reset()


    def _init_simulator(self):
        connect_type = p.GUI if self.args.rendering else p.DIRECT
        self.client_id = p.connect(connect_type)#or p.DIRECT for non-graphical version
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if not self.args.debug:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.client_id)
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)


    def update_objects_database(self):
        # Right now we only have two stages, object can not appear in both stages! We need to figure out how to deal with this problem
        self.obj_dataset_folder = os.path.join(self.args.asset_root, self.args.object_pool_folder)
        # 0: "Table", "Bookcase", "Dishwasher", "Microwave", all storage furniture
        self.obj_uni_names_dataset = {}
        obj_categories = sorted(os.listdir(self.obj_dataset_folder))
        for cate in obj_categories:
            obj_folder = os.path.join(self.obj_dataset_folder, cate)
            obj_indexes = sorted(os.listdir(obj_folder))
            for idx in obj_indexes:
                obj_uni_name = f"{cate}_{idx}"
                obj_urdf_path = f"{self.obj_dataset_folder}/{cate}/{idx}/mobility.urdf"
                obj_label_path = f"{self.obj_dataset_folder}/{cate}/{idx}/label.json"
                assert os.path.exists(obj_urdf_path), f"Object {obj_uni_name} does not exist! Given path: {obj_urdf_path}"
                assert os.path.exists(obj_label_path), f"Object {obj_uni_name} does not exist! Given path: {obj_label_path}"
                self.obj_uni_names_dataset.update({obj_uni_name: {"urdf": obj_urdf_path, "label": read_json(obj_label_path)}})
        
        # Load fixed scenes path
        self.fixed_scene_dataset_folder = os.path.join(self.args.asset_root, self.args.scene_pool_folder)
        self.scene_uni_names_dataset = {}
        scene_categories = sorted(os.listdir(self.fixed_scene_dataset_folder))
        for scene in scene_categories:
            scene_pool_folder = os.path.join(self.fixed_scene_dataset_folder, scene)
            scene_indexes = sorted(os.listdir(scene_pool_folder))
            for idx in scene_indexes:
                scene_uni_name = f"{scene}_{idx}"
                scene_urdf_path = f"{self.fixed_scene_dataset_folder}/{scene}/{idx}/mobility.urdf"
                scene_label_path = f"{self.fixed_scene_dataset_folder}/{scene}/{idx}/label.json"
                assert os.path.exists(scene_urdf_path), f"Scene {scene_uni_name} does not exist! Given path: {scene_urdf_path}"
                assert os.path.exists(scene_label_path), f"Scene {scene_uni_name} does not exist! Given path: {scene_label_path}"
                self.scene_uni_names_dataset.update({scene_uni_name: {"urdf": scene_urdf_path, "label": read_json(scene_label_path)}})


    def load_scenes(self):
        self.fixed_scene_name_data = {}; self.unquried_scene_name = []
        self.prepare_area = [[-1100., -1100., 0.], [-1000, -1000, 100]]
        # Plane
        planeHalfExtents = [1., 1., 0.]
        planeId = self.loadURDF("plane.urdf", 
                                basePosition=[0, 0, 0.], 
                                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), 
                                useFixedBase=True)
        pu.change_obj_color(planeId, rgba_color=[1., 1., 1., 0.2])
        qr_pose, qr_ori, qr_half_extents = [0., 0., 0.], p.getQuaternionFromEuler([0., 0., 0.]), [1., 1., 1.]
        plane_pc_sample_region = self.to_numpy(planeHalfExtents)
        plane_pc = self.rng.uniform(-plane_pc_sample_region, plane_pc_sample_region, size=(self.args.max_num_urdf_points, 3))
        plane_bbox = [0., 0., 0., *p.getQuaternionFromEuler([0, 0, 0]), *planeHalfExtents]
        # self.fixed_scene_name_data["plane"] = {"id": planeId,
        #                                         "init_pose": [0., 0., 0.],
        #                                         "bbox": plane_bbox,
        #                                         "queried_region": "on", 
        #                                         "pc": plane_pc,
        
        # Table
        tableHalfExtents = [0.4, 0.5, 0.35]
        tableId = pu.create_box_body(position=[0., 0., tableHalfExtents[2]], orientation=p.getQuaternionFromEuler([0., 0., 0.]),
                                          halfExtents=tableHalfExtents, rgba_color=[1, 1, 1, 1], mass=0., client_id=self.client_id)
        
        # Default region using table position and table half extents
        qr_pose, qr_ori, qr_half_extents = [0., 0., tableHalfExtents[2]+0.1], p.getQuaternionFromEuler([0., 0., 0.]), [*tableHalfExtents[:2], 0.1]
        table_pc = self.get_obj_pc_from_id(tableId, num_points=self.args.max_num_urdf_points, use_worldpos=False)
        table_axes_bbox = [0., 0., 0., *p.getQuaternionFromEuler([0, 0, 0]), *tableHalfExtents]
        self.fixed_scene_name_data["table"] = {"id": tableId,
                                               "init_z_offset": 0.0,
                                               "init_pose": ([0., 0., tableHalfExtents[2]], p.getQuaternionFromEuler([0., 0., 0.])),
                                               "bbox": table_axes_bbox,
                                               "queried_region": "on", 
                                               "pc": table_pc
                                              }
        # All other fixed scenes, using for loop to load later; All scene bbox are in the baselink frame not world frame! Baselink are at the origin of world frame!
        if self.args.random_select_scene_pool: # Randomly choose scene categories from the pool and their index
            selected_scene_pool = self.rng.choice(list(self.scene_uni_names_dataset.keys()), min(self.args.num_pool_scenes-1, len(self.scene_uni_names_dataset)), replace=False).tolist()
            if self.args.num_pool_scenes > len(self.scene_uni_names_dataset): print(f"WARNING: Only {len(self.scene_uni_names_dataset)} scenes are loaded!")
        else:
            selected_scene_pool = list(self.scene_uni_names_dataset.keys())[:self.args.num_pool_scenes-1]
        for scene_uni_name in selected_scene_pool:
            scene_urdf_path = self.scene_uni_names_dataset[scene_uni_name]["urdf"]
            scene_label = self.scene_uni_names_dataset[scene_uni_name]["label"]
            try:
                basePosition, baseOrientation = self.rng.uniform([-5, -5, 0.], [5, 5, 10]), p.getQuaternionFromEuler([self.rng.uniform(0., np.pi)]*3)
                scene_id = self.loadURDF(scene_urdf_path, basePosition=basePosition, baseOrientation=baseOrientation, globalScaling=scene_label["globalScaling"], useFixedBase=True)
                scene_mesh_num = pu.get_body_mesh_num(scene_id, client_id=self.client_id)
                scene_pc = self.get_obj_pc_from_id(scene_id, num_points=self.args.max_num_urdf_points, use_worldpos=False)
                scene_axes_bbox = pu.get_obj_axes_aligned_bbox_from_pc(scene_pc)
                stable_init_pos_z = scene_axes_bbox[9] - scene_axes_bbox[2] # Z Half extent - Z offset between pc center and baselink
                init_z_offset = 5.
                assert scene_label["queried_region"] is not None, f"Scene {scene_uni_name} does not have queried region!"
                self.fixed_scene_name_data[scene_uni_name] = {  # Init_pose is floating on the air to avoid rely on the ground
                                                                "id": scene_id,
                                                                "init_z_offset": init_z_offset,
                                                                "init_pose": ([0., 0., stable_init_pos_z+init_z_offset], p.getQuaternionFromEuler([0., 0., 0.])), # For training, we hang the object in the air
                                                                "init_stable_pose": ([0., 0., stable_init_pos_z], p.getQuaternionFromEuler([0., 0., 0.])), # For evaluation, we place the object on the ground
                                                                "bbox": scene_axes_bbox,
                                                                "queried_region": scene_label["queried_region"], 
                                                                "pc": scene_pc,
                                                                "mass": pu.get_mass(scene_id, client_id=self.client_id),
                                                            }
                
                # For scene who has "in" relation, set transparence to 0.5
                if scene_label["queried_region"] == "in":
                    pu.change_obj_color(scene_id, rgba_color=[0, 0, 0, 0.2], client_id=self.client_id)
                
                # If the scene has joints, we need to set the joints to the lower limit
                self.fixed_scene_name_data[scene_uni_name]["joint_limits"] = self.set_obj_joints_to_lower_limit(scene_id)
            
            except p.error as e:
                print(f"Failed to load scene {scene_uni_name} | Error: {e}")


    def load_objects(self):
        self.obj_name_data = {}; self.unplaced_objs_name = []; self.queriable_obj_names = [] # The object can be queried or placed
        if self.args.random_select_objs_pool: # Randomly choose object categories from the pool and their index
            cate_uni_names = self.rng.choice(list(self.obj_uni_names_dataset.keys()), min(self.args.num_pool_objs, len(self.obj_uni_names_dataset)), replace=False).tolist()
            if self.args.num_pool_objs > len(self.obj_uni_names_dataset): print(f"WARNING: Only {len(self.obj_uni_names_dataset)} objects are loaded!")
        else:
            cate_uni_names = list(self.obj_uni_names_dataset.keys())[:self.args.num_pool_objs]

        for i, obj_uni_name in enumerate(cate_uni_names):
            obj_urdf_path = self.obj_uni_names_dataset[obj_uni_name]["urdf"]
            obj_label = self.obj_uni_names_dataset[obj_uni_name]["label"]
            try: 
                rand_basePosition, rand_baseOrientation = self.rng.uniform([-5, -5, 0.], [5, 5, 10]), p.getQuaternionFromEuler([self.rng.uniform(0., np.pi)]*3)
                object_id = self.loadURDF(obj_urdf_path, basePosition=rand_basePosition, baseOrientation=rand_baseOrientation, globalScaling=obj_label["globalScaling"])  # Load an object at position [0, 0, 1]
                obj_mesh_num = pu.get_body_mesh_num(object_id, client_id=self.client_id)
                obj_pc = self.get_obj_pc_from_id(object_id, num_points=self.args.max_num_urdf_points, use_worldpos=False)
                obj_axes_bbox = pu.get_obj_axes_aligned_bbox_from_pc(obj_pc)
                obj_init_pos_z = obj_axes_bbox[9] - obj_axes_bbox[2] # Z Half extent - Z offset between pc center and baselink
                self.obj_name_data[obj_uni_name] = {
                                                    "id": object_id,
                                                    "init_pose": ([0., 0., obj_init_pos_z+5.], p.getQuaternionFromEuler([0., 0., 0.])),
                                                    "init_stable_pose": ([0., 0., obj_init_pos_z], p.getQuaternionFromEuler([0., 0., 0.])),
                                                    "bbox": obj_axes_bbox,
                                                    "queried_region": obj_label["queried_region"], 
                                                    "pc": obj_pc,
                                                    "mass": pu.get_mass(object_id, client_id=self.client_id),
                                                    }

                # object queried region; This is a dataset bug that some objects are labeled as string "None"
                if obj_label["queried_region"] != None and obj_label["queried_region"] != "None":
                    self.queriable_obj_names.append(obj_uni_name)
                    # For object who has "in" relation, set transparence to 0.5
                    if obj_label["queried_region"] == "in":
                        pu.change_obj_color(object_id, rgba_color=[0, 0, 0, 0.5], client_id=self.client_id)
                
                # If the object has joints, we need to set the joints to the lower limit
                self.obj_name_data[obj_uni_name]["joint_limits"] = self.set_obj_joints_to_lower_limit(object_id)
            
            except p.error as e:
                print(f"Failed to load object {obj_uni_name} | Error: {e}")
        
        num_queriable_scenes = len(self.queriable_obj_names) + len(self.fixed_scene_name_data.keys()) if not self.args.fixed_scene_only else len(self.fixed_scene_name_data.keys())
        assert num_queriable_scenes >= self.args.max_num_qr_scenes, f"Only {num_queriable_scenes} scenes are loaded, but we need {self.args.max_num_qr_scenes} scenes!"
        assert len(self.obj_name_data) >= self.args.max_num_placing_objs, f"Only {len(self.obj_name_data)} objects are loaded, but we need {self.args.max_num_placing_objs} objects!"

    
    def load_world(self):
        self._init_misc_variables()
        self._init_obs_act_space()
        self.load_scenes()
        self.load_objects()
        if self.args.blender_record: 
            self.pybullet_recorder = PyBulletRecorder(client_id=self.client_id)


    def post_checker(self, verbose=False):
        self.failed_objs = []; headers = ["Type", "Env ID", "Name", "ID", "Value"]

        for obj_name in self.obj_name_data.key():
            obj_id = self.obj_name_data[obj_name]["id"]
            obj_vel = pu.getObjVelocity(obj_id, client_id=self.client_id)
            if not self.vel_checker(obj_vel):
                self.failed_objs.append(["VEL_FAIL", self.client_id, obj_name, obj_id, obj_vel])

        if verbose: 
            # Generate the table and print it; Needs 0.0013s to generate one table
            self.check_table = tabulate(self.failed_objs, headers, tablefmt="pretty")
            print(self.check_table)


    #########################################
    ######### Training Functions ############
    #########################################

    def _init_obs_act_space(self):
        # Observation space: [scene_pc_feature, obj_pc_feature, bbox, action, history (hitory_len * (obj_pos + obj_vel)))]
        # 1 is the env number to align with the isaacgym env
        # We have two kinds of observation: seq_obs [qr_region, prev_action, obj_sim_history]; pc_obs [scene_pc, obj_pc]
        self.qr_region_dim = 10; self.action_dim = 6; self.traj_hist_dim = self.args.max_traj_history_len*(6+7)
        self.raw_act_hist_qr_obs_shape = (1, self.args.sequence_len, self.qr_region_dim + self.action_dim + self.traj_hist_dim)

        self.history_ft_dim = self.traj_hist_dim if not self.args.use_traj_encoder else 512; 
        self.qr_region_ft_dim = self.qr_region_dim; self.action_ft_dim = self.action_dim
        self.post_act_hist_qr_ft_shape = (1, self.args.sequence_len, self.qr_region_ft_dim + self.action_ft_dim + self.history_ft_dim)
        
        self.scene_ft_dim = 1024; self.obj_ft_dim = 1024; self.seq_info_ft_dim = 2048
        self.post_observation_shape = (1, self.seq_info_ft_dim + self.scene_ft_dim + self.obj_ft_dim)
        
        # Action space: [x, y, z, roll, pitch, yaw]
        self.action_shape = (1, 6)
        # trajectory shape
        self.traj_history_shape = (self.args.max_traj_history_len, 6+7) # obj_pos dimension + obj_vel dimension
        # slice
        self.qr_region_slice = slice(0, self.qr_region_dim)
        self.action_slice = slice(self.qr_region_slice.stop, self.qr_region_slice.stop+self.action_dim)
        self.traj_history_slice = slice(self.action_slice.stop, self.action_slice.stop+self.traj_hist_dim)

    
    def _init_misc_variables(self):
        self.args.vel_threshold = self.args.vel_threshold if hasattr(self.args, "vel_threshold") else [0.005, np.pi/360] # 0.5cm/s and 0.1 degree/s
        self.args.acc_threshold = self.args.acc_threshold if hasattr(self.args, "acc_threshold") else [0.1, np.pi/36] # 0.1m/s^2 and 1degree/s^2
        self.args.max_num_placing_objs = self.args.max_num_placing_objs if hasattr(self.args, "max_num_placing_objs") else 16
        self.args.max_traj_history_len = self.args.max_traj_history_len if hasattr(self.args, "max_traj_history_len") else 240
        self.args.step_divider = self.args.step_divider if hasattr(self.args, "step_divider") else 6
        self.args.reward_pobj = self.args.reward_pobj if hasattr(self.args, "reward_pobj") else 10
        self.args.vel_reward_scale = self.args.vel_reward_scale if hasattr(self.args, "vel_reward_scale") else 0.005
        self.args.max_stable_steps = self.args.max_stable_steps if hasattr(self.args, "max_stable_steps") else 50
        self.args.min_continue_stable_steps = self.args.min_continue_stable_steps if hasattr(self.args, "min_continue_stable_steps") else 20
        self.args.max_trials = self.args.max_trials if hasattr(self.args, "max_trials") else 10
        self.args.specific_scene = self.args.specific_scene if hasattr(self.args, "specific_scene") else None
        self.args.max_num_urdf_points = self.args.max_num_urdf_points if hasattr(self.args, "max_num_urdf_points") else 2048
        self.args.max_num_scene_points = self.args.max_num_scene_points if hasattr(self.args, "max_num_scene_points") else 10240
        self.args.fixed_qr_region = self.args.fixed_qr_region if hasattr(self.args, "fixed_qr_region") else False
        self.args.use_traj_encoder = self.args.use_traj_encoder if hasattr(self.args, "use_traj_encoder") else False
        self.args.blender_record = self.args.blender_record if hasattr(self.args, "blender_record") else False
        # Buffer does not need to be reset
        self.info = {'success': 0., 'stepping': 1., 'his_steps': 0, 'success_placed_obj_num': 0, 'selected_qr_scene_name': None, 
                     'obj_success_rate': {}, 'scene_obj_success_num': {}, 'pc_change_indicator': 1.}
        self.num_episode = 0
        self.default_qr_region_z = 0.3

    
    def reset_env(self):
        # Place all objects to the prepare area
        for obj_name in self.obj_name_data.keys():
            pu.set_pose(body=self.obj_name_data[obj_name]["id"], 
                pose=(self.rng.uniform(*self.prepare_area), [0., 0., 0., 1.]), client_id=self.client_id)
        
        # Place all scenes to the prepare area
        for scene_name in self.fixed_scene_name_data.keys():
            pu.set_pose(body=self.fixed_scene_name_data[scene_name]["id"], 
                pose=(self.rng.uniform(*self.prepare_area), [0., 0., 0., 1.]), client_id=self.client_id)
        
        # Scene
        self.placed_obj_poses = {}
        self.success_obj_num = 0
        if len(self.unquried_scene_name) == 0: 
            self.update_unquery_scenes()
        self.query_scene_done = True
        self.obj_done = True

        # Training
        self.cur_trial = 1
        # Observations
        self.traj_history = [[0.]* (7 + 6)] * self.args.max_traj_history_len  # obj_pos dimension + obj_vel dimension
        self.last_raw_action = np.zeros(6, dtype=self.numpy_dtype)
        self.last_seq_obs = np.zeros(self.raw_act_hist_qr_obs_shape[1:], dtype=self.numpy_dtype)
        # Rewards
        self.accm_vel_reward = 0.

        # Blender record
        if self.args.blender_record: 
            self.pybullet_recorder.reset(links=True)
            [self.pybullet_recorder.add_keyframe() for _ in range(120)] # Add 0.5s keyframes to show the start

    
    def update_unplaced_objs(self):
        # You can use unplaced_objs to decide how many objs should be placed on the scene
        selected_obj_pool = list(self.obj_name_data.keys())
        self.unplaced_objs_name = []
        
        # Sequence selection to check the dataset stabability (only for evaluation)
        if hasattr(self.args, "seq_select_placing") and self.args.seq_select_placing:
            if not hasattr(self, "cur_seq_index"): self.cur_seq_index = 0
            candidate_obj_name = selected_obj_pool[self.cur_seq_index:self.cur_seq_index+self.args.max_num_placing_objs]
            self.unplaced_objs_name.extend(candidate_obj_name)
            self.cur_seq_index = (self.cur_seq_index + self.args.max_num_placing_objs) % len(selected_obj_pool)
            print(f"Current Sequence Index: {self.cur_seq_index}; NUmber of pool objs: {len(selected_obj_pool)}")
        else:
            while True:
                candidate_obj_name = self.rng.choice(selected_obj_pool) if self.args.random_select_placing else selected_obj_pool[0]
                selected_obj_pool.remove(candidate_obj_name)
                if candidate_obj_name == self.selected_qr_scene_name: continue

                # Use scene bbox and obj bbox to do a simple filtering (especially for the "in" relation)
                candidate_obj_bbox = self.obj_name_data[candidate_obj_name]["bbox"]
                if self.selected_qr_scene_region == "in": # obj bbox dimension should be smaller than scene bbox dimension
                    # XYZ dimension should be smaller than scene dimension
                    if all([candidate_obj_bbox[i]<self.selected_qr_scene_bbox[i] for i in range(7, 10)]):
                        self.unplaced_objs_name.append(candidate_obj_name)
                elif self.selected_qr_scene_region == "on":
                    # XY dimension should be smaller than scene dimension
                    if all([candidate_obj_bbox[i]<self.selected_qr_scene_bbox[i] for i in range(7, 9)]):
                        self.unplaced_objs_name.append(candidate_obj_name)
                else:
                    raise NotImplementedError(f"Scene region {self.selected_qr_scene_region} is not implemented!")
                
                if len(self.unplaced_objs_name) >= self.args.max_num_placing_objs or len(selected_obj_pool)==0: break


    def update_unquery_scenes(self):
        self.unquried_scene_name = list(self.fixed_scene_name_data.keys())
        if not self.args.fixed_scene_only:
            self.unquried_scene_name.extend(self.queriable_obj_names)
        self.unquried_scene_name = self.unquried_scene_name[:self.args.max_num_qr_scenes]
        
        
    def convert_actions(self, action):
        # action shape is (num_env, action_dim) / action is action logits we need to convert it to (0, 1)
        action = action.squeeze(dim=0).sigmoid().cpu().numpy() # Map action to (0, 1)
        action[3:5] = 0. # No x,y rotation
        self.last_raw_action = action.copy()
        
        # action = [x, y, z, roll, pitch, yaw]
        World_2_QRscene = pu.get_body_pose(self.selected_qr_scene_id, client_id=self.client_id)
        QRscene_2_QRregion = self.selected_qr_region
        half_extents = QRscene_2_QRregion[7:]
        QRregion_2_ObjCenter = action[:3] * (2 * half_extents) - half_extents
        QRregion_2_ObjBase_xyz = p.multiplyTransforms(QRregion_2_ObjCenter, [0, 0, 0, 1.], -self.selected_obj_bbox[:3], [0, 0, 0, 1.])[0]
        # In the qr_scene baseLink frame
        QRsceneBase_2_ObjBase_xyz = p.multiplyTransforms(QRscene_2_QRregion[:3], QRscene_2_QRregion[3:7], QRregion_2_ObjBase_xyz, [0., 0., 0., 1.])[0]
        # In the simulator world frame
        World_2_ObjBase_xyz = p.multiplyTransforms(World_2_QRscene[0], World_2_QRscene[1], QRsceneBase_2_ObjBase_xyz, [0., 0., 0., 1.])[0]
        World_2_ObjBase_quat = p.getQuaternionFromEuler((action[3:] * 2*np.pi))

        if self.args.rendering:
            world2qr_region = p.multiplyTransforms(*World_2_QRscene, QRscene_2_QRregion[:3], QRscene_2_QRregion[3:7])
            if hasattr(self, "last_world2qr_region") and world2qr_region == self.last_world2qr_region: pass
            else:
                if hasattr(self, "region_vis_id"): p.removeBody(self.region_vis_id, physicsClientId=self.client_id)
                self.region_vis_id = pu.draw_box_body(position=world2qr_region[0], orientation=world2qr_region[1],
                                                      halfExtents=half_extents, rgba_color=[1, 0, 0, 0.1], client_id=self.client_id)
                self.last_world2qr_region = world2qr_region

        return World_2_ObjBase_xyz, World_2_ObjBase_quat


    def step(self, action):
        # stepping == 1 means this action is from the agent and is meaningful
        # Pre-physical step
        if self.info['stepping'] == 1.:
            pose_xyz, pose_quat = self.convert_actions(action)
            pu.set_pose(self.selected_obj_id, (pose_xyz, pose_quat), client_id=self.client_id)
            self.traj_history = [[0.]* (7 + 6)] * self.args.max_traj_history_len  # obj_pos dimension + obj_vel dimension
            self.accm_vel_reward = 0.
            self.his_steps = 0
            self.continue_stable_steps = 0
            self.info['stepping'] = 0.
            self.info['his_steps'] = 0
            self.info['pc_change_indicator'] = 0.
        
        # stepping == 0 means previous action is still running, we need to wait until the object is stable
        # In-pysical step
        if self.info['stepping'] == 0.:
            self.prev_obj_vel = np.array([0.]*6, dtype=self.numpy_dtype)
            for _ in range(ceil(self.args.max_traj_history_len/self.args.step_divider)):
                self.simstep(1/240)
                self.blender_record()

                obj_pos, obj_quat = pu.get_body_pose(self.selected_obj_id, client_id=self.client_id)
                obj_vel = pu.getObjVelocity(self.selected_obj_id, to_array=True, client_id=self.client_id)
                # Update the trajectory history [0, 0, 0, ..., T0, T1..., Tn]
                self.traj_history.pop(0)
                self.traj_history.append(obj_pos + obj_quat + obj_vel.tolist())
                # Accumulate velocity reward
                self.accm_vel_reward += -obj_vel[:].__abs__().sum()
                # Jump out if the object is not moving (in the future, we might need to add acceleration checker)
                self.his_steps += 1
                if (self.vel_checker(obj_vel) and self.acc_checker(self.prev_obj_vel, obj_vel)):
                    self.continue_stable_steps += 1
                    if self.continue_stable_steps >= self.args.min_continue_stable_steps: 
                        self.info['stepping'] = 1.
                        self.info['his_steps'] = self.his_steps
                        break
                else: self.continue_stable_steps = 0

                if self.his_steps >= self.args.max_traj_history_len:
                    self.info['stepping'] = 1.
                    self.info['his_steps'] = self.his_steps
                    break

                self.prev_obj_vel = obj_vel
        
        # Post-physical step
        reward = self.compute_reward() if self.info['stepping']==1 else 0. # Must compute reward before observation since the velocity will be reset in the observaion. Only stepping environment will be recorded
        done = self.compute_done() if self.info['stepping']==1 else False
        # Success Visualization here since observation needs to be reset. Only for evaluation!
        if self.args.rendering and self.info['stepping']==1:
            print(f"Placing {self.selected_obj_name} {self.selected_qr_scene_region} the {self.selected_qr_scene_name} | Stable Steps: {self.his_steps} | Trial: {self.cur_trial}")
            if done and self.info['success'] == 1:
                print(f"Successfully Place {self.success_obj_num} Objects {self.selected_qr_scene_region} the {self.selected_qr_scene_name}!")
                if hasattr(self.args, "eval_result") and self.args.eval_result: time.sleep(3.)

        # This point should be considered as the start of the episode! Stablebaseline3 will automatically reset the environment when done is True; Therefore our environment does not have reset function called, it is called outside.
        if self.info['stepping'] == 1. and not done: observation = self.compute_observations()
        else: observation = self.last_seq_obs

        # Reset successfully placed object pose
        if self.info['stepping'] == 1.:
            obj_names, obj_poses = dict2list(self.placed_obj_poses)
            for i, obj_name in enumerate(obj_names):
                pu.set_pose(self.obj_name_data[obj_name]["id"], obj_poses[i], client_id=self.client_id)

        return observation, reward, done, self.info


    def reset(self):
        # Since we can not load all of the scenes and objects at one time for training, we need to reset the environment at the certain number of episodes for training.
        if self.num_episode >= self.args.num_episode_to_replace_pool:
            self.num_episode = 0 # Reset the episode counter!!
            p.resetSimulation(physicsClientId=self.client_id)
            p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
            self.load_scenes()
            self.load_objects()
        self.reset_env()
        return self.compute_observations()

    
    def compute_observations(self):
        # Choose query scene 
        if self.query_scene_done:
            self.query_scene_done = False
            if hasattr(self, "selected_qr_scene_id"): # Move the previous scene to the prepare area
                pu.set_pose(body=self.selected_qr_scene_id, pose=(self.rng.uniform(*self.prepare_area), [0., 0., 0., 1.]), client_id=self.client_id) # Reset the pose of the scene
            while True:
                self.selected_qr_scene_name = self.rng.choice(list(self.unquried_scene_name))
                self.scene_name_data = self.obj_name_data if self.selected_qr_scene_name in self.obj_name_data.keys() else self.fixed_scene_name_data
                self.selected_qr_scene_id = self.scene_name_data[self.selected_qr_scene_name]["id"]
                self.selected_qr_scene_pc = self.scene_name_data[self.selected_qr_scene_name]["pc"].copy()
                self.selected_qr_scene_bbox = self.scene_name_data[self.selected_qr_scene_name]["bbox"]
                self.selected_qr_scene_region = self.scene_name_data[self.selected_qr_scene_name]["queried_region"]
                self.tallest_placed_half_z_extend = 0.
                self.scene_init_stable_pose = self.scene_name_data[self.selected_qr_scene_name]["init_pose"]
                pu.set_pose(body=self.selected_qr_scene_id, pose=self.scene_init_stable_pose, client_id=self.client_id) # Reset the pose of the scene
                pu.set_mass(self.selected_qr_scene_id, mass=0., client_id=self.client_id) # Set the scene mass to 0

                self.update_unplaced_objs() # Refill all objects based on the new selected scene when the queried scene changed
                if len(self.unplaced_objs_name) > 0: break
                else: # This scene has no objects to place based on the naive filtering, remove it from the unquery scene list
                    self.unquried_scene_name.remove(self.selected_qr_scene_name)
                    if len(self.unquried_scene_name) == 0: self.update_unquery_scenes()
            
            self.blender_register(self.selected_qr_scene_id, self.selected_qr_scene_name)
            
        # We need object description (index), bbox, reward (later)
        # Choose query object placement order
        if self.obj_done:
            self.obj_done = False
            self.selected_obj_name = self.rng.choice(list(self.unplaced_objs_name))
            self.selected_obj_id = self.obj_name_data[self.selected_obj_name]["id"]
            self.selected_obj_pc = self.obj_name_data[self.selected_obj_name]["pc"].copy()
            self.selected_obj_bbox = self.obj_name_data[self.selected_obj_name]["bbox"]
            pu.set_mass(self.selected_obj_id, self.obj_name_data[self.selected_obj_name]["mass"], client_id=self.client_id)
            self.info['pc_change_indicator'] = 1.

            self.blender_register(self.selected_obj_id, self.selected_obj_name)

        # Compute query region area based on the selected object and scene
        if self.selected_qr_scene_region == "on":
            max_z_half_extent = self.tallest_placed_half_z_extend + self.selected_obj_bbox[9] # If on, bbox is half extent + tallest placed obj half extent (equivalent to the current scene bbox).
            if self.args.fixed_qr_region: max_z_half_extent = self.default_qr_region_z
            self.selected_qr_region = get_on_bbox(self.selected_qr_scene_bbox.copy(), z_half_extend=max_z_half_extent)
        elif self.selected_qr_scene_region == "in":
            max_z_half_extent = self.selected_obj_bbox[9] # max z-half extend is half extent of the object! we can not handle the case that if some objects already placed on the scene and we want to stack on them.
            self.selected_qr_region = get_in_bbox(self.selected_qr_scene_bbox.copy(), z_half_extend=max_z_half_extent)
        else:
            raise NotImplementedError(f"Object {self.selected_qr_scene_name} Queried region {self.selected_qr_scene_region} is not implemented!")

        # Update the queried scene points cloud
        self.info["selected_qr_scene_pc"] = self.selected_qr_scene_pc
        self.info["selected_obj_pc"] = self.selected_obj_pc
        # Update the sequence observation | pop the first observation and append the last observation
        his_traj = self.to_numpy(self.traj_history).flatten()
        cur_seq_obs = np.concatenate([self.selected_qr_region, self.last_raw_action, his_traj])
        self.last_seq_obs = np.concatenate([self.last_seq_obs[1:, :], np.expand_dims(cur_seq_obs, axis=0)])
        
        return self.last_seq_obs


    def compute_reward(self):
        self.cur_trial += 1
        vel_reward = self.args.vel_reward_scale * self.accm_vel_reward
        if self.his_steps <= self.args.max_stable_steps: 
            # This object is successfully placed!! Jump to the next object, object is stable within 10 simulation steps
            self.obj_done = True; self.cur_trial = 1; self.success_obj_num += 1
            self.unplaced_objs_name.remove(self.selected_obj_name)
            # Update the placement success rate of each object
            self.update_running_avg(record_dict=self.info['obj_success_rate'],
                                    key_name=self.selected_obj_name,
                                    new_value=1.)
            # Record the successful object pose
            selected_obj_pose = pu.get_body_pose(self.selected_obj_id, client_id=self.client_id)
            self.placed_obj_poses[self.selected_obj_name] = selected_obj_pose
            vel_reward += len(self.placed_obj_poses) * self.args.reward_pobj
            # Update the scene observation | transform the selected object point cloud to world frame using the current pose
            scene_obj_pose = pu.get_body_pose(self.selected_qr_scene_id, client_id=self.client_id)
            scene_obj2_selected_obj_pose = p.multiplyTransforms(*p.invertTransform(*scene_obj_pose), selected_obj_pose[0], selected_obj_pose[1])
            transformed_selected_obj_pc = self.to_numpy([p.multiplyTransforms(*scene_obj2_selected_obj_pose, point, [0., 0., 0., 1.])[0] for point in self.selected_obj_pc])
            # transformed_selected_obj_pc = tf_apply(self.to_torch(selected_obj_pose[1]), self.to_torch(selected_obj_pose[0]), self.to_torch(self.selected_obj_pc)).cpu().numpy()
            self.selected_qr_scene_pc = np.concatenate([self.selected_qr_scene_pc, transformed_selected_obj_pc], axis=0)
            self.selected_qr_scene_pc = pc_random_downsample(self.selected_qr_scene_pc, self.args.max_num_scene_points)
            self.tallest_placed_half_z_extend = max(self.tallest_placed_half_z_extend, self.obj_name_data[self.selected_obj_name]["bbox"][9])
            # Run one inference needs ~0.5s!

            # pu.visualize_pc(self.selected_qr_scene_pc)
            # pu.visualize_pc(self.selected_obj_pc)

        self.last_reward = vel_reward

        return self.last_reward
        

    def compute_done(self):
        done = False # Not real done, done signal means one training episode is done
        # When all cur stage objects have been placed, move to the next scene, refill all objects
        if len(self.unplaced_objs_name)==0 or self.cur_trial >= self.args.max_trials:
            self.unquried_scene_name.remove(self.selected_qr_scene_name)
            self.query_scene_done = True # Choose new queried scene in the current stage
            self.obj_done = True         # Choose new object in the current stage
            done = True
            if self.cur_trial >= self.args.max_trials:
                # Failed to place the object, reset its pose to the prepare area
                pu.set_pose(body=self.obj_name_data[self.selected_obj_name]["id"], pose=(self.rng.uniform(*self.prepare_area), [0., 0., 0., 1.]), client_id=self.client_id)
                # record the object placement failure
                self.update_running_avg(record_dict=self.info['obj_success_rate'],
                                        key_name=self.selected_obj_name,
                                        new_value=0.)        
        
        # Record miscs info
        if done:
            self.num_episode += 1
            # Record the scene pose here
            final_scene_obj_pose_dict = deepcopy(self.placed_obj_poses)
            scene_pos, scene_quat = self.scene_init_stable_pose
            scene_init_z_offset = self.fixed_scene_name_data[self.selected_qr_scene_name]["init_z_offset"]
            final_scene_obj_pose_dict[self.selected_qr_scene_name] = [[scene_pos[0], scene_pos[1], scene_pos[2] - scene_init_z_offset], scene_quat]

            self.info['placed_obj_poses'] = final_scene_obj_pose_dict
            self.info['success_placed_obj_num'] = self.success_obj_num
                                    
            # Record the success number if objects successfully placed in one scene
            self.update_running_avg(record_dict=self.info['scene_obj_success_num'], 
                                    key_name=self.selected_qr_scene_name, 
                                    new_value=self.success_obj_num)
            
            if self.success_obj_num >= self.args.max_num_placing_objs:
                self.info['success'] = 1.
            else:
                self.info['success'] = 0.

            # Must deepcopy the recorder since it will be reset before the info gets returned (stablebaseline3 will reset the environment after done is True)
            if self.args.blender_record:
                self.info['blender_recorder'] = deepcopy(self.pybullet_recorder) 
        
        return done
        

    ######################################
    ####### Evaluation Functions #########
    ######################################
    def visualize_actor_prob(self, raw_actions, act_log_prob):
        # action shape is (num_env, action_dim) / action is action logits we need to convert it to (0, 1)
        action = raw_actions.sigmoid() # Map action to (0, 1)
        action[..., 3:5] = 0. # No x,y rotation
        # action = [x, y, z, roll, pitch, yaw]
        World_2_QRscene = pu.get_body_pose(self.selected_qr_scene_id, client_id=self.client_id)
        World_2_QRscene_pos, World_2_QRscene_ori = self.to_torch(World_2_QRscene[0]), self.to_torch(World_2_QRscene[1])
        QRscene_2_QRregion = self.to_torch(self.selected_qr_region)
        half_extents = QRscene_2_QRregion[7:]
        QRregion_2_ObjCenter = action[..., :3] * (2 * half_extents) - half_extents
        # In the qr_scene baseLink frame
        QRregion_2_ObjCenter_shape_head = QRregion_2_ObjCenter.shape[:-1] # tf_combine requires all the dimensions are equal; we need repeat
        QRscene_2_QRregion_xyz, QRscene_2_QRregion_quat = QRscene_2_QRregion[:3].repeat(*QRregion_2_ObjCenter_shape_head, 1), QRscene_2_QRregion[3:7].repeat(*QRregion_2_ObjCenter_shape_head, 1)
        QRsceneBase_2_ObjBase_xyz = tf_combine(QRscene_2_QRregion_quat, QRscene_2_QRregion_xyz, self.to_torch([0., 0., 0., 1.]).repeat(*QRregion_2_ObjCenter_shape_head, 1), QRregion_2_ObjCenter)[1]
        # In the simulator world frame
        QRsceneBase_2_ObjBase_xyz_shape_head = QRsceneBase_2_ObjBase_xyz.shape[:-1]
        World_2_QRscene_pos, World_2_QRscene_ori = World_2_QRscene_pos.repeat(*QRsceneBase_2_ObjBase_xyz_shape_head, 1), World_2_QRscene_ori.repeat(*QRsceneBase_2_ObjBase_xyz_shape_head, 1)
        World_2_ObjBase_xyz = tf_combine(World_2_QRscene_ori, World_2_QRscene_pos, self.to_torch([0., 0., 0., 1.]).repeat(*QRsceneBase_2_ObjBase_xyz_shape_head, 1), QRsceneBase_2_ObjBase_xyz)[1]
        # Convert Rotation to Quaternion
        World_2_ObjBase_euler = action[..., 3:] * 2*np.pi
        World_2_ObjBase_quat = quat_from_euler(World_2_ObjBase_euler)

        xyz_act_prob = act_log_prob[..., :3].sum(-1).exp()
        r_act_prob = act_log_prob[..., 5].exp()
        using_act_prob = xyz_act_prob # or xyzr_act_prob
        # Compute each voxel size
        voxel_half_x = (half_extents[0] / (action.shape[0]-1)).item() # We have num_steps -1 intervals
        voxel_half_y = (half_extents[1] / (action.shape[1]-1)).item()
        voxel_half_z = (half_extents[2] / (action.shape[2]-1)).item()

        # We did not fill the x,y rotation but we need to make sure the last dimension is 6 before, now we removed it.
        World_2_ObjBase_xyz_i = World_2_ObjBase_xyz[:, :, :, 0, 0, 0].view(-1, 3)
        World_2_ObjBase_quat_i = World_2_ObjBase_quat[0, 0, 0, 0, 0, :].view(-1, 4)

        step_euler_i = World_2_ObjBase_euler[:, :, :, 0, 0, 0].view(-1, 3)
        using_act_prob_i = using_act_prob[:, :, :, 0, 0, 0].view(-1, 1)
        # print(f"Euler Angle: {step_euler_i.unique(dim=0)}")
        # Normalize the prob sum to 1
        using_act_prob_i = using_act_prob_i / (using_act_prob_i.sum() + 1e-10)
        # Strenthen the range to [0, 1.] to strengthen the color
        using_act_prob_i = (using_act_prob_i - using_act_prob_i.min()) / (using_act_prob_i.max() - using_act_prob_i.min() + 1e-10)
        r_act_prob = (r_act_prob - r_act_prob.min()) / (r_act_prob.max() - r_act_prob.min() + 1e-10)
            
        if self.args.rendering:
            if hasattr(self, "act_vs_box_ids"): 
                [p.removeBody(act_vs_box_id, physicsClientId=self.client_id) for act_vs_box_id in self.act_vs_box_ids]
            
            self.act_vs_box_ids = []
            for j in range(World_2_ObjBase_xyz_i.shape[0]):
                # Use Yellow color
                if torch.isclose(using_act_prob_i[j], torch.zeros_like(using_act_prob_i[j])): continue
                rgba_color = [1, 1, 0, using_act_prob_i[j].item()]
                act_vs_box_id = pu.draw_box_body(World_2_ObjBase_xyz_i[j].cpu().numpy(), 
                                                halfExtents=[voxel_half_x, voxel_half_y, voxel_half_z], 
                                                client_id=self.client_id, rgba_color=rgba_color)
                self.act_vs_box_ids.append(act_vs_box_id)
            time.sleep(3.)
        
        return World_2_ObjBase_xyz_i, using_act_prob_i, World_2_ObjBase_quat_i, r_act_prob
    
    ######################################
    ######### Utils FUnctions ############
    ######################################

    def blender_record(self):
        if self.args.blender_record:
            self.pybullet_recorder.add_keyframe()
    

    def blender_register(self, obj_id, body_name=None):
        if self.args.blender_record: self.pybullet_recorder.register_object(obj_id, body_name=body_name)

    
    def blender_save(self, save_path=None):
        save_path = save_path if save_path is not None else self.args.blender_record_path
        if self.args.blender_record: self.pybullet_recorder.save(save_path)

    
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
    

    def set_obj_joints_to_lower_limit(self, obj_id):
        joints_num = pu.get_num_joints(obj_id, client_id=self.client_id)
        if joints_num > 0:
            joints_limits = np.array([pu.get_joint_limits(obj_id, joint_i, client_id=self.client_id) for joint_i in range(joints_num)])
            pu.set_joint_positions(obj_id, list(range(joints_num)), joints_limits[:, 0], client_id=self.client_id)
            pu.control_joints(obj_id, list(range(joints_num)), joints_limits[:, 0], client_id=self.client_id)
            return joints_limits
    

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
    

    def update_running_avg(self, record_dict, key_name, new_value):
        if key_name not in record_dict.keys(): 
            record_dict[key_name] = [0., 0]
        avg_value, num = record_dict[key_name]
        record_dict[key_name] = [(avg_value * num + new_value) / (num + 1), num + 1]
        return record_dict


    def append_item(self, lst, item, max_len=100):
        lst.append(item)
        if len(lst) > max_len: lst.pop(0)
        return lst
    

    def close(self):
        p.disconnect(physicsClientId=self.client_id)
    

if __name__=="__main__":
    from argparse import ArgumentParser

    args = ArgumentParser()
    args.rendering = True
    args.debug = False
    args.asset_root = "assets"
    args.object_pool_folder = "union_objects_train"
    args.scene_pool_folder = "union_scene"
    args.num_pool_objs = 32
    args.num_pool_scenes = 1
    args.random_select_objs_pool = False
    args.random_select_scene_pool = False
    args.random_select_placing = False
    args.fixed_scene_only = True
    args.num_episode_to_replace_pool = 1000
    args.max_num_placing_objs = 5
    args.max_num_qr_scenes = 1
    args.sequence_len = 1
    args.default_scaling = 0.5
    args.realtime = True
    args.force_threshold = 20.
    # Original threshold:
    # args.vel_threshold = [0.005, np.pi/1800] # This two threshold need to be tuned
    # args.acc_threshold = [0.1, np.pi/180]
    args.vel_threshold = [0.005, np.pi/180] # This two threshold need to be tuned
    args.acc_threshold = [0.1, np.pi/18]
    args.eval_result = True

    args.specific_scene = "microwave_0"

    env = RoboSensaiBullet(args)

    all_pc = np.concatenate([env.obj_name_data[name]["pc"] for name in env.obj_name_data] + [env.fixed_scene_name_data[name]["pc"] for name in env.fixed_scene_name_data], axis=0)
    pu.visualize_pc(all_pc)

    while True:
        random_action = (torch.rand((1, 6), device=env.device) * 2 - 1) * 5
        _, _, done, _ = env.step(random_action)
        # print(done)
        # input("Press Enter to continue...")

    # env.step_manual()
