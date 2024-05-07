from utils import read_json, get_on_bbox, get_in_bbox, pc_random_downsample, natural_keys, se3_transform_pc
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
import pybullet_utils_cust as pu
try:
    from pynput import keyboard
except ImportError: 
    print("*** Warning: pynput can not be used on the server ***")

# PointNet
from PointNet_Model.pointnet2_cls_ssg import get_model

# Visualization
from tabulate import tabulate
from torch_utils import tf_combine, quat_from_euler, tf_apply
from Blender_script.PybulletRecorder import PyBulletRecorder


class RoboSensaiBullet:
    def __init__(self, args=None) -> None:
        self.args = deepcopy(args)
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
        obj_categories = sorted(os.listdir(self.obj_dataset_folder), key=natural_keys)
        for cate in obj_categories:
            obj_folder = os.path.join(self.obj_dataset_folder, cate)
            obj_indexes = sorted(os.listdir(obj_folder), key=natural_keys)
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
        scene_categories = sorted(os.listdir(self.fixed_scene_dataset_folder), key=natural_keys)
        for scene in scene_categories:
            scene_pool_folder = os.path.join(self.fixed_scene_dataset_folder, scene)
            scene_indexes = sorted(os.listdir(scene_pool_folder), key=natural_keys)
            for idx in scene_indexes:
                scene_uni_name = f"{scene}_{idx}"
                scene_urdf_path = f"{self.fixed_scene_dataset_folder}/{scene}/{idx}/mobility.urdf"
                scene_label_path = f"{self.fixed_scene_dataset_folder}/{scene}/{idx}/label.json"
                assert os.path.exists(scene_urdf_path), f"Scene {scene_uni_name} does not exist! Given path: {scene_urdf_path}"
                assert os.path.exists(scene_label_path), f"Scene {scene_uni_name} does not exist! Given path: {scene_label_path}"
                self.scene_uni_names_dataset.update({scene_uni_name: {"urdf": scene_urdf_path, "label": read_json(scene_label_path)}})


    def load_scenes(self):
        if not hasattr(self, "pc_extractor"): self._init_pc_extractor()

        self.fixed_scene_name_data = {}; self.unquried_scene_name = []
        self.prepare_area = [[-1100., -1100., 0.], [-1000, -1000, 100]]
        # Plane
        planeHalfExtents = [1., 1., 0.]
        planeId = self.loadURDF("plane.urdf", 
                                basePosition=[0, 0, 0.], 
                                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), 
                                useFixedBase=True)
        pu.change_obj_color(planeId, rgba_color=[1., 1., 1., 0.2], client_id=self.client_id)
        qr_pose, qr_ori, qr_half_extents = [0., 0., 0.], p.getQuaternionFromEuler([0., 0., 0.]), [1., 1., 1.]
        plane_pc_sample_region = self.to_numpy(planeHalfExtents)
        plane_pc = self.rng.uniform(-plane_pc_sample_region, plane_pc_sample_region, size=(self.args.max_num_qr_scene_points, 3))
        plane_bbox = [0., 0., 0., *p.getQuaternionFromEuler([0, 0, 0]), *planeHalfExtents]
        # self.fixed_scene_name_data["plane"] = {"id": planeId,
        #                                         "init_pose": [0., 0., 0.],
        #                                         "bbox": plane_bbox,
        #                                         "queried_region": "on", 
        #                                         "pc": plane_pc,
        
        # Table
        self.tableHalfExtents = self.args.tablehalfExtents 
        if self.args.new_tablehalfExtents is not None:
            self.tableHalfExtents = self.args.new_tablehalfExtents
        tableId = pu.create_box_body(position=[0., 0., self.tableHalfExtents[2]], orientation=p.getQuaternionFromEuler([0., 0., 0.]),
                                          halfExtents=self.tableHalfExtents, rgba_color=[1, 1, 1, 1], mass=0., client_id=self.client_id)
        
        # Default region using table position and table half extents
        qr_pose, qr_ori, qr_half_extents = [0., 0., self.tableHalfExtents[2]+0.1], p.getQuaternionFromEuler([0., 0., 0.]), [*self.tableHalfExtents[:2], 0.1]
        table_pc = self.get_obj_pc_from_id(tableId, num_points=self.args.max_num_qr_scene_points, use_worldpos=False)
        table_axes_bbox = pu.get_obj_axes_aligned_bbox_from_pc(table_pc)
        table_pc = self.convert_BaseLinkPC_2_BboxCenterPC(table_pc, table_axes_bbox)
        self.fixed_scene_name_data["table"] = {"id": tableId,
                                               "init_z_offset": 0.0,
                                               "init_pose": ([0., 0., self.tableHalfExtents[2]], p.getQuaternionFromEuler([0., 0., 0.])),
                                               "bbox": table_axes_bbox,
                                               "queried_region": "on", 
                                               "pc": table_pc,
                                               "pc_feature": self.pc_extractor(self.to_torch(table_pc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)).squeeze(0).detach().to(self.tensor_dtype)
                                              }
        # All other fixed scenes, using for loop to load later; All scene bbox are in the geometry center frame not world frame! Baselink are at the origin of world frame!
        if self.args.specific_scene is not None:
            if self.args.specific_scene == "table":
                selected_scene_pool = [] # Only load table
            else:
                assert self.args.specific_scene in self.scene_uni_names_dataset, f"Scene {self.args.specific_scene} does not exist!"
                selected_scene_pool = [self.args.specific_scene]
        elif self.args.random_select_scene_pool: # Randomly choose scene categories from the pool and their index
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
                scene_pc = self.get_obj_pc_from_id(scene_id, num_points=self.args.max_num_qr_scene_points, use_worldpos=False)
                scene_axes_bbox = pu.get_obj_axes_aligned_bbox_from_pc(scene_pc)
                scene_pc = self.convert_BaseLinkPC_2_BboxCenterPC(scene_pc, scene_axes_bbox)
                stable_init_pos_z = scene_axes_bbox[9] - scene_axes_bbox[2] # Z Half extent - Z offset between pc center and baselink
                init_z_offset = 5. if not self.args.eval_result else 0. # For training, we hang the scene in the air; For evaluation, we place the scene on the ground
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
                                                                "pc_feature": self.pc_extractor(self.to_torch(scene_pc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)).squeeze(0).detach().to(self.tensor_dtype)
                                                            }
                
                # For scene who has "in" relation, set transparence to 0.5
                if scene_label["queried_region"] == "in":
                    pu.change_obj_color(scene_id, rgba_color=[0, 0, 0, 0.2], client_id=self.client_id)
                
                # If the scene has joints, we need to set the joints to the lower limit
                self.fixed_scene_name_data[scene_uni_name]["joint_limits"] = self.set_obj_joints_to_lower_limit(scene_id)
                
                # Only for evaluation
                if self.args.eval_result:
                    self.fixed_scene_name_data[scene_uni_name]["joint_limits"] = self.set_obj_joints_to_higher_limit(scene_id)
            
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
                obj_pc = self.convert_BaseLinkPC_2_BboxCenterPC(obj_pc, obj_axes_bbox)
                obj_init_pos_z = obj_axes_bbox[9] - obj_axes_bbox[2] # Z Half extent - Z offset between pc center and baselink
                self.obj_name_data[obj_uni_name] = {
                                                    "id": object_id,
                                                    "init_pose": ([0., 0., obj_init_pos_z+5.], p.getQuaternionFromEuler([0., 0., 0.])),
                                                    "init_stable_pose": ([0., 0., obj_init_pos_z], p.getQuaternionFromEuler([0., 0., 0.])),
                                                    "bbox": obj_axes_bbox,
                                                    "queried_region": obj_label["queried_region"], 
                                                    "pc": obj_pc,
                                                    "mass": pu.get_mass(object_id, client_id=self.client_id),
                                                    "pc_feature": self.pc_extractor(self.to_torch(obj_pc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)).squeeze(0).detach().to(self.tensor_dtype)
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
        assert num_queriable_scenes >= self.args.num_pool_scenes, f"Only {num_queriable_scenes} scenes are loaded, but we need {self.args.num_pool_scenes} scenes!"
        assert len(self.obj_name_data) >= self.args.max_num_placing_objs, f"Only {len(self.obj_name_data)} objects are loaded, but we need {self.args.max_num_placing_objs} objects!"
        
        verbose_info = f"Loaded {len(self.obj_name_data)} objects from {self.args.object_pool_folder} | {self.args.max_num_placing_objs} objects will be placed."
        print("*"*len(verbose_info)); print(verbose_info); print("*"*len(verbose_info))
        if hasattr(self, "pc_extractor"): self._del_pc_extractor()

    
    def load_world(self):
        self._init_misc_variables()
        self._init_obs_act_space()
        self.load_scenes()
        self.load_objects()
        if self.args.blender_record: 
            self.pybullet_recorder = PyBulletRecorder(client_id=self.client_id)
        
        # Only for evaluation
        if self.args.eval_result:
            self.stable_placement_task_init()
            if self.args.use_robot_sp:
                self.stable_placement_task_robot_init()


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

    def _init_pc_extractor(self):
        self.pc_extractor = get_model(num_class=40, normal_channel=False).to(self.device) # num_classes is used for loading checkpoint to make sure the model is the same
        self.pc_extractor.load_checkpoint(ckpt_path="PointNet_Model/checkpoints/best_model.pth", evaluate=True, map_location=self.device)


    def _del_pc_extractor(self):
        del self.pc_extractor      
    

    def _init_obs_act_space(self):
        # Observation space: [scene_pc_feature, obj_pc_feature, bbox, action, history (hitory_len * (obj_pos + obj_vel)))]
        # 1 is the env number to align with the isaacgym env
        # We have two kinds of observation: seq_obs [qr_region, prev_action, obj_sim_history]; pc_obs [scene_pc, obj_pc]
        self.qr_region_dim = 10; self.action_dim = 4; self.traj_hist_dim = self.args.max_traj_history_len*(6+7)
        self.traj_history_shape = (self.args.max_traj_history_len, 6+7) # obj_pos dimension + obj_vel dimension
        self.raw_act_hist_qr_obs_shape = (1, self.args.sequence_len, self.qr_region_dim + self.action_dim + self.traj_hist_dim)

        self.history_ft_dim = self.traj_hist_dim if not self.args.use_traj_encoder else 512; 
        self.qr_region_ft_dim = self.qr_region_dim; self.action_ft_dim = self.action_dim
        self.post_act_hist_qr_ft_shape = (1, self.args.sequence_len, self.qr_region_ft_dim + self.action_ft_dim + self.history_ft_dim)
        
        self.scene_ft_dim = 1024; self.obj_ft_dim = 1024
        self.seq_info_ft_dim = 2048 if self.args.use_seq_obs_encoder else np.prod(self.post_act_hist_qr_ft_shape[1:])
        self.post_observation_shape = (1, self.seq_info_ft_dim + self.scene_ft_dim + self.obj_ft_dim)
        
        # Action space: [x, y, z, yaw]
        self.action_shape = (1, self.action_dim)

        # slice
        self.qr_region_slice = slice(0, self.qr_region_dim)
        self.action_slice = slice(self.qr_region_slice.stop, self.qr_region_slice.stop+self.action_dim)
        self.traj_history_slice = slice(self.action_slice.stop, self.action_slice.stop+self.traj_hist_dim)

    
    def _init_misc_variables(self):
        self.args.tablehalfExtents = self.args.tablehalfExtents if hasattr(self.args, "tablehalfExtents") else [0.2, 0.3, 0.35]
        self.args.tableQueryRegion = self.args.tableQueryRegion if hasattr(self.args, "tableQueryRegion") else None
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
        self.args.max_num_qr_scene_points = self.args.max_num_qr_scene_points if hasattr(self.args, "max_num_qr_scene_points") else 10 * self.args.max_num_urdf_points
        self.args.max_num_scene_points = self.args.max_num_scene_points if hasattr(self.args, "max_num_scene_points") else 10240
        self.args.fixed_qr_region = self.args.fixed_qr_region if hasattr(self.args, "fixed_qr_region") else False
        self.args.use_traj_encoder = self.args.use_traj_encoder if hasattr(self.args, "use_traj_encoder") else False
        self.args.blender_record = self.args.blender_record if hasattr(self.args, "blender_record") else False
        self.args.step_async = self.args.step_async if hasattr(self.args, "step_async") else False
        self.args.step_sync = self.args.step_sync if hasattr(self.args, "step_sync") else False
        self.args.eval_result = self.args.eval_result if hasattr(self.args, "eval_result") else False
        # Buffer does not need to be reset
        self.create_info_buffer()
        self.num_episode = 0
        self.default_qr_region_z = 0.3
        # Evluation Args
        self.args.seq_select_placing = self.args.seq_select_placing if hasattr(self.args, "seq_select_placing") else False
        self.args.new_tablehalfExtents = self.args.new_tablehalfExtents if hasattr(self.args, "new_tablehalfExtents") else None
        self.args.use_robot_sp = self.args.use_robot_sp if hasattr(self.args, "use_robot_sp") else False
        # Verify the args are correct
        assert self.args.max_stable_steps + self.args.min_continue_stable_steps <= self.args.max_traj_history_len, \
            "The max_stable_steps and min_continue_stable_steps should be smaller than max_traj_history_len! max_traj_history_len is the steps of simulation!"
        if self.args.tableQueryRegion is not None:
            assert all([self.args.tableQueryRegion[i] <= self.args.tablehalfExtents[i] for i in range(len(self.args.tableQueryRegion[:2]))]), "Table query region should be smaller than table half extents!"

    
    def create_info_buffer(self):
        self.info = {
            "success": 0., 
            "stepping": 1., 
            "success_placed_obj_num": 0, 
            "obj_success_rate": {}, 
            "scene_obj_success_num": {}, 
            "pc_change_indicator": 1.,
            "placement_trajs": {},
            "placement_trajs_temp": {},
        }

    
    def reset_env(self):
        # Place all objects to the prepare area
        for obj_name in self.obj_name_data.keys():
            pu.set_pose(body=self.obj_name_data[obj_name]["id"], 
                pose=(self.rng.uniform(*self.prepare_area), [0., 0., 0., 1.]), client_id=self.client_id)
            pu.set_mass(self.obj_name_data[obj_name]["id"], mass=0., client_id=self.client_id)
        
        # Place all scenes to the prepare area
        for scene_name in self.fixed_scene_name_data.keys():
            pu.set_pose(body=self.fixed_scene_name_data[scene_name]["id"], 
                pose=(self.rng.uniform(*self.prepare_area), [0., 0., 0., 1.]), client_id=self.client_id)
            pu.set_mass(self.obj_name_data[obj_name]["id"], mass=0., client_id=self.client_id)
        
        # Scene
        self.placed_obj_poses = {}
        self.success_obj_num = 0
        if len(self.unquried_scene_name) == 0: 
            self.update_unquery_scenes()
        self.query_scene_done = True
        self.obj_done = True

        # Training
        self.cur_trial = 0
        # Observations
        self.traj_history = [[0.]* (7 + 6)] * self.args.max_traj_history_len  # obj_pos dimension + obj_vel dimension
        self.last_raw_action = np.zeros(self.action_dim, dtype=self.numpy_dtype)
        self.last_seq_obs = np.zeros(self.raw_act_hist_qr_obs_shape[1:], dtype=self.numpy_dtype)
        # Rewards
        self.accm_vel_reward = 0.
        # Information reset (some information does not need to be reset!)
        self.info["placement_trajs"] = deepcopy(self.info["placement_trajs_temp"])
        self.info["placement_trajs_temp"] = {}

        # Blender record
        if self.args.blender_record: 
            self.pybullet_recorder.reset(links=True)
            [self.pybullet_recorder.add_keyframe() for _ in range(120)] # Add 0.5s keyframes to show the start

    
    def update_unplaced_objs(self):
        # You can use unplaced_objs to decide how many objs should be placed on the scene
        selected_obj_pool = list(self.obj_name_data.keys())
        self.unplaced_objs_name = []
        if self.args.random_select_placing:
            self.rng.shuffle(selected_obj_pool)
        
        # Sequence selection to check the dataset stabability (only for evaluation)
        if self.args.seq_select_placing:
            if not hasattr(self, "cur_seq_index"): self.cur_seq_index = 0
            candidate_obj_name = selected_obj_pool[self.cur_seq_index:self.cur_seq_index+self.args.max_num_placing_objs]
            self.unplaced_objs_name.extend(candidate_obj_name)
            self.cur_seq_index = (self.cur_seq_index + self.args.max_num_placing_objs) % len(selected_obj_pool)
            print(f"Current Sequence Index: {self.cur_seq_index}; NUmber of pool objs: {len(selected_obj_pool)}")
        else:
            while True:
                candidate_obj_name = selected_obj_pool.pop(0)
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
        if self.args.specific_scene is not None:
            self.unquried_scene_name = [self.args.specific_scene]
        else:
            self.unquried_scene_name = list(self.fixed_scene_name_data.keys())
            if not self.args.fixed_scene_only:
                self.unquried_scene_name.extend(self.queriable_obj_names)
        
        
    def convert_actions(self, action):
        # action shape is (num_env, action_dim) / action is action logits we need to convert it to (0, 1)
        action = action.squeeze(dim=0).cpu().numpy()
        self.last_raw_action = action.copy()
        
        # action = [x, y, z, roll, pitch, yaw]
        QRsceneCenter_2_QRregionCenter = self.selected_qr_region
        QRregion_half_extents = QRsceneCenter_2_QRregionCenter[7:]
        QRregionCenter_2_ObjBboxCenter = action[:3] * (2 * QRregion_half_extents) - QRregion_half_extents
        QRregionCenter_2_ObjBase_xyz = p.multiplyTransforms(QRregionCenter_2_ObjBboxCenter, [0, 0, 0, 1.], -self.selected_obj_bbox[:3], [0, 0, 0, 1.])[0]
        # In the qr_scene center frame
        QRsceneCenter_2_ObjBase_xyz = p.multiplyTransforms(QRsceneCenter_2_QRregionCenter[:3], QRsceneCenter_2_QRregionCenter[3:7], QRregionCenter_2_ObjBase_xyz, [0., 0., 0., 1.])[0]
        QRsceneCenter_2_ObjBase_quat = p.getQuaternionFromEuler((np.array([0., 0., action[3]], dtype=self.numpy_dtype)* 2*np.pi))
        # In the simulator world frame
        World_2_QRsceneBase = pu.get_body_pose(self.selected_qr_scene_id, client_id=self.client_id)
        QRsceneBase_2_QRsceneCenter = self.selected_qr_scene_bbox
        World_2_QRsceneCenter = p.multiplyTransforms(World_2_QRsceneBase[0], World_2_QRsceneBase[1], QRsceneBase_2_QRsceneCenter[:3], QRsceneBase_2_QRsceneCenter[3:7])
        World_2_ObjBase_xyz, World_2_ObjBase_quat \
              = p.multiplyTransforms(World_2_QRsceneCenter[0], World_2_QRsceneCenter[1], QRsceneCenter_2_ObjBase_xyz, QRsceneCenter_2_ObjBase_quat)

        if self.args.rendering:
            World_2_QRregionCenter = p.multiplyTransforms(*World_2_QRsceneCenter, QRsceneCenter_2_QRregionCenter[:3], QRsceneCenter_2_QRregionCenter[3:7])
            if hasattr(self, "last_World_2_QRregionCenter") and World_2_QRregionCenter == self.last_World_2_QRregionCenter: pass
            else:
                if hasattr(self, "region_vis_id"): p.removeBody(self.region_vis_id, physicsClientId=self.client_id)
                self.region_vis_id = pu.draw_box_body(position=World_2_QRregionCenter[0], orientation=World_2_QRregionCenter[1],
                                                      halfExtents=QRregion_half_extents, rgba_color=[1, 0, 0, 0.1], client_id=self.client_id)
                self.last_World_2_QRregionCenter = World_2_QRregionCenter

        return World_2_ObjBase_xyz, World_2_ObjBase_quat


    def step(self, action):
        if self.args.step_async:
            return self.step_async(action)
        elif self.args.step_sync:
            return self.step_sync(action)


    def step_sync(self, action):
        pose_xyz, pose_quat = self.convert_actions(action)
        pu.set_pose(self.selected_obj_id, (pose_xyz, pose_quat), client_id=self.client_id)
        
        self.traj_history = [[0.]* (7 + 6)] * self.args.max_traj_history_len  # obj_pos dimension + obj_vel dimension
        self.accm_vel_reward = 0.
        self.prev_obj_vel = np.array([0.]*6, dtype=self.numpy_dtype)
        self.continue_stable_steps = 0.
        self.info['pc_change_indicator'] = 0.

        for self.his_steps in range(self.args.max_traj_history_len):
            self.simstep(1/240)
            self.blender_record()
            
            (obj_pos, obj_quat), obj_vel = self.get_QRregionCenter2ObjCenter_pose_vel()
            # Update the trajectory history [0, 0, 0, ..., T0, T1..., Tn]; Left Shift
            self.traj_history.pop(0)
            self.traj_history.append(obj_pos + obj_quat + obj_vel.tolist())
            # Accumulate velocity reward
            self.accm_vel_reward += -obj_vel[:].__abs__().sum()
            # Jump out if the object is not moving and keep stable for a continuous certain number of steps
            if self.vel_checker(obj_vel) and self.acc_checker(self.prev_obj_vel, obj_vel):
                self.continue_stable_steps += 1
                if self.continue_stable_steps >= self.args.min_continue_stable_steps: 
                    break
            # else:
            #     self.continue_stable_steps = 0

            self.prev_obj_vel = obj_vel

        self.record_placement_traj((pose_xyz, pose_quat), self.his_steps)
        reward = self.compute_reward() # Must compute reward before observation since we use the velocity to compute reward
        done = self.compute_done()
        observation = self.compute_observations() if not done else self.last_seq_obs # This point should be considered as the start of the episode!
        self.re_place_placed_obj_poses()

        # 1 Env we do not use stableBaseline which will ask for manually reset
        if self.args.num_envs == 1:
            observation = self.reset() if done else observation 
            return np.expand_dims(observation, axis=0), np.array([reward]), \
                   np.array([done]), [self.info]
        else:
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

        if self.args.num_envs == 1:
            return np.expand_dims(self.compute_observations(), axis=0)
        else:
            return self.compute_observations()

    
    def compute_observations(self):
        # Choose query scene 
        if self.query_scene_done:
            self.query_scene_done = False
            if hasattr(self, "selected_qr_scene_id"): # Move the previous scene to the prepare area
                pu.set_pose(body=self.selected_qr_scene_id, pose=(self.rng.uniform(*self.prepare_area), [0., 0., 0., 1.]), client_id=self.client_id) # Reset the pose of the scene
            while True:
                self.selected_qr_scene_name = self.unquried_scene_name[0]
                self.scene_name_data = self.obj_name_data if self.selected_qr_scene_name in self.obj_name_data.keys() else self.fixed_scene_name_data
                self.selected_qr_scene_id = self.scene_name_data[self.selected_qr_scene_name]["id"]
                self.selected_qr_scene_pc = self.scene_name_data[self.selected_qr_scene_name]["pc"]
                self.selected_qr_scene_ft = self.scene_name_data[self.selected_qr_scene_name]["pc_feature"]
                self.selected_qr_scene_bbox = self.scene_name_data[self.selected_qr_scene_name]["bbox"]
                self.selected_qr_scene_region = self.scene_name_data[self.selected_qr_scene_name]["queried_region"]
                self.tallest_placed_half_z_extend = 0.
                self.scene_init_stable_pose = self.scene_name_data[self.selected_qr_scene_name]["init_pose"]
                pu.set_pose(body=self.selected_qr_scene_id, pose=self.scene_init_stable_pose, client_id=self.client_id) # Reset the pose of the scene
                pu.set_mass(self.selected_qr_scene_id, mass=0., client_id=self.client_id) # Set the scene mass to 0

                self.update_unplaced_objs() # Refill all objects based on the new selected scene when the queried scene changed
                if len(self.unplaced_objs_name) > 0: 
                    self.info["selected_init_qr_scene_ft"] = self.selected_qr_scene_ft
                    break
                else: # This scene has no objects to place based on the naive filtering, remove it from the unquery scene list
                    self.unquried_scene_name.remove(self.selected_qr_scene_name)
                    if len(self.unquried_scene_name) == 0: self.update_unquery_scenes()
            
            self.blender_register(self.selected_qr_scene_id, self.selected_qr_scene_name)
            
        # We need object description (index), bbox, reward (later)
        # Choose query object placement order
        if self.obj_done:
            self.obj_done = False
            self.selected_obj_name = self.unplaced_objs_name[0]
            self.selected_obj_id = self.obj_name_data[self.selected_obj_name]["id"]
            self.selected_obj_pc = self.obj_name_data[self.selected_obj_name]["pc"]
            self.selected_obj_pc_ft = self.obj_name_data[self.selected_obj_name]["pc_feature"]
            self.selected_obj_bbox = self.obj_name_data[self.selected_obj_name]["bbox"]
            pu.set_mass(self.selected_obj_id, self.obj_name_data[self.selected_obj_name]["mass"], client_id=self.client_id)
            self.info['pc_change_indicator'] = 1.

            self.blender_register(self.selected_obj_id, self.selected_obj_name)

        # Compute query region area based on the selected object and scene
        if self.selected_qr_scene_region == "on":
            max_z_half_extent = self.tallest_placed_half_z_extend + self.selected_obj_bbox[9] # If on, bbox is half extent + tallest placed obj half extent (equivalent to the current scene bbox).
            selected_qr_scene_bbox = self.selected_qr_scene_bbox.copy()
            if self.args.fixed_qr_region: max_z_half_extent = self.default_qr_region_z
            if self.selected_qr_scene_name == "table" and self.args.tableQueryRegion is not None: # If the table query region is set, we use it to define the query region of the table on x, y
                selected_qr_scene_bbox[7:9] = self.args.tableQueryRegion[:2] 
            self.selected_qr_region = get_on_bbox(selected_qr_scene_bbox, z_half_extend=max_z_half_extent)
        elif self.selected_qr_scene_region == "in":
            # max z-half extend should be max(self.selected_obj_bbox[9], self.latest_scene_bbox[9]) ## The latest_scene_bbox should be recomputed after the object is placed (but we did not do this for now)
            # Our current bbox is max(self.selected_obj_bbox[9], self.selected_qr_scene_bbox[9])
            max_z_half_extent = self.selected_obj_bbox[9] 
            self.selected_qr_region = get_in_bbox(self.selected_qr_scene_bbox.copy(), z_half_extend=max_z_half_extent)
        else:
            raise NotImplementedError(f"Object {self.selected_qr_scene_name} Queried region {self.selected_qr_scene_region} is not implemented!")

        # Update the queried scene points cloud; All observations are in the query region center frame apart from the object point cloud (feature)
        # The reason I did not padding zeros here is because the multi-envs transporation will be time-consuming; We pad zeros in the training.
        QRregionCenter_2_QRsceneCenter = p.invertTransform(self.selected_qr_region[:3], self.selected_qr_region[3:7])
        QRregionCenter_2_QRscenePC = se3_transform_pc(*QRregionCenter_2_QRsceneCenter, self.selected_qr_scene_pc)
        self.info["selected_qr_scene_pc"] = QRregionCenter_2_QRscenePC
        self.info["selected_obj_pc_ft"] = self.selected_obj_pc_ft
        # Update the sequence observation | pop the first observation and append the last observation
        his_traj = self.to_numpy(self.traj_history).flatten()
        cur_seq_obs = np.concatenate([self.selected_qr_region, self.last_raw_action, his_traj])
        # Left shift the sequence observation
        self.last_seq_obs = np.concatenate([self.last_seq_obs[1:, :], np.expand_dims(cur_seq_obs, axis=0)])
        
        return self.last_seq_obs


    def compute_reward(self):
        self.cur_trial += 1
        if self.args.rendering:
            print(f"Placing {self.selected_obj_name} {self.selected_qr_scene_region} the {self.selected_qr_scene_name} | Stable Steps: {self.his_steps} | Trial: {self.cur_trial}")

        vel_reward = self.args.vel_reward_scale * self.accm_vel_reward
        if self.his_steps <= (self.args.max_stable_steps + self.args.min_continue_stable_steps):
            # This object is successfully placed!! Jump to the next object, object is stable within 10 simulation steps
            self.obj_done = True; self.cur_trial = 0; self.success_obj_num += 1
            self.unplaced_objs_name.remove(self.selected_obj_name)
            # Update the placement success rate of each object
            self.update_running_avg(record_dict=self.info['obj_success_rate'],
                                    key_name=self.selected_obj_name,
                                    new_value=1.)
            # Record the successful object base pose
            World2SelectedObjBase_pose = pu.get_body_pose(self.selected_obj_id, client_id=self.client_id)
            World2SelectedObjCenter_pose = p.multiplyTransforms(
                World2SelectedObjBase_pose[0], World2SelectedObjBase_pose[1], 
                self.selected_obj_bbox[:3], self.selected_obj_bbox[3:7]
            )
            self.placed_obj_poses[self.selected_obj_name] = World2SelectedObjBase_pose
            # Update the scene observation | transform the selected object point cloud to world frame using the current pose
            # Merge the placed object points cloud to the scene points cloud (scene center frame)
            World2QRsceneBase_pose = pu.get_body_pose(self.selected_qr_scene_id, client_id=self.client_id)
            World2QRsceneCenter_pose = p.multiplyTransforms(
                World2QRsceneBase_pose[0], World2QRsceneBase_pose[1], 
                self.selected_qr_scene_bbox[:3], self.selected_qr_scene_bbox[3:7]
            )
            QRsceneCenter_2_SelectedObjCenter = pu.multiply_multi_transforms(
                p.invertTransform(*World2QRsceneCenter_pose), 
                World2SelectedObjCenter_pose
            )
            transformed_selected_obj_pc = se3_transform_pc(*QRsceneCenter_2_SelectedObjCenter, self.selected_obj_pc)
            self.selected_qr_scene_pc = np.concatenate([self.selected_qr_scene_pc, transformed_selected_obj_pc], axis=0)
            self.selected_qr_scene_pc = pc_random_downsample(self.selected_qr_scene_pc, self.args.max_num_scene_points)
            self.info["selected_init_qr_scene_ft"] = None # The scene points need to be re-extracted
            self.tallest_placed_half_z_extend = max(self.tallest_placed_half_z_extend, self.obj_name_data[self.selected_obj_name]["bbox"][9])

            # vel_reward += len(self.placed_obj_poses) * self.args.reward_pobj
            vel_reward += max(100, self.args.reward_pobj * self.args.max_num_placing_objs) \
                          if len(self.placed_obj_poses) >= self.args.max_num_placing_objs \
                          else self.args.reward_pobj

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
            self.info['qr_scene_pose'] = [[scene_pos[0], scene_pos[1], scene_pos[2] - scene_init_z_offset], 
                                           scene_quat, 
                                           self.selected_qr_scene_bbox.tolist()]
            self.info['placed_obj_poses'] = final_scene_obj_pose_dict
            self.info['success_placed_obj_num'] = self.success_obj_num          
            # Record the success number if objects successfully placed in one scene
            self.update_running_avg(record_dict=self.info['scene_obj_success_num'], 
                                    key_name=self.selected_qr_scene_name, 
                                    new_value=self.success_obj_num)
            
            if self.success_obj_num >= self.args.max_num_placing_objs:
                # Post-Check about the stability of the placement in stable placement data collection
                if self.args.eval_result and (self.args.strict_checking or self.args.sp_data_collection): 
                    self.info['success'] = float(self.post_check_scene_stable())
                else:
                    self.info['success'] = 1.
            else:
                self.info['success'] = 0.

            if self.args.rendering and self.info['success']==1: # Success visualization; Only for evaluation!
                print(f"Successfully Place {self.success_obj_num} Objects {self.selected_qr_scene_region} the {self.selected_qr_scene_name}!")
                if self.args.eval_result: pu.step_real(duration=3., client_id=self.client_id)

            # Must deepcopy the recorder since it will be reset before the info gets returned (stablebaseline3 will reset the environment after done is True)
            if self.args.blender_record:
                self.info['blender_recorder'] = deepcopy(self.pybullet_recorder) 
            
            # Stable placement demo
            if self.args.eval_result and self.args.use_robot_sp:
                sp_workwell = self.stable_placement_task_step()
                if not sp_workwell: 
                    self.stable_placement_reset()
            
            # Stable placement data collection
            if self.args.eval_result and self.args.sp_data_collection and self.info['success']==1:
                self.stable_placement_data_collection()
        
        return done
    

    def post_check_scene_stable(self, placed_obj_poses=None, re_place_obj=True, reset_mass=False):
        # All objects should be continuously stable for self.args.min_continue_stable_steps steps
        placed_obj_poses = self.placed_obj_poses if placed_obj_poses is None else placed_obj_poses
        if re_place_obj:
            self.re_place_placed_obj_poses(placed_obj_poses=placed_obj_poses, reset_mass=reset_mass)
        continue_stable_steps = np.zeros(len(placed_obj_poses), dtype=self.numpy_dtype)
        stabability_record = np.zeros(len(placed_obj_poses), dtype=self.numpy_dtype)
        prev_obj_vel = np.zeros((len(placed_obj_poses), 6), dtype=self.numpy_dtype)
        
        for i in range(self.args.max_stable_steps+self.args.min_continue_stable_steps):
            self.simstep(1/240)
            for j, obj_name in enumerate(placed_obj_poses.keys()):
                (obj_pos, obj_quat), obj_vel = self.get_QRregionCenter2ObjCenter_pose_vel(objUniName=obj_name)

                if self.vel_checker(obj_vel) and self.acc_checker(obj_vel, prev_obj_vel[j, :]):
                    continue_stable_steps[j] += 1
                    if continue_stable_steps[j] >= self.args.min_continue_stable_steps:
                        stabability_record[j] = 1
                else:
                    continue_stable_steps[j] = 0

                prev_obj_vel[j, :] = obj_vel

            # If all the objects are continuous stable for min_continue_stable_steps, we consider the placement is stable
            if (stabability_record==1).all():
                return True
        return False
    

    # Only for stable placement evaluation
    def post_check_obj_stable(self, specific_obj_name, placed_obj_poses=None, re_place_obj=True, reset_mass=False):
        # We need to check the stability of the placement
        # The object should be stable for 50 steps
        placed_obj_poses = self.placed_obj_poses if placed_obj_poses is None else placed_obj_poses
        assert specific_obj_name in placed_obj_poses.keys(), f"Specific object {specific_obj_name} is not in the placed object list!"
        if re_place_obj:
            self.re_place_placed_obj_poses(placed_obj_poses=placed_obj_poses, reset_mass=reset_mass)
        continue_stable_steps = 0
        prev_obj_vel = np.zeros(6, dtype=self.numpy_dtype)
        
        for i in range(self.args.max_stable_steps+self.args.min_continue_stable_steps):
            self.simstep(1/240)
            (obj_pos, obj_quat), obj_vel = self.get_QRregionCenter2ObjCenter_pose_vel(objUniName=specific_obj_name)
            # Jump out if the object is not moving
            if self.vel_checker(obj_vel) and self.acc_checker(obj_vel, prev_obj_vel):
                continue_stable_steps += 1
                if continue_stable_steps >= self.args.min_continue_stable_steps: 
                    return True
            else:
                continue_stable_steps = 0

            prev_obj_vel = obj_vel
        
        # If all the objects are continuous stable for min_continue_stable_steps, we consider the placement is stable
        return False
        

    ######################################
    ####### Evaluation Functions #########
    ######################################
    def visualize_actor_prob(self, raw_actions, act_log_prob, step_action):
        World_2_ObjBasePlace_xyz, World_2_ObjBasePlace_quat = self.convert_actions(step_action)
        World_2_ObjBboxCenterPlace_xyz = p.multiplyTransforms(World_2_ObjBasePlace_xyz, [0, 0, 0, 1.], self.selected_obj_bbox[:3], [0, 0, 0, 1.])[0]

        # action shape is (num_env, action_dim) / action is action logits we need to convert it to (0, 1)
        action = raw_actions # Map action to (0, 1)
        # action = [x, y, z, yaw]

        QRsceneCenter_2_QRregionCenter = self.to_torch(self.selected_qr_region)
        QRregion_half_extents = QRsceneCenter_2_QRregionCenter[7:]
        QRregionCenter_2_ObjBboxCenter_xyz = action[..., :3] * (2 * QRregion_half_extents) - QRregion_half_extents
        # Compute the grid half extents to compute the voxel size
        temp_QRregionCenter_2_ObjBboxCenter_xyz = QRregionCenter_2_ObjBboxCenter_xyz.view(-1, 3) # view, so that max and min are easier to commpute
        Grid_half_extents = (temp_QRregionCenter_2_ObjBboxCenter_xyz.max(dim=0)[0] - temp_QRregionCenter_2_ObjBboxCenter_xyz.min(dim=0)[0]) / 2
        # In the qr_scene baseLink frame
        QRregionCenter_2_ObjBboxCenter_shape_head = QRregionCenter_2_ObjBboxCenter_xyz.shape[:-1] # tf_combine requires all the dimensions are equal; we need repeat
        QRsceneCenter_2_QRregionCenter_xyz, QRsceneCenter_2_QRregionCenter_quat = \
            QRsceneCenter_2_QRregionCenter[:3].repeat(*QRregionCenter_2_ObjBboxCenter_shape_head, 1), QRsceneCenter_2_QRregionCenter[3:7].repeat(*QRregionCenter_2_ObjBboxCenter_shape_head, 1)
        QRsceneCenter_2_ObjBboxCenter_xyz = \
            tf_combine(
                QRsceneCenter_2_QRregionCenter_quat, QRsceneCenter_2_QRregionCenter_xyz, 
                self.to_torch([0., 0., 0., 1.]).repeat(*QRregionCenter_2_ObjBboxCenter_shape_head, 1), QRregionCenter_2_ObjBboxCenter_xyz
            )[1]
        QRsceneCenter_2_ObjBboxCenter_euler = torch.zeros(action.shape[:-1]+(3, ), dtype=self.tensor_dtype, device=action.device)
        QRsceneCenter_2_ObjBboxCenter_euler[..., 2] = action[..., 3] * 2*np.pi
        QRsceneCenter_2_ObjBboxCenter_quat = quat_from_euler(QRsceneCenter_2_ObjBboxCenter_euler)
        
        # In the simulator world frame
        World_2_QRsceneBase = pu.get_body_pose(self.selected_qr_scene_id, client_id=self.client_id)
        World_2_QRsceneBase_pos, World_2_QRsceneBase_ori = self.to_torch(World_2_QRsceneBase[0]), self.to_torch(World_2_QRsceneBase[1])
        QRsceneBase_2_QRsceneCenter = self.to_torch(self.selected_qr_scene_bbox)
        QRsceneBase_2_QRsceneCenter_pos, QRsceneBase_2_QRsceneCenter_quat = QRsceneBase_2_QRsceneCenter[:3], QRsceneBase_2_QRsceneCenter[3:7]
        World_2_QRsceneCenter_quat, World_2_QRsceneCenter_pos = \
            tf_combine(
                World_2_QRsceneBase_ori, World_2_QRsceneBase_pos, 
                QRsceneBase_2_QRsceneCenter_quat, QRsceneBase_2_QRsceneCenter_pos
            )
        QRsceneCenter_2_ObjBboxCenter_xyz_shape_head = QRsceneCenter_2_ObjBboxCenter_xyz.shape[:-1]
        World_2_QRsceneCenter_pos, World_2_QRsceneCenter_quat = \
            World_2_QRsceneCenter_pos.repeat(*QRsceneCenter_2_ObjBboxCenter_xyz_shape_head, 1), World_2_QRsceneCenter_quat.repeat(*QRsceneCenter_2_ObjBboxCenter_xyz_shape_head, 1)
        
        World_2_ObjBboxCenter_quat, World_2_ObjBboxCenter_xyz = tf_combine(
            World_2_QRsceneCenter_quat, World_2_QRsceneCenter_pos, 
            QRsceneCenter_2_ObjBboxCenter_quat, QRsceneCenter_2_ObjBboxCenter_xyz
        )

        # act_log_prob shape is [dim_x, dim_y, dim_z, dim_roll, dim_pitch, dim_yaw, 6]
        # 6 represents one prob of certain combination of x, y, z, r, p, y
        xyz_act_prob = act_log_prob[..., :3].sum(-1).exp()
        r_act_prob = act_log_prob[..., 3].exp()
        using_act_prob = xyz_act_prob # or xyzr_act_prob
        # Compute each voxel size
        voxel_half_x = (Grid_half_extents[0] / (action.shape[0]-1)).item() # We have num_steps -1 intervals
        voxel_half_y = (Grid_half_extents[1] / (action.shape[1]-1)).item()
        voxel_half_z = (Grid_half_extents[2] / (action.shape[2]-1)).item()

        # We did not fill the x,y rotation but we need to make sure the last dimension is 6 before, now we removed it.
        World_2_ObjBboxCenter_xyz_i = World_2_ObjBboxCenter_xyz[:, :, :, 0].view(-1, 3).cpu().numpy()
        World_2_ObjBboxCenter_quat_i = World_2_ObjBboxCenter_quat[0, 0, 0, :].view(-1, 4).cpu().numpy()

        step_euler_i = QRsceneCenter_2_ObjBboxCenter_euler[:, :, :, 0].view(-1, 3)
        using_act_prob_i = using_act_prob[:, :, :, 0].view(-1, 1)
        # print(f"Euler Angle: {step_euler_i.unique(dim=0)}")
        # Strenthen the range to [0, 1.] to strengthen the color
        using_act_prob_i = (using_act_prob_i - using_act_prob_i.min()) / (using_act_prob_i.max() - using_act_prob_i.min() + 1e-10)
        r_act_prob = (r_act_prob - r_act_prob.min()) / (r_act_prob.max() - r_act_prob.min() + 1e-10)
            
        if self.args.rendering:
            if hasattr(self, "act_vs_box_ids"): 
                [p.removeBody(act_vs_box_id, physicsClientId=self.client_id) for act_vs_box_id in self.act_vs_box_ids]
            
            self.act_vs_box_ids = []
            if using_act_prob_i.max() == 0: # The resolution is too low and the agent has a high confidence about one certain pos
                act_vs_box_id = pu.draw_box_body(World_2_ObjBboxCenterPlace_xyz, 
                                                 halfExtents=[voxel_half_x, voxel_half_y, voxel_half_z], 
                                                 client_id=self.client_id, rgba_color=[1, 1, 0, 0.5])
                self.act_vs_box_ids.append(act_vs_box_id)
            else:
                for j in range(World_2_ObjBboxCenter_xyz_i.shape[0]):
                    # Use Yellow color
                    if torch.isclose(using_act_prob_i[j], torch.zeros_like(using_act_prob_i[j])): 
                        continue
                    rgba_color = [1, 1, 0, using_act_prob_i[j].item()]
                    act_vs_box_id = pu.draw_box_body(World_2_ObjBboxCenter_xyz_i[j], 
                                                     halfExtents=[voxel_half_x, voxel_half_y, voxel_half_z], 
                                                     client_id=self.client_id, rgba_color=rgba_color)
                    self.act_vs_box_ids.append(act_vs_box_id)
        
        actor_step_data = {
            "World_2_ObjBboxCenter_xyz_i": World_2_ObjBboxCenter_xyz_i,
            "World_2_ObjBboxCenter_quat_i": World_2_ObjBboxCenter_quat_i,
            "visual_voxel_half_extents": [voxel_half_x, voxel_half_y, voxel_half_z],
            "using_act_prob_i": using_act_prob_i.cpu().numpy(),
            "r_act_prob": r_act_prob.cpu().numpy(),
            "World_2_ObjBboxCenterPlace_xyz": World_2_ObjBboxCenterPlace_xyz,
            "World_2_ObjBasePlace_quat": World_2_ObjBasePlace_quat
        }

        return actor_step_data
    

    ######################################
    ####### Stable Placement Task ########
    ######################################
    def init_camera(self):
        self.scene_cameras = {}
        self.obj_cameras = {}
        self.width = 224  # Width of the depth image
        self.height = 224  # Height of the depth image
        fov = 60  # Field of view in degrees
        aspect = self.width / self.height  # Aspect ratio
        near = 0.02  # Near clipping plane
        far = 5.0  # Far clipping plane
        projection_matrix = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=near, farVal=far)

        tableheight = self.tableHalfExtents[2] * 2
        scene_camera_height = 0.4 + tableheight
        obj_camera_height = 0.2 + self.sp_obj_prepared_World2Center_pose[0][2]

        num_camera_poses = 4
        scene_camera_focus_pose = [0, 0, tableheight]
        scene_camera_view_traj_x = [[-self.tableHalfExtents[0]*2, 0., scene_camera_height], [self.tableHalfExtents[0]*2, 0., scene_camera_height]]
        scene_camera_view_traj_y = [[0, -self.tableHalfExtents[1]*2, scene_camera_height], [0, self.tableHalfExtents[1]*2, scene_camera_height]]
        for i, scene_camera_view_pose in enumerate(np.linspace(scene_camera_view_traj_x[0], scene_camera_view_traj_x[1], num_camera_poses)):
            camera_view_matrix = pu.compute_camera_matrix(scene_camera_view_pose, scene_camera_focus_pose)
            self.scene_cameras[i] = {"viewMatrix": camera_view_matrix, "projectionMatrix": projection_matrix}
            cam_pos, cam_rot_mat = pu.get_pose_from_view_matrix(camera_view_matrix)
            # self.scene_cameras[i]["axis_ids"] = pu.draw_camera_axis(cam_pos, cam_rot_mat, client_id=self.client_id)
        
        for j, obj_camera_view_pose in enumerate(np.linspace(scene_camera_view_traj_y[0], scene_camera_view_traj_y[1], num_camera_poses)):
            camera_view_matrix = pu.compute_camera_matrix(obj_camera_view_pose, scene_camera_focus_pose)
            self.scene_cameras[j+num_camera_poses] = {"viewMatrix": camera_view_matrix, "projectionMatrix": projection_matrix}
            cam_pos, cam_rot_mat = pu.get_pose_from_view_matrix(camera_view_matrix)
            # self.scene_cameras[j+num_camera_poses]["axis_ids"] = pu.draw_camera_axis(cam_pos, cam_rot_mat, client_id=self.client_id)
        
        obj_camera_focus_pose = self.sp_obj_prepared_World2Center_pose[0]
        obj_camera_view_traj_x = [[self.sp_obj_prepared_World2Center_pose[0][0]-0.3, self.sp_obj_prepared_World2Center_pose[0][1], obj_camera_height],
                                  [self.sp_obj_prepared_World2Center_pose[0][0]+0.3, self.sp_obj_prepared_World2Center_pose[0][1], obj_camera_height]]
        obj_camera_view_traj_y = [[self.sp_obj_prepared_World2Center_pose[0][0], self.sp_obj_prepared_World2Center_pose[0][1]-0.3, obj_camera_height],
                                    [self.sp_obj_prepared_World2Center_pose[0][0], self.sp_obj_prepared_World2Center_pose[0][1]+0.3, obj_camera_height]]
        for i, obj_camera_view_pose in enumerate(np.linspace(obj_camera_view_traj_x[0], obj_camera_view_traj_x[1], num_camera_poses)):
            camera_view_matrix = pu.compute_camera_matrix(obj_camera_view_pose, obj_camera_focus_pose)
            self.obj_cameras[i] = {"viewMatrix": camera_view_matrix, "projectionMatrix": projection_matrix}
            cam_pos, cam_rot_mat = pu.get_pose_from_view_matrix(camera_view_matrix)
            # self.obj_cameras[i]["axis_ids"] = pu.draw_camera_axis(cam_pos, cam_rot_mat, client_id=self.client_id)
        
        for j, obj_camera_view_pose in enumerate(np.linspace(obj_camera_view_traj_y[0], obj_camera_view_traj_y[1], num_camera_poses)):
            camera_view_matrix = pu.compute_camera_matrix(obj_camera_view_pose, obj_camera_focus_pose)
            self.obj_cameras[j+num_camera_poses] = {"viewMatrix": camera_view_matrix, "projectionMatrix": projection_matrix}
            cam_pos, cam_rot_mat = pu.get_pose_from_view_matrix(camera_view_matrix)
            # self.obj_cameras[j+num_camera_poses]["axis_ids"] = pu.draw_camera_axis(cam_pos, cam_rot_mat, client_id=self.client_id)


    def get_scene_points_cloud(self, camera_scan=True):
        scene_pc = []
        for cam_id, cam_info in self.scene_cameras.items():
            scene_pc.append(pu.get_pc_from_camera(self.width, self.height, cam_info["viewMatrix"], cam_info["projectionMatrix"], client_id=self.client_id))
        scene_pc = np.concatenate(scene_pc, axis=0)
        return scene_pc
    

    def get_qr_obj_points_cloud(self, camera_scan=True):
        qr_obj_pc = []
        for cam_id, cam_info in self.obj_cameras.items():
            qr_obj_pc.append(pu.get_pc_from_camera(self.width, self.height, cam_info["viewMatrix"], cam_info["projectionMatrix"], client_id=self.client_id))
        qr_obj_pc = np.concatenate(qr_obj_pc, axis=0)
        return qr_obj_pc
    

    def crop_pc(self, pc, low_up_bbox):
        # Crop the points cloud based on the bbox
        # Both are in the world frame
        low_bound, up_bound = low_up_bbox
        mask = np.all((pc >= low_bound) & (pc <= up_bound), axis=1)
        pc = pc[mask]
        return pc


    def sp_compute_observation(self, qr_obj_name, camera_scan=False):
        qr_obj_pc = self.get_qr_obj_points_cloud()
        scene_surface_pc = self.get_scene_points_cloud()
        return scene_surface_pc, qr_obj_pc
    

    def sp_convert_pred_qr_obj_pose(self, qr_obj_name, pred_qr_obj_pose):
        QRsceneSurface_2_QRobjBboxCenter = pu.split_7d(pred_qr_obj_pose.squeeze().detach().cpu().numpy())
        World_2_QRobjBboxCenter = pu.multiply_multi_transforms(self.World2tableSurfaceCenter, QRsceneSurface_2_QRobjBboxCenter)
        return World_2_QRobjBboxCenter
    
    
    def stable_placement_task_init(self):
        # Table Pose
        self.table_base_pose = pu.get_body_pose(self.fixed_scene_name_data["table"]["id"], client_id=self.client_id)
        TableBase2TableCenter = pu.split_7d(self.fixed_scene_name_data["table"]["bbox"][:7])
        self.sp_tableHalfExtents = self.fixed_scene_name_data["table"]["bbox"][7:]
        TableCenter2TableSurfaceCenter = [[0., 0., self.sp_tableHalfExtents[2]], [0., 0., 0., 1.]]
        self.World2tableSurfaceCenter = pu.multiply_multi_transforms(self.table_base_pose, TableBase2TableCenter, TableCenter2TableSurfaceCenter)

    
    def stable_placement_task_robot_init(self):
        from robot_related.ur5_robotiq_pybullet import UR5RobotiqPybulletController, load_ur_robotiq_robot
        from robot_related.robotiq_grasp_planner import RobotiqGraspPlanner

        # Load the robot; We need to use relative pose for the robot later
        self.robot_initial_pose = [[-0.7, 0., 0.79], [0., 0., 0., 1.]]
        robot_id, urdf_path = load_ur_robotiq_robot(robot_initial_pose=self.robot_initial_pose, client_id=self.client_id)
        self.robot = UR5RobotiqPybulletController(robot_id, rng=None, client_id=self.client_id)
        self.robot.update_collision_check()
        self.robot.reset()

        # Object prepared pose
        self.sp_obj_prepared_World2Center_pose = [[self.robot_initial_pose[0][0]+0.1, 0.6, 0.9], [0., 0., 0., 1.]]

        # Grasp Planner
        self.grasp_planner = RobotiqGraspPlanner()

        # Set the camera for data collection
        self.init_camera()
        self.compute_crop_region()
        self.stable_placement_reset()

        # Grasp Pose Initial
        self.grasp_pose_id = None

    
    def compute_crop_region(self):
        # Compute the crop region based on the table pose
        TableBase2LowerBound = [-self.sp_tableHalfExtents[0], -self.sp_tableHalfExtents[1], self.sp_tableHalfExtents[2]]
        TableBase2UpperBound = [self.sp_tableHalfExtents[0], self.sp_tableHalfExtents[1], 3.]
        World2TableCropLowerBound = pu.multiply_multi_transforms(self.table_base_pose, [TableBase2LowerBound, [0., 0., 0., 1.]])[0]
        World2TableCropUpperBound = pu.multiply_multi_transforms(self.table_base_pose, [TableBase2UpperBound, [0., 0., 0., 1.]])[0]
        self.World2SceneCropRegion = [self.to_numpy(World2TableCropLowerBound), self.to_numpy(World2TableCropUpperBound)]
        # Compute the crop region based on the object prepared pose
        PreparedObjBase2LowerrBound = [-0.15, -0.15, 0.]
        PreparedObjBase2UpperBound = [0.15, 0.15, 0.5]
        World2PreparedObjCropLowerBound = pu.multiply_multi_transforms(self.sp_obj_prepared_World2Center_pose, [PreparedObjBase2LowerrBound, [0., 0., 0., 1.]])[0]
        World2PreparedObjCropUpperBound = pu.multiply_multi_transforms(self.sp_obj_prepared_World2Center_pose, [PreparedObjBase2UpperBound, [0., 0., 0., 1.]])[0]
        self.World2ObjCropRegion = [self.to_numpy(World2PreparedObjCropLowerBound), self.to_numpy(World2PreparedObjCropUpperBound)]


    def stable_placement_reset(self):
        self.robot.reset()
        # Open the gripper
        self.robot.open_gripper()
        self.robot.update_collision_check()
    

    def stable_placement_compute_observation(self, qr_obj_name, placed_obj_poses, camera_scan=False):
        # Place the quried object to the prepared area
        qr_obj_id = self.obj_name_data[qr_obj_name]["id"]
        Obj_Center2Base_pose = self.get_obj_center2base_pose(qr_obj_name)
        Obj_half_extents = self.get_obj_half_extents(qr_obj_name)
        InitWorld2ObjCenter_pose = deepcopy(self.sp_obj_prepared_World2Center_pose)
        InitWorld2ObjCenter_pose[0][2] += Obj_half_extents[2]
        pu.set_center_pose(qr_obj_id, InitWorld2ObjCenter_pose, Obj_Center2Base_pose, client_id=self.client_id)
        pu.fix_base(qr_obj_id, client_id=self.client_id)
        # Place the other objects to the scene
        self.re_place_placed_obj_poses(placed_obj_poses=placed_obj_poses, reset_mass=True)
        World2Scene_pc, World2Qr_obj_pc = self.sp_compute_observation(qr_obj_name=qr_obj_name, camera_scan=camera_scan)
        scene_surface_pc = self.crop_pc(World2Scene_pc, self.World2SceneCropRegion)
        qr_obj_pc = self.crop_pc(World2Qr_obj_pc, self.World2ObjCropRegion)
        scene_surface_pc = pc_random_downsample(scene_surface_pc, self.args.max_num_scene_points)
        qr_obj_pc = pc_random_downsample(qr_obj_pc, self.args.max_num_obj_points)
        # Transform the scene points cloud to the table surface center frame
        TableSurfaceCenter2World = p.invertTransform(*self.World2tableSurfaceCenter)
        TableSurfaceCenter2ScenePC = se3_transform_pc(*TableSurfaceCenter2World, scene_surface_pc)
        self.scene_surface_pc = scene_surface_pc
        
        qr_obj_bbox = pu.get_obj_axes_aligned_bbox_from_pc(qr_obj_pc)
        self.World2QrobjCenter_pose = pu.split_7d(qr_obj_bbox[:7])
        QrobjCenter2World_pose = p.invertTransform(*self.World2QrobjCenter_pose)
        ObjCenter2ObjPC = se3_transform_pc(*QrobjCenter2World_pose, qr_obj_pc)
        # ObjCenter2ObjPC = self.obj_name_data[qr_obj_name]["pc"]
        # # Compute the object grasp pose
        # pu.visualize_pc(scene_surface_pc)
        # pu.visualize_pc(ObjCenter2ObjPC)
        self.ObjCenter2PreGripperBase_pose, self.ObjCenter2GripperBase_pose = self.grasp_planner.plan_grasps(ObjCenter2ObjPC, qr_obj_bbox[7:])
        return TableSurfaceCenter2ScenePC, ObjCenter2ObjPC
    
    
    def stable_placement_task_step(self, qr_obj_name, pred_qr_obj_pose, placed_obj_poses):
        placed_obj_poses[qr_obj_name] = None # Add the qr_obj_name to the placed_obj_poses but we do not need its pose
        World2QRobjGripperBase_pose = pu.multiply_multi_transforms(self.World2QrobjCenter_pose, self.ObjCenter2GripperBase_pose)
        World2QRobjPreGripperBase_pose = pu.multiply_multi_transforms(self.World2QrobjCenter_pose, self.ObjCenter2PreGripperBase_pose)
        World2QRobjGoalCenter_pose = self.sp_convert_pred_qr_obj_pose(qr_obj_name, pred_qr_obj_pose)
        World2QRobjGoalGripperBase_pose = pu.multiply_multi_transforms(World2QRobjGoalCenter_pose, self.ObjCenter2GripperBase_pose)

        WorkWell = self.stable_placement_reach_close_back(qr_obj_name, World2QRobjPreGripperBase_pose, World2QRobjGripperBase_pose)
        if not WorkWell: return False

        WorkWell = self.stable_placement_place_open_back(qr_obj_name, World2QRobjGoalGripperBase_pose, placed_obj_poses=placed_obj_poses)
        if not WorkWell: return False

        return True


    def stable_placement_reach_close_back(self, qr_obj_name, World2PreGripperBase_pose, World2QRobjGripperBase_pose):
        qr_obj_id = self.obj_name_data[qr_obj_name]["id"]
        qr_obj_mass = self.obj_name_data[qr_obj_name]["mass"]
        if self.args.rendering:
            self.grasp_pose_id = pu.create_arrow_marker(World2QRobjGripperBase_pose, replace_frame_id=self.grasp_pose_id, client_id=self.client_id)

        maximum_motion_trials = 3
        rtol = 0.2; norm_tol = 0.01
        for i in range(maximum_motion_trials):
            joint_values = self.robot.get_arm_ik(World2PreGripperBase_pose, avoid_collisions=True, mix_solve=True)
            trajectory = self.robot.plan_arm_motion(joint_values)
            if trajectory is None: 
                return
            self.robot.execute_arm_plan(trajectory, realtime=self.args.rendering)
            cur_grasp_pose = self.robot.get_gripper_base_pose()
            if np.isclose(cur_grasp_pose[0], World2PreGripperBase_pose[0], rtol=rtol).all():
                break
            elif i==maximum_motion_trials-1:
                if np.linalg.norm(self.to_numpy(cur_grasp_pose[0]) - self.to_numpy(World2PreGripperBase_pose[0])) < norm_tol:
                    break # Extra check for the final pose
                print(f"Reach Stage: Failed to reach the grasp pose {i+1} times!")
                return False

        # Reach the grasp pose
        print("Reach the grasp pose...")
        gsp_jv_goal = self.robot.get_arm_ik(World2QRobjGripperBase_pose, avoid_collisions=False, mix_solve=True)
        if gsp_jv_goal is None:
            print("Reach Stage: Failed to reach the grasp pose!")
            return False
        discretized_plan = self.robot.plan_arm_joint_values_simple(gsp_jv_goal)
        self.robot.execute_arm_plan(discretized_plan, realtime=self.args.rendering)

        # Close the gripper
        print("Close the gripper...")
        close_gripper = self.robot.plan_gripper_joint_values(self.robot.CLOSED_POSITION)
        self.robot.attach_object(qr_obj_id)
        self.robot.execute_gripper_plan(close_gripper, realtime=self.args.realtime)
        self.robot.update_collision_check()
        pu.set_mass(qr_obj_id, qr_obj_mass, client_id=self.client_id)

        # lift the object
        print("Lift the object...")
        lift_pose = deepcopy(self.robot.get_gripper_base_pose())
        lift_pose[0][2] += 0.1
        jv_goal = self.robot.get_arm_ik(lift_pose, avoid_collisions=False, mix_solve=True)
        if jv_goal is not None:
            jv_start = self.robot.get_arm_joint_values()
            trajectory = np.linspace(jv_start, jv_goal, 240)
            self.robot.execute_arm_plan(trajectory, realtime=self.args.rendering)

        # Go to the waiting pose
        for i in range(maximum_motion_trials):
            trajectory = self.robot.plan_arm_motion(self.robot.HOME)
            self.robot.execute_arm_plan(trajectory, realtime=self.args.rendering)
            if i==maximum_motion_trials-1 and trajectory is None:
                print(f"Reach Stage: Failed to reach the waiting pose {i} times!")
                return False
        return True


    def stable_placement_place_open_back(self, qr_obj_name, World2QRobjGoalGripperBase_pose, placed_obj_poses=None):
        World2PreplaceGripperbase_pose = [list(World2QRobjGoalGripperBase_pose[0]), list(World2QRobjGoalGripperBase_pose[1])]
        World2PreplaceGripperbase_pose[0][2] += 0.1

        maximum_motion_trials = 3
        rtol = 0.2; norm_tol = 0.01
        for i in range(maximum_motion_trials):
            joint_values = self.robot.get_arm_ik(World2PreplaceGripperbase_pose, mix_solve=True)
            trajectory = self.robot.plan_arm_motion(joint_values)
            if trajectory is None: 
                return False

            self.robot.execute_arm_plan(trajectory, realtime=self.args.rendering)
            cur_grasp_pose = self.robot.get_gripper_base_pose()
            if np.isclose(cur_grasp_pose[0], World2PreplaceGripperbase_pose[0], rtol=rtol, atol=0.01).all():
                break
            elif i==maximum_motion_trials-1:
                if np.linalg.norm(self.to_numpy(cur_grasp_pose[0]) - self.to_numpy(World2PreplaceGripperbase_pose[0])) < norm_tol:
                    break # Extra check for the final pose
                print(f"Place Stage: Failed to reach the placement pose {i+1} times!")
                return False

        # Go down to the place pose
        jv_goal = self.robot.get_arm_ik(World2QRobjGoalGripperBase_pose, avoid_collisions=False, mix_solve=True)
        self.robot.set_arm_joint_values(jv_goal)
        
        if jv_goal is None:
            print("Place Stage: Failed to reach the place pose!")
            return False
        trajectory = self.robot.plan_arm_joint_values_simple(jv_goal)
        self.robot.execute_arm_plan(trajectory, realtime=self.args.rendering)

        # open the gripper
        trajectory = self.robot.plan_gripper_joint_values(self.robot.OPEN_POSITION)
        self.robot.execute_gripper_plan(trajectory, realtime=self.args.rendering)
        self.robot.detach()
        self.robot.update_collision_check()

        # Check the stability of the placement
        if placed_obj_poses is not None:
            if not self.post_check_scene_stable(placed_obj_poses, re_place_obj=False):
                print("Place Stage: The placement is not stable!")
                return False

        # Go up the the pre-place pose
        jv_preplace_goal = self.robot.get_arm_ik(World2PreplaceGripperbase_pose, avoid_collisions=False, mix_solve=True)
        if jv_preplace_goal is None:
            print("Place Stage: Failed to reach the pre-place pose!")
            return False
        trajectory = self.robot.plan_arm_joint_values_simple(jv_preplace_goal)
        self.robot.execute_arm_plan(trajectory, realtime=self.args.rendering)

        # Go to the waiting pose
        print("Go to the waiting pose...")
        for i in range(maximum_motion_trials):
            trajectory = self.robot.plan_arm_motion(self.robot.HOME)
            self.robot.execute_arm_plan(trajectory, realtime=self.args.rendering)
            if i==maximum_motion_trials-1 and trajectory is None:
                print(f"Place Stage: Failed to reach the waiting pose {i} times!")
                return False
        return True


    def replay_placement_scene(self, scene_dict_file):
        scene_dict = read_json(scene_dict_file)
        success_scene_cfg = scene_dict["success_scene_cfg"]
        self.replay_obj_dict_id = {}
        with keyboard.Events() as events:
            for episode, placed_obj_poses in success_scene_cfg.items():
                for obj_name, obj_pose in placed_obj_poses.items():

                    while True:
                        key = None
                        event = events.get(0.0001)
                        if event is not None:
                            if isinstance(event, events.Press) and hasattr(event.key, 'char'):
                                key = event.key.char
                        if key == 's':
                            break
                        pu.step(1./240, client_id=self.client_id)
                        time.sleep(1./240)

                    if obj_name not in self.obj_uni_names_dataset: continue

                    if obj_name not in self.replay_obj_dict_id.keys():
                        obj_urdf_path = self.obj_uni_names_dataset[obj_name]["urdf"]
                        obj_label = self.obj_uni_names_dataset[obj_name]["label"]
                        self.replay_obj_dict_id[obj_name] = self.loadURDF(
                            obj_urdf_path, 
                            basePosition=self.rng.uniform(*self.prepare_area), baseOrientation=[0., 0., 0., 1.], 
                            globalScaling=obj_label["globalScaling"], useFixedBase=False)
                    print(f"Placing {obj_name} {obj_pose} in {episode}...")
                    
                    pu.set_pose(self.replay_obj_dict_id[obj_name], obj_pose, client_id=self.client_id)

    
    # Below is the data collection for the stable placement task
    def stable_placement_data_collection(self):
        # We focus on the table top scene
        World_2_QRsceneBase = pu.get_body_pose(self.selected_qr_scene_id, client_id=self.client_id)
        QRsceneBase_2_QRsceneCenter = self.selected_qr_scene_bbox
        World_2_QRsceneCenter = p.multiplyTransforms(World_2_QRsceneBase[0], World_2_QRsceneBase[1], QRsceneBase_2_QRsceneCenter[:3], QRsceneBase_2_QRsceneCenter[3:7])
        QRsceneCenter_2_QRsceneSurfaceCenter = [[0., 0., self.selected_qr_scene_bbox[9]], [0., 0., 0., 1.]]
        World_2_QRsceneSurfaceCenter = pu.multiply_multi_transforms(
            World_2_QRsceneCenter, QRsceneCenter_2_QRsceneSurfaceCenter
        )
        QRscene_half_extents = self.selected_qr_scene_bbox[7:]
        # The scene top surface points cloud
        QRsceneSurface_pc = self.rng.uniform(-QRscene_half_extents, QRscene_half_extents, size=(self.args.max_num_qr_scene_points, 3))
        QRsceneSurface_pc[:, 2] = 0. # The z value is 0 for the top surface
        
        self.QRsceneSurface_2_placed_obj_poses = deepcopy(self.placed_obj_poses)
        for placed_obj_name in self.placed_obj_poses.keys():
            World_2_objBase = self.placed_obj_poses[placed_obj_name]
            objBase_2_objCenter = [self.obj_name_data[placed_obj_name]["bbox"][:3], [0., 0., 0., 1.]]
            World_2_objCenter = pu.multiply_multi_transforms(
                World_2_objBase, objBase_2_objCenter
            )
            QRsceneSurface_2_objCenter = pu.multiply_multi_transforms(
                p.invertTransform(*World_2_QRsceneSurfaceCenter), World_2_objCenter
            )
            self.QRsceneSurface_2_placed_obj_poses[placed_obj_name] = QRsceneSurface_2_objCenter

        cur_scene_pc = [QRsceneSurface_pc]
        obj_names = list(self.QRsceneSurface_2_placed_obj_poses.keys())
        
        World2PlacedObj_poses = {}; sp_dataset = {}
        for j, query_obj_name in enumerate(self.QRsceneSurface_2_placed_obj_poses.keys()):
            query_obj_pc = self.obj_name_data[query_obj_name]["pc"] # in the object center frame
            QRsceneSurface_2_QRobjCenter = self.QRsceneSurface_2_placed_obj_poses[query_obj_name]

            if j > 0: # Record the previous object point cloud
                prev_query_obj_name = obj_names[j-1]
                prev_query_obj_pc = self.obj_name_data[prev_query_obj_name]["pc"] # in the object center frame
                QRsceneSurface_2_prevObjCenter = self.QRsceneSurface_2_placed_obj_poses[prev_query_obj_name]
                transformed_selected_obj_pc = se3_transform_pc(*QRsceneSurface_2_prevObjCenter, prev_query_obj_pc)
                cur_scene_pc.append(transformed_selected_obj_pc)
                World2PlacedObj_poses[prev_query_obj_name] = self.to_numpy(
                    self.placed_obj_poses[prev_query_obj_name][0]+self.placed_obj_poses[prev_query_obj_name][1]
                )
            
            np_cur_scene_pc = pc_random_downsample(
                np.concatenate(cur_scene_pc, axis=0), 
                self.args.max_num_scene_points, 
                autopad=True # file.h5 requires the same number of points to create np array
            )

            sp_dataset[j] = {
                "scene_pc": np_cur_scene_pc,
                "qr_obj_pc": query_obj_pc,
                "qr_obj_pose": self.to_numpy(QRsceneSurface_2_QRobjCenter[0]+QRsceneSurface_2_QRobjCenter[1]),
                "qr_obj_name": query_obj_name,
                "qr_scene_name": self.selected_qr_scene_name,
                "qr_scene_pose": self.to_numpy(World_2_QRsceneBase[0]+World_2_QRsceneBase[1]),
                "World2PlacedObj_poses": deepcopy(World2PlacedObj_poses),
            }
            # pu.visualize_pc(query_obj_pc)
            # pu.visualize_pc(np_cur_scene_pc)

        # Save the data
        self.info["sp_dataset"] = sp_dataset

    
    def stable_placement_eval_step(self, qr_obj_name, pred_qr_obj_pose, placed_obj_poses):
        self.reset()
        World_2_QRobjBboxCenter = self.sp_convert_pred_qr_obj_pose(qr_obj_name, pred_qr_obj_pose)
        QRobjBboxCenter_2_objBase = self.get_obj_center2base_pose(qr_obj_name) # We assume we know the object points cloud, ground truth
        World_2_QRobjBase = pu.multiply_multi_transforms(World_2_QRobjBboxCenter, QRobjBboxCenter_2_objBase)
        placed_obj_poses_copy = placed_obj_poses.copy()
        placed_obj_poses_copy[qr_obj_name] = World_2_QRobjBase
        stable_Flag = self.post_check_scene_stable(placed_obj_poses_copy, reset_mass=True)
        return stable_Flag, World_2_QRobjBase

    
    ######################################
    ######### Rigid Body Transformation ############
    ######################################

    def get_QRregionCenter2ObjCenter_pose_vel(self, objUniName=None):
        if objUniName is None:
            obj_id = self.selected_obj_id
            obj_bbox = self.selected_obj_bbox
        else:
            obj_id = self.obj_name_data[objUniName]["id"]
            obj_bbox = self.obj_name_data[objUniName]["bbox"]
        # Convert Object World2ObjBase pose to QRregionCenter2ObjCenter pose
        World2ObjBase_pos, World2ObjBase_quat = pu.get_body_pose(obj_id, client_id=self.client_id)
        World2ObjCenter_pos, World2ObjCenter_quat = p.multiplyTransforms(World2ObjBase_pos, World2ObjBase_quat, obj_bbox[:3], [0., 0., 0., 1.])
        World2QRsceneBase_pos, World2QRsceneBase_quat = pu.get_body_pose(self.selected_qr_scene_id, client_id=self.client_id)
        World2QRsceneCenter_pos, World2QRsceneCenter_quat = p.multiplyTransforms(World2QRsceneBase_pos, World2QRsceneBase_quat, self.selected_qr_scene_bbox[:3], self.selected_qr_scene_bbox[3:7])
        QRsceneCenter2QRregionCenter_pos, QRsceneCenter2QRregionCenter_quat = self.selected_qr_region[:3], self.selected_qr_region[3:7]
        QRregionCenter2QRsceneCenter_pos, QRregionCenter2QRsceneCenter_quat = p.invertTransform(QRsceneCenter2QRregionCenter_pos, QRsceneCenter2QRregionCenter_quat)
        QRsceneCenter2World_pos, QRsceneCenter2World_quat = p.invertTransform(World2QRsceneCenter_pos, World2QRsceneCenter_quat)
        QRregionCenter2World_pos, QRregionCenter2World_quat = \
            p.multiplyTransforms(
                QRregionCenter2QRsceneCenter_pos, QRregionCenter2QRsceneCenter_quat, 
                QRsceneCenter2World_pos, QRsceneCenter2World_quat
            )
        QRregionCenter2ObjCenter_pos, QRregionCenter2ObjCenter_quat = \
            p.multiplyTransforms(
                QRregionCenter2World_pos, QRregionCenter2World_quat,
                World2ObjCenter_pos, World2ObjCenter_quat
            )
        
        # Convert Object World2ObjBase velocity to QRsceneCenter2ObjCenter velocity
        World2ObjCenter_vel = World2ObjBase_vel = pu.getObjVelocity(obj_id, to_array=True, client_id=self.client_id) # no rotation between the base and the center
        QRregionCenter2ObjCenter_linvel = pu.quat_apply(QRregionCenter2ObjCenter_quat, World2ObjCenter_vel[:3])
        QRregionCenter2ObjCenter_rotvel = pu.quat_apply(QRregionCenter2ObjCenter_quat, World2ObjCenter_vel[3:])
        return (list(QRregionCenter2ObjCenter_pos), list(QRregionCenter2ObjCenter_quat)), \
                self.to_numpy(list(QRregionCenter2ObjCenter_linvel) + list(QRregionCenter2ObjCenter_rotvel))
    

    def convert_BaseLinkPC_2_BboxCenterPC(self, pc, Base2BboxCenter):
        BboxCenter2Base = p.invertTransform(Base2BboxCenter[:3], [0., 0., 0., 1.])
        return se3_transform_pc(*BboxCenter2Base, pc)


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

    
    def record_placement_traj(self, obj_pose, stable_steps):
        # Record the placement trajectory
        if self.args.eval_result:
            if self.selected_obj_name not in self.info["placement_trajs_temp"].keys():
                self.info["placement_trajs_temp"][self.selected_obj_name] = {
                    "pose": [],
                    "stable_steps": []
                }
            self.info["placement_trajs_temp"][self.selected_obj_name]["pose"].append(obj_pose)
            self.info["placement_trajs_temp"][self.selected_obj_name]["stable_steps"].append(stable_steps)

    
    def re_place_placed_obj_poses(self, placed_obj_poses=None, reset_mass=False):
        placed_obj_poses = placed_obj_poses if placed_obj_poses is not None else self.placed_obj_poses
        # Reset placed object pose | when reset, the placed_obj_poses will be empty
        for obj_name, obj_pose in placed_obj_poses.items():
            pu.set_pose(self.obj_name_data[obj_name]["id"], obj_pose, client_id=self.client_id)
            if reset_mass:
                pu.set_mass(self.obj_name_data[obj_name]["id"], self.obj_name_data[obj_name]["mass"], client_id=self.client_id)

    
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
    

    def get_obj_center2base_pose(self, obj_name):
        Base2Center_pose = pu.split_7d(self.obj_name_data[obj_name]["bbox"][:7])
        Center2Base_pose = p.invertTransform(*Base2Center_pose)
        return Center2Base_pose
    

    def get_obj_half_extents(self, obj_name):
        return self.obj_name_data[obj_name]["bbox"][7:]


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
        
    
    def set_obj_joints_to_higher_limit(self, obj_id):
        joints_num = pu.get_num_joints(obj_id, client_id=self.client_id)
        if joints_num > 0:
            joints_limits = np.array([pu.get_joint_limits(obj_id, joint_i, client_id=self.client_id) for joint_i in range(joints_num)])
            pu.set_joint_positions(obj_id, list(range(joints_num)), joints_limits[:, 1], client_id=self.client_id)
            pu.control_joints(obj_id, list(range(joints_num)), joints_limits[:, 1], client_id=self.client_id)
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
