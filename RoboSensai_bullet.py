from utils import read_json
from torch_utils import tf_apply
import time
import os
import numpy as np
from math import ceil
import torch
import open3d as o3d

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
    def __init__(self, args=None) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.numpy_dtype = np.float16 if (hasattr(self.args, "use_bf16") and self.args.use_bf16) else np.float32
        self.tensor_dtype = torch.bfloat16 if (hasattr(self.args, "use_bf16") and self.args.use_bf16) else torch.float32
        self.rng = np.random.default_rng(args.seed if hasattr(args, "seed") else None)
        self._init_simulator()
        self.update_objects_back_pool()
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


    def update_objects_back_pool(self):
        # self.obj_categories = read_json(f'{self.args.asset_root}/{self.args.object_pool_folder}/obj_categories.json')
        self.obj_categories = {
            "mustard_bottle": 1,
            "bowl": 1,
            "ball": 1,
            "potted_meat_can": 1,
            "tomato_soup_can": 1,
            "master_chef_can": 1,
            "pitcher_base": 1,
            "pentagram": 1,
            "cuboid": 1,
            "cube": 1,
            "cube_arrow": 1,
            "banana": 1,
            "mug": 1,
            "bleach_cleanser": 1,
            "cracker_box": 1,
            "sugar_box": 1,
            # "power_drill": 1, # Power drill is too difficult to place need 100 steps to be stable
        }
        self.obj_back_pool_name = np.array(list(self.obj_categories.keys()))
        self.obj_back_pool_indexes_nums = np.array(list(self.obj_categories.values()))


    def _load_default_scene(self):
        self.default_scene_name_id = {}
        # Plane
        self.planeId = self.loadURDF("plane.urdf")
        self.default_scene_name_id["plane"] = self.planeId
        # Walls
        self.wallHalfExtents = [1., 0.05, 0.75]
        # self.wallxId = pu.create_box_body(position=[-1, 0., self.wallHalfExtents[2]], orientation=p.getQuaternionFromEuler([0., 0., np.pi/2]),
        #                                   halfExtents=self.wallHalfExtents, rgba_color=[1, 1, 1, 1], mass=0, client_id=self.client_id)
        # self.wallyId = pu.create_box_body(position=[0., -1, self.wallHalfExtents[2]], orientation=p.getQuaternionFromEuler([0., 0., 0.]),
        #                                     halfExtents=self.wallHalfExtents, rgba_color=[1, 1, 1, 1], client_id=self.client_id)
        # Table
        self.tableHalfExtents = [0.2, 0.3, 0.2]
        self.tableId = pu.create_box_body(position=[0., 0., self.tableHalfExtents[2]], orientation=p.getQuaternionFromEuler([0., 0., 0.]),
                                          halfExtents=self.tableHalfExtents, rgba_color=[1, 1, 1, 1], mass=0., client_id=self.client_id)
        self.default_scene_name_id["table"] = self.tableId
        # Default region using table position and table half extents
        self.default_region = [[-self.tableHalfExtents[0], -self.tableHalfExtents[1], self.tableHalfExtents[2]*2],
                               [self.tableHalfExtents[0], self.tableHalfExtents[1], self.tableHalfExtents[2]*2+0.2]]
        self.default_region_np = self.to_numpy(sum(self.default_region, []))
        # self.default_region = [[-1., -1., 0.], [1., 1., 1.5]]
        self.prepare_area = [[-1001., -1001., 0.], [-1000, -1000, 1.5]]
        self.default_scene_points = 2048

        default_scene_pc = []
        for name, id in self.default_scene_name_id.items():
            if name == "plane": continue # Plane needs to use default region to sample (will do this later)
            object_pc_local_frame = self.get_link_pc_from_id(id, num_points=self.default_scene_points)
            object_pos, object_quat = pu.get_body_pose(id, client_id=self.client_id)
            object_pc_world_frame = self.to_numpy([p.multiplyTransforms(object_pos, object_quat, point, [0., 0., 0., 1.])[0] for point in object_pc_local_frame])
            default_scene_pc.append(object_pc_world_frame)
        self.default_scene_pc = np.concatenate(default_scene_pc, axis=0)
        self.default_scene_pc_feature = self.pc_extractor(self.to_torch(self.default_scene_pc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)).squeeze(0).detach().cpu().numpy().astype(self.numpy_dtype)


    def load_objects(self, num_objects=10, random=True, default_scaling=0.5, init_region=None):
        self.obj_name_id = {}; init_region = init_region if init_region is not None else self.default_region
        self.obj_name_pc = {}; self.obj_name_pc_feature = {}
        if random: # Random choose object categories from the pool and their index
            obj_back_pool_indexes = self.rng.choice(np.arange(len(self.obj_back_pool_name)), num_objects, replace=False)
            objects_act_pool_name = self.obj_back_pool_name[obj_back_pool_indexes]
            objects_act_pool_indexes = self.rng.integers(self.obj_back_pool_indexes_nums[obj_back_pool_indexes])
        else:
            objects_act_pool_name = self.obj_back_pool_name[:num_objects]
            objects_act_pool_indexes = np.zeros(num_objects, dtype=np.int32)

        for i, obj_name in enumerate(objects_act_pool_name):
            obj_urdf_file = f"{self.args.asset_root}/{self.args.object_pool_folder}/{obj_name}/{objects_act_pool_indexes[i]}/mobility.urdf"
            assert os.path.exists(obj_urdf_file), f"Object {obj_name} does not exist! Given path: {obj_urdf_file}"
            
            try: 
                basePosition, baseOrientation = self.rng.uniform(*init_region), p.getQuaternionFromEuler([self.rng.uniform(0., np.pi)]*3)
                object_id = self.loadURDF(f"{self.args.asset_root}/{self.args.object_pool_folder}/{obj_name}/{objects_act_pool_indexes[i]}/mobility.urdf", 
                                        basePosition=basePosition, baseOrientation=baseOrientation, globalScaling=default_scaling)  # Load an object at position [0, 0, 1]
                self.obj_name_id[f"{obj_name}_{objects_act_pool_indexes[i]}"] = object_id
                self.obj_name_pc[f"{obj_name}_{objects_act_pool_indexes[i]}"] = self.get_link_pc_from_id(object_id, num_points=1024)
            except:
                print(f"Failed to load object {obj_name}")
        
        # Pre-extract the feature for each object and store here 
        with torch.no_grad():
            for obj_name, obj_pc in self.obj_name_pc.items(): # Extract the feature for each object
                self.obj_name_pc_feature[obj_name] = self.pc_extractor(self.to_torch(obj_pc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)).squeeze(0).detach().cpu().numpy().astype(self.numpy_dtype)
        self.num_pool_objs = len(self.obj_name_id)
        self.args.num_placing_objs = min(self.args.num_placing_objs, self.num_pool_objs)
        self.reset_unplaced_objs()

    
    def load_world(self, num_objects=10, random=True, default_scaling=0.5):
        self._init_pc_extractor()
        self._load_default_scene()
        self._init_misc_variables()
        self._init_obs_act_space()
        self.load_objects(num_objects=num_objects, random=random, default_scaling=default_scaling)


    def post_checker(self, verbose=False):
        self.failed_objs = []; headers = ["Type", "Env ID", "Name", "ID", "Value"]

        for obj_name, obj_id in self.obj_name_id.items():
            obj_vel = pu.getObjVelocity(obj_id, client_id=self.client_id)
            if (obj_vel[:3].__abs__() > self.args.vel_threshold[0]).any() \
                or (obj_vel[3:].__abs__() > self.args.vel_threshold[1]).any():
                
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
        
        # stepping == 0 means previous action is still running, we need to wait until the object is stable
        # In-pysical step
        if self.info['stepping'] == 0.:
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
                if ((obj_vel[:3].__abs__() < self.args.vel_threshold[0]).all() \
                    and (obj_vel[3:].__abs__() < self.args.vel_threshold[1]).all()) \
                    or self.his_steps >= self.args.max_traj_history_len:
                    self.info['stepping'] = 1.
                    break
        
        # Post-physical step
        reward = self.compute_reward() if self.info['stepping']==1 else 0. # Must compute reward before observation since we use the velocity to compute reward
        done = self.compute_done() if self.info['stepping']==1 else False
        # Success Visualization here since observation needs to be reset. Only for evaluation!
        if self.args.rendering and self.info['stepping']==1:
            print(f"Obj Name: {self.selected_obj_name} | Stable Steps: {self.his_steps}")
            if done and self.info['success'] == 1:
                print(f"Successfully Place {self.selected_obj_name}! | Stable steps: {self.his_steps}")
                if hasattr(self.args, "eval_result") and self.args.eval_result: time.sleep(3.)

        if self.info['stepping'] == 1.: observation = self.compute_observations() if not done else self.reset() # This point should be considered as the start of the episode!
        else: observation = self.last_observation

        # Reset placed object pose | when reset, the placed_obj_poses will be empty
        if self.info['stepping'] == 1.:
            for obj_name, obj_pose in self.placed_obj_poses.items():
                pu.set_pose(self.obj_name_id[obj_name], obj_pose, client_id=self.client_id)

        return observation, reward, done, self.info


    def reset(self):
        # Place all objects to the prepare area
        for obj_name in self.obj_name_id.keys():
            pu.set_pose(body=self.obj_name_id[obj_name], pose=(self.rng.uniform(*self.prepare_area), [0., 0., 0., 1.]), client_id=self.client_id)
        self.reset_unplaced_objs()
        # Reset Training buffer and update the start observation for the next episode
        self.reset_buffer()
        observation = self.compute_observations()
        return observation


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
        self.observation_shape = (1, 1024+1024+6+6+self.args.max_traj_history_len*(6+7))
        # Action space: [x, y, z, roll, pitch, yaw]
        self.action_shape = (1, 6)
        # slice
        self.act_scene_feature_slice = slice(0, 1024)
        self.selected_obj_pc_feature_slice = slice(1024, 2048)
        self.placed_region_slice = slice(2048, 2054)
        self.last_action_slice = slice(2054, 2060)
        self.traj_history_slice = slice(2060, 2060+self.args.max_traj_history_len*(6+7))

    
    def _init_misc_variables(self):
        self.args.vel_threshold = self.args.vel_threshold if hasattr(self.args, "vel_threshold") else [1/240, np.pi/2400] # 1m/s^2 and 18 degree/s^2
        self.args.num_placing_objs = self.args.num_placing_objs if hasattr(self.args, "num_placing_objs") else 1
        self.args.max_traj_history_len = self.args.max_traj_history_len if hasattr(self.args, "max_traj_history_len") else 240
        self.args.step_divider = self.args.step_divider if hasattr(self.args, "step_divider") else 6
        self.args.reward_pobj = self.args.reward_pobj if hasattr(self.args, "reward_pobj") else 10
        self.args.vel_reward_scale = self.args.vel_reward_scale if hasattr(self.args, "vel_reward_scale") else 0.005
        self.args.max_stable_steps = self.args.max_stable_steps if hasattr(self.args, "max_stable_steps") else 50
        self.max_trials = self.args.max_trials if hasattr(self.args, "max_trials") else 10
        # Buffer does not need to be reset
        self.info = {'success': 0., 'stepping': 1.}

    
    def _init_pc_extractor(self):
        self.pc_extractor = get_model(num_class=40, normal_channel=False).to(self.device) # num_classes is used for loading checkpoint to make sure the model is the same
        self.pc_extractor.load_checkpoint(ckpt_path="PointNet_Model/checkpoints/best_model.pth", evaluate=True, map_location=self.device)


    def reset_buffer(self):
        # Training
        self.moving_steps = 0
        self.done = False
        # Observations
        self.obj_done = True
        self.act_scene_pc = self.default_scene_pc.copy()
        self.act_scene_pc_feature = self.default_scene_pc_feature.copy()
        self.traj_history = [[0.]* (7 + 6)] * self.args.max_traj_history_len  # obj_pos dimension + obj_vel dimension
        self.last_raw_action = np.zeros(6, dtype=self.numpy_dtype)
        # Rewards
        self.accm_vel_reward = 0.
        # Scene
        self.placed_obj_poses = {}

    
    def reset_unplaced_objs(self):
        if self.args.random_select_placing:
            unplaced_objs_name = self.rng.choice(list(self.obj_name_id.keys()), self.args.num_placing_objs, replace=False)
        else:
            unplaced_objs_name = list(self.obj_name_id.keys())[:self.args.num_placing_objs]
        self.unplaced_objs_name_id = {obj_name: self.obj_name_id[obj_name] for obj_name in unplaced_objs_name}

        
    def convert_actions(self, action):
        # action shape is (num_env, action_dim) / action is action logits we need to convert it to (0, 1)
        action = action.squeeze(dim=0).sigmoid().cpu().numpy() # Map action to (0, 1)
        action[3:5] = 0. # No x,y rotation
        self.last_raw_action = action.copy()
        
        # action = [x, y, z, roll, pitch, yaw]
        placed_bbox = self.last_observation[self.placed_region_slice]
        step_action_xyz = (action[:3] * (placed_bbox[3:] - placed_bbox[:3]) + placed_bbox[:3]).tolist()
        step_action_quat = p.getQuaternionFromEuler((action[3:] * 2*np.pi))

        if self.args.rendering and not hasattr(self, "region_vis_id"):
            self.region_vis_id = pu.draw_box_body(position=(placed_bbox[3:] + placed_bbox[:3])/2, orientation=p.getQuaternionFromEuler([0., 0., 0.]),
                                                    halfExtents=abs(placed_bbox[3:] - placed_bbox[:3])/2, rgba_color=[1, 0, 0, 0.3], client_id=self.client_id)

        return step_action_xyz, step_action_quat


    def compute_observations(self):
        # We need object description (index), bbox, reward (later)
        if self.obj_done:
            self.obj_done = False
            # Always start from random picked object
            self.selected_obj_name = self.rng.choice(list(self.unplaced_objs_name_id.keys()))
            self.selected_obj_id = self.obj_name_id[self.selected_obj_name]
            self.selected_obj_pc = self.obj_name_pc[self.selected_obj_name]
            self.selected_obj_pc_feature = self.obj_name_pc_feature[self.selected_obj_name]
        
        # Convert history to tensor
        his_traj = self.to_numpy(self.traj_history).flatten()

        self.last_observation = np.concatenate([self.act_scene_pc_feature, self.selected_obj_pc_feature, self.default_region_np, self.last_raw_action, his_traj])
        
        return self.last_observation


    def compute_reward(self):
        self.moving_steps += 1
        vel_reward = self.args.vel_reward_scale * self.accm_vel_reward
        if self.his_steps <= self.args.max_stable_steps: # Jump to the next object, object is stable within 10 simulation steps
            self.obj_done = True; self.moving_steps = 0
            self.unplaced_objs_name_id.pop(self.selected_obj_name)
            
            # Record the successful object pose
            selected_obj_pose = pu.get_body_pose(self.selected_obj_id, client_id=self.client_id)
            self.placed_obj_poses[self.selected_obj_name] = selected_obj_pose
            vel_reward += max(100, self.args.reward_pobj * self.args.num_placing_objs) if len(self.placed_obj_poses) >= self.args.num_placing_objs \
                          else self.args.reward_pobj
            # Update the scene observation | transform the selected object point cloud to world frame using the current pose
            transformed_selected_obj_pc = self.to_numpy([p.multiplyTransforms(selected_obj_pose[0], selected_obj_pose[1], point, [0., 0., 0., 1.])[0] for point in self.selected_obj_pc])
            # transformed_selected_obj_pc = tf_apply(self.to_torch(selected_obj_pose[1]), self.to_torch(selected_obj_pose[0]), self.to_torch(self.selected_obj_pc)).cpu().numpy()
            self.act_scene_pc = np.concatenate([self.act_scene_pc, transformed_selected_obj_pc], axis=0)
            # Run one inference needs ~0.5s!
            with torch.no_grad():
                self.act_scene_pc_feature = self.pc_extractor(self.to_torch(self.act_scene_pc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)).squeeze(0).cpu().numpy()

        self.last_reward = vel_reward

        return self.last_reward
        

    def compute_done(self): # The whole episode done
        # Failed condition
        if self.moving_steps >= self.max_trials:
            self.done = True
            self.info['success'] = 0.
        # Goal condition
        if len(self.placed_obj_poses) >= self.args.num_placing_objs:
            self.done = True
            self.info['success'] = 1.
        return self.done
        

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


    def loadURDF(self, urdf_path, basePosition=None, baseOrientation=None, globalScaling=1.0):
        basePosition = basePosition if basePosition is not None else [0., 0., 0.]
        baseOrientation = baseOrientation if baseOrientation is not None else p.getQuaternionFromEuler([0., 0., 0.])
        return p.loadURDF(urdf_path, basePosition=basePosition, baseOrientation=baseOrientation, 
                          globalScaling=globalScaling, physicsClientId=self.client_id)
    

    def sample_pc_from_mesh(self, mesh_path, mesh_scaling=[1., 1., 1.], num_points=1024, visualize=False):
        mesh = o3d.io.read_triangle_mesh(mesh_path)

        # Sample a point cloud from the mesh
        point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

        # Convert the Open3D point cloud to a NumPy array
        point_cloud_np = np.asarray(point_cloud.points) * self.to_numpy(mesh_scaling)

        if visualize:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
            o3d.visualization.draw_geometries([pcd])

        return point_cloud_np
    

    def get_link_pc_from_id(self, obj_id, link_index=-1, num_points=1024):
        object_info = pu.get_link_collision_shape(obj_id, link_index, cline_id=self.client_id)
        object_type, object_mesh_scale, object_mesh_path = object_info[2], object_info[3], object_info[4]
        if object_type == p.GEOM_MESH:
            return self.sample_pc_from_mesh(object_mesh_path, object_mesh_scale, num_points)
        elif object_type == p.GEOM_BOX:
            # object_mesh_scale is object dimension if object_type is GEOM_BOX
            object_halfExtents = self.to_numpy(object_mesh_scale) / 2
            return self.rng.uniform(-object_halfExtents, object_halfExtents, size=(num_points, 3))
        elif object_type == p.GEOM_CYLINDER:
            raise NotImplementedError


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
    args.object_pool_folder = "objects/ycb_objects_origin_at_center_vhacd"
    args.num_pool_objs = 13
    args.num_placing_objs = 1
    args.random_select_pool = False
    args.random_select_placing = True
    args.default_scaling = 0.5
    args.realtime = True
    args.force_threshold = 20.
    args.vel_threshold = [1/240, np.pi/2400] # Probably need to compute acceleration threshold!

    env = RoboSensaiBullet(args)

    all_pc = np.concatenate(list(env.obj_name_pc.values()) + [env.default_scene_pc], axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pc)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, coord_frame])

    while True:
        random_action = env.to_torch(env.rng.uniform(-1., 1., size=(1, 6)))
        env.step(random_action)

    # env.step_manual()
