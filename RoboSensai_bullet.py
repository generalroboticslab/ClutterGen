from utils import read_json
import time
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

# Visualization
from tabulate import tabulate


class RoboSensaiBullet:
    def __init__(self, args=None) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_simulator()
        self.update_objects_back_pool()
        self.load_world(num_objects=self.args.num_objects, 
                        random=self.args.random_select, 
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
        # self.obj_categories = read_json(f'{self.args.asset_root}/{self.args.object_folder}/obj_categories.json')
        self.obj_categories = {
            "mustard_bottle": 1
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
                                          halfExtents=self.tableHalfExtents, rgba_color=[1, 1, 1, 1], mass=1e5, client_id=self.client_id)
        self.default_scene_name_id["table"] = self.tableId
        # Default region using table position and table half extents
        self.default_region = [[-self.tableHalfExtents[0], -self.tableHalfExtents[1], self.tableHalfExtents[2]*2],
                                 [self.tableHalfExtents[0], self.tableHalfExtents[1], self.tableHalfExtents[2]*2+0.2]]
        # self.default_region = [[-1., -1., 0.], [1., 1., 1.5]]
        self.prepare_area = [[-1001., -1001., 0.], [-1000, -1000, 1.5]]
        self.default_scene_points = 1024

        default_scene_pc = []
        for name, id in self.default_scene_name_id.items():
            if name == "plane": continue # Plane needs to use default region to sample (will do this later)
            object_pc_local_frame = self.get_link_pc_from_id(id, num_points=self.default_scene_points)
            object_pos, object_quat = pu.get_body_pose(id, client_id=self.client_id)
            object_pc_world_frame = np.array([p.multiplyTransforms(object_pos, object_quat, point, [0., 0., 0., 1.])[0] for point in object_pc_local_frame])
            default_scene_pc.append(object_pc_world_frame)
        self.default_scene_pc = np.concatenate(default_scene_pc, axis=0)


    def load_objects(self, num_objects=10, random=True, default_scaling=0.5, init_region=None):
        self.obj_name_id = {}; init_region = init_region if init_region is not None else self.default_region
        self.obj_name_pc = {}
        if random: # Random choose object categories from the pool and their index
            obj_back_pool_indexes = np.random.choice(np.arange(len(self.obj_back_pool_name)), num_objects, replace=False)
            objects_act_pool_name = self.obj_back_pool_name[obj_back_pool_indexes]
            objects_act_pool_indexes = np.random.randint(self.obj_back_pool_indexes_nums[obj_back_pool_indexes])
        else:
            objects_act_pool_name = self.obj_back_pool_name[:num_objects]
            objects_act_pool_indexes = np.zeros(num_objects, dtype=np.int32)

        for i, obj_name in enumerate(objects_act_pool_name):
            try: 
                basePosition, baseOrientation = np.random.uniform(*init_region), p.getQuaternionFromEuler([np.random.uniform(0., np.pi)]*3)
                object_id = self.loadURDF(f"{self.args.asset_root}/{self.args.object_folder}/{obj_name}/{objects_act_pool_indexes[i]}/mobility.urdf", 
                                        basePosition=basePosition, baseOrientation=baseOrientation, globalScaling=default_scaling)  # Load an object at position [0, 0, 1]
                self.obj_name_id[f"{obj_name}_{objects_act_pool_indexes[i]}"] = object_id
                self.obj_name_pc[f"{obj_name}_{objects_act_pool_indexes[i]}"] = self.get_link_pc_from_id(object_id)
            except:
                print(f"Failed to load object {obj_name}")
        self.num_objs = len(self.obj_name_id)
        self.unplaced_objs_name_id = self.obj_name_id.copy()

    
    def load_world(self, num_objects=10, random=True, default_scaling=0.5):
        self._load_default_scene()
        self._init_misc_variables()
        self._init_obs_act_space()
        self.load_objects(num_objects=num_objects, random=random, default_scaling=default_scaling)


    def post_checker(self, verbose=False):
        self.failed_objs = []; headers = ["Type", "Env ID", "Name", "ID", "Value"]

        for obj_name, obj_id in self.obj_name_id.items():
            obj_vel = pu.getObjVelocity(obj_id)
            if (obj_vel[:3].__abs__() > self.args.vel_threshold[0]).any() \
                or (obj_vel[3:].__abs__() > self.args.vel_threshold[1]).any():
                
                self.failed_objs.append(["VEL_FAIL", self.client_id, obj_name, obj_id, obj_vel])

        if verbose: 
            # Generate the table and print it; Needs 0.0013s to generate one table
            self.check_table = tabulate(self.failed_objs, headers, tablefmt="pretty")
            print(self.check_table)

    
    def step(self, action):
        
        pose_xyz, pose_quat = self.convert_actions(action)
        pu.set_pose(self.selected_obj_id, (pose_xyz, pose_quat), client_id=self.client_id)
        
        self.traj_history = [[0.]* (7 + 6)] * self.args.max_traj_history_len  # obj_pos dimension + obj_vel dimension
        self.accm_vel_reward = 0.
        for self.his_steps in range(self.args.max_traj_history_len):
            self.simstep(1/240)
            obj_pos, obj_quat = pu.get_body_pose(self.selected_obj_id, client_id=self.client_id)
            obj_vel = pu.getObjVelocity(self.selected_obj_id, to_array=True)
            # Update the trajectory history
            self.traj_history[self.his_steps] = obj_pos + obj_quat + obj_vel.tolist()
            # Accumulate velocity reward
            self.accm_vel_reward += -obj_vel[:].__abs__().sum()
            # Jump out if the object is not moving
            if (obj_vel[:3].__abs__() < self.args.vel_threshold[0]).all() \
                and (obj_vel[3:].__abs__() < self.args.vel_threshold[1]).all():
                break

        reward = self.compute_reward() # Must compute reward before observation since we use the velocity to compute reward
        done = self.compute_done()
        observation = self.compute_observations() if not done else self.reset() # This point should be considered as the start of the episode!

        # TOdo set all placed objects back

        return observation, reward, done, None


    def reset(self):
        # Place all objects to the prepare area
        for obj_name in self.obj_name_id.keys():
            pu.set_pose(body=self.obj_name_id[obj_name], pose=(np.random.uniform(*self.prepare_area), [0., 0., 0., 1.]), client_id=self.client_id)
        self.unplaced_objs_name_id = self.obj_name_id.copy()
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
        # Observation space: [obj_id, bbox, action, history (hitory_len * (obj_pos + obj_vel)))]
        self.observation_shape = (1, 1+6+6+self.args.max_traj_history_len*(6+7))
        # Action space: [x, y, z, roll, pitch, yaw]
        self.action_shape = (1, 6)

    
    def _init_misc_variables(self):
        self.args.vel_threshold = self.args.vel_threshold if hasattr(self.args, "vel_threshold") else [1/240, np.pi/2400] # 1m/s^2 and 18 degree/s^2
        self.maximum_steps = self.args.maximum_steps if hasattr(self.args, "maximum_steps") else 128
        self.args.max_traj_history_len = self.args.max_traj_history_len if hasattr(self.args, "max_traj_history_len") else 240


    def reset_buffer(self):
        # Training
        self.moving_steps = 0
        self.done = False
        # Observations
        self.selected_obj_name = np.random.choice(list(self.unplaced_objs_name_id.keys()))
        self.obj_done = True
        self.obj_vel = [0.] * 6
        self.traj_history = [[0.]* (7 + 6)] * self.args.max_traj_history_len  # obj_pos dimension + obj_vel dimension
        self.act_scene_pc = self.default_scene_pc.copy()
        self.last_raw_action = [0.] * 6
        # Rewards
        self.accm_vel_reward = 0.
        self.previous_reward = 0.
        self.success_buf = self.to_torch([0.], dtype=torch.float32)

        
    def convert_actions(self, action):
        # action shape is (num_env, action_dim) / action is action logits we need to convert it to (0, 1)
        action = action.squeeze(dim=0).sigmoid() # Map action to (0, 1)
        action[3:5] = 0. # No x,y rotation
        self.last_raw_action = action.clone()
        
        # action = [x, y, z, roll, pitch, yaw]
        placed_bbox = self.last_observation[1:7]
        step_action_xyz = (action[:3] * (placed_bbox[3:] - placed_bbox[:3]) + placed_bbox[:3]).cpu().tolist()
        step_action_quat = p.getQuaternionFromEuler((action[3:] * 2*np.pi))

        return step_action_xyz, step_action_quat


    def compute_observations(self):
        # We need object description (index), bbox, reward (later)
        if self.obj_done:
            self.selected_obj_name = np.random.choice(list(self.unplaced_objs_name_id.keys()))
            self.obj_done = False
        self.selected_obj_id = self.obj_name_id[self.selected_obj_name]
        self.selected_obj_pc = self.obj_name_pc[self.selected_obj_name]
        bbox_region = sum(self.default_region, [])
        his_traj = sum(self.traj_history, [])

        self.last_observation = self.to_torch([self.selected_obj_id, *bbox_region, *self.last_raw_action, *his_traj], dtype=torch.float32)
        
        return self.last_observation.unsqueeze(0)


    def compute_reward(self):
        vel_reward = self.accm_vel_reward
        self.previous_reward = self.accm_vel_reward
        if self.his_steps <= 10: # Jump to the next object
            self.obj_done = True
            self.unplaced_objs_name_id.pop(self.selected_obj_name)
            vel_reward = 100 if len(self.unplaced_objs_name_id) == 0 else 2

            # Update the scene observation
            # self.act_scene_pc = np.concatenate([self.act_scene_pc, self.obj_name_pc[self.selected_obj_name]], axis=0)

        self.last_reward = self.to_torch([vel_reward], dtype=torch.float32)

        return self.last_reward
        

    def compute_done(self): # The whole episode done
        self.moving_steps += 1
        if self.moving_steps >= self.maximum_steps:
            self.done = True
            self.success_buf[0] = 0
        if len(self.unplaced_objs_name_id) == 0:
            self.done = True
            self.success_buf[0] = 1
        return self.to_torch([self.done], dtype=torch.float32)
        

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
    

    def sample_pc_from_mesh(self, mesh_path, num_points=1024, visualize=False):
        mesh = o3d.io.read_triangle_mesh("assets/objects/ycb_objects_origin_at_center_vhacd/mustard_bottle/0/textured_objs/collision.obj")

        # Sample a point cloud from the mesh
        point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

        # Convert the Open3D point cloud to a NumPy array
        point_cloud_np = np.asarray(point_cloud.points)

        if visualize:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
            o3d.visualization.draw_geometries([pcd])

        return point_cloud_np
    

    def get_link_pc_from_id(self, obj_id, link_index=-1, num_points=1024):
        object_info = pu.get_link_collision_shape(obj_id, link_index)
        object_type, object_mesh_scale, object_mesh_path = object_info[2], object_info[3], object_info[4]
        if object_type == p.GEOM_MESH:
            return self.sample_pc_from_mesh(object_mesh_path, num_points)
        elif object_type == p.GEOM_BOX:
            # object_mesh_scale is object dimension if object_type is GEOM_BOX
            object_halfExtents = np.array(object_mesh_scale) / 2
            return np.random.uniform(-object_halfExtents, object_halfExtents, size=(num_points, 3))
        elif object_type == p.GEOM_CYLINDER:
            raise NotImplementedError


    def to_torch(self, x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=self.device)
    

if __name__=="__main__":
    from argparse import ArgumentParser

    args = ArgumentParser()
    args.rendering = True
    args.debug = False
    args.asset_root = "assets"
    args.object_folder = "objects/ycb_objects_origin_at_center_vhacd"
    args.num_objects = 1
    args.random_select = False
    args.default_scaling = 0.5
    args.realtime = True
    args.force_threshold = 20.
    args.vel_threshold = [1/240, np.pi/2400] # 1m/s^2 and 18 degree/s^2

    env = RoboSensaiBullet(args)

    all_pc = np.concatenate(list(env.obj_name_pc.values()) + [env.default_scene_pc], axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pc)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, coord_frame])

    while True:
        random_action = env.to_torch(np.random.uniform(-1., 1., size=(1, 6)), dtype=torch.float32)
        env.step(random_action)

    # env.step_manual()
