import pybullet as p
import pybullet_data
import pybullet_utils_cust as pu
import os
import numpy as np
from utils import natural_keys, read_json


class PybulletVisualizer:
    # Simple visualizer for frankapanda and objects using pybullet
    def __init__(self, obj_uni_names_dataset, rendering=True) -> None:
        self.rendering = rendering
        self.obj_uni_names_dataset = obj_uni_names_dataset
        self.panda_urdf = "robot_related/franka_description/robots/franka_panda.urdf"
        self.rng = np.random.default_rng()

        self._init_simulator()
        self.load_panda()
        self.load_objects(obj_uni_names_dataset)


    def _init_simulator(self):
        connect_type = p.GUI if self.rendering else p.DIRECT
        self.client_id = p.connect(connect_type) # or p.DIRECT for non-graphical version

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.client_id)
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)

        planeId = self.loadURDF("plane.urdf", 
                basePosition=[0, 0, 0.], 
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), 
                useFixedBase=True)
        pu.change_obj_color(planeId, rgba_color=[1., 1., 1., 0.2], client_id=self.client_id)
        
        
    def load_panda(self):
        self.panda_id = self.loadURDF(self.panda_urdf, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], useFixedBase=True)
        self.panda_joints = [p.getJointInfo(self.panda_id, i, physicsClientId=self.client_id) for i in range(p.getNumJoints(self.panda_id, physicsClientId=self.client_id))]
        self.panda_arm_joints_index = [joint[0] for joint in self.panda_joints if joint[2] == p.JOINT_REVOLUTE]
        self.panda_gripper_joints_index = [joint[0] for joint in self.panda_joints if joint[2] == p.JOINT_PRISMATIC]
        self.panda_joint_names = [joint[1].decode("utf-8") for joint in self.panda_joints if joint[2] == p.JOINT_REVOLUTE]
        self.panda_joint_lower_limits = [joint[8] for joint in self.panda_joints if joint[2] == p.JOINT_REVOLUTE]
        self.panda_joint_upper_limits = [joint[9] for joint in self.panda_joints if joint[2] == p.JOINT_REVOLUTE]
        self.panda_joint_ranges = [joint[9] - joint[8] for joint in self.panda_joints if joint[2] == p.JOINT_REVOLUTE]


    def load_objects(self, obj_uni_names_dataset):
        self.obj_name_data = {}; self.prepare_area = [[-1100., -1100., 0.], [-1000, -1000, 100]]
        cate_uni_names = list(obj_uni_names_dataset.keys())
        for i, obj_uni_name in enumerate(cate_uni_names):
            obj_urdf_path = obj_uni_names_dataset[obj_uni_name]["urdf_path"]
            obj_label = obj_uni_names_dataset[obj_uni_name]["label"]
            try: 
                rand_basePosition, rand_baseOrientation = self.rng.uniform([-5, -5, 0.], [5, 5, 10]), p.getQuaternionFromEuler([self.rng.uniform(0., np.pi)]*3)
                object_id = self.loadURDF(obj_urdf_path, basePosition=rand_basePosition, baseOrientation=rand_baseOrientation, globalScaling=obj_label["globalScaling"], useFixedBase=True)  # Load an object at position [0, 0, 1]
                obj_mesh_num = pu.get_body_mesh_num(object_id, client_id=self.client_id)
                self.obj_name_data[obj_uni_name] = {
                                                    "id": object_id,
                                                    }

                # If the object has joints, we need to set the joints to the lower limit
                self.obj_name_data[obj_uni_name]["joint_limits"] = self.set_obj_joints_to_lower_limit(object_id)
            
            except p.error as e:
                print(f"Failed to load object {obj_uni_name} | Error: {e}")
        
        self.place_all_objects_to_prepare_area()


    def set_obj_pose(self, objUniName, pose):
        if not objUniName in self.obj_name_data:
            print(f"Object {objUniName} is not loaded.")
            return
        obj_id = self.obj_name_data[objUniName]["id"]
        pu.set_pose(obj_id, pose, client_id=self.client_id)

    
    def set_panda_joints(self, joint_positions):
        assert len(joint_positions) == len(self.panda_arm_joints_index), f"joint_positions should have length {len(self.panda_joint_names)}, but got {len(joint_positions)}."
        pu.set_joint_positions(self.panda_id, self.panda_arm_joints_index, joint_positions, client_id=self.client_id)

    
    def set_panda_gripper(self, joint_positions):
        assert len(joint_positions) == len(self.panda_gripper_joints_index), f"joint_positions should have length 2, but got {len(joint_positions)}."
        pu.set_joint_positions(self.panda_id, self.panda_gripper_joints_index, joint_positions, client_id=self.client_id)


    def place_all_objects_to_prepare_area(self):
        for obj_name in self.obj_name_data.keys():
            pu.set_pose(body=self.obj_name_data[obj_name]["id"], 
                pose=(self.rng.uniform(*self.prepare_area), [0., 0., 0., 1.]), client_id=self.client_id)
        

    # Utils Functions
    def draw_camera_axis(self, camera_pose, old_axis_ids=None, axis_length=0.1):
        pu.draw_camera_axis(camera_pose, old_axis_ids=old_axis_ids, axis_length=axis_length, client_id=self.client_id)


    def loadURDF(self, urdf_path, basePosition=None, baseOrientation=None, globalScaling=1.0, useFixedBase=False):
        basePosition = basePosition if basePosition is not None else [0., 0., 0.]
        baseOrientation = baseOrientation if baseOrientation is not None else p.getQuaternionFromEuler([0., 0., 0.])
        return p.loadURDF(urdf_path, basePosition=basePosition, baseOrientation=baseOrientation, 
                          globalScaling=globalScaling, useFixedBase=useFixedBase, physicsClientId=self.client_id)
    

    def set_obj_joints_to_lower_limit(self, obj_id):
        joints_num = pu.get_num_joints(obj_id, client_id=self.client_id)
        if joints_num > 0:
            joints_limits = np.array([pu.get_joint_limits(obj_id, joint_i, client_id=self.client_id) for joint_i in range(joints_num)])
            pu.set_joint_positions(obj_id, list(range(joints_num)), joints_limits[:, 0], client_id=self.client_id)
            pu.control_joints(obj_id, list(range(joints_num)), joints_limits[:, 0], client_id=self.client_id)
            return joints_limits
        

    def step_simulation(self, duration=1./240.):
        pu.step(duration=duration, client_id=self.client_id)