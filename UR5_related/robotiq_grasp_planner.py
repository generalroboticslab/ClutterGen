import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory path
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import pybullet as p
import pybullet_utils_cust as pu
import pybullet_data
import os
from utils import save_json, read_json, natural_keys, se3_transform_pc
import numpy as np
from math import pi
from argparse import Namespace
from collections import namedtuple
import time


class RobotiqGraspPlanner:
    def __init__(self):
        pass

    def plan_grasps(self, ObjCenter2ObjPC, ObjhalfExtents):
        # Convert point cloud to numpy array

        # Plan grasps
        
        GripperBase2Grasp_pose = [[0, 0, 0.20], [0., 0., 0., 1.]]
        GripperBase2PreGrasp_pose = [[0, 0, 0.25], [0., 0., 0., 1.]]
        z_half_extent = ObjhalfExtents[2]
        x_half_extent = ObjhalfExtents[0]

        ObjCenter2Gripper_pos = [-(x_half_extent-GripperBase2Grasp_pose[0][2]), 0., 0.]
        ObjCenter2Gripper_pos = [0., 0., z_half_extent+GripperBase2Grasp_pose[0][2]]
        ObjCenter2PreGripper_pos = [0., 0., z_half_extent+GripperBase2PreGrasp_pose[0][2]]
        
        InitGripperFaceDirection = [0., 0., 1.]
        ObjCenter2Gripper_dir = [-v for v in ObjCenter2Gripper_pos]
        
        ObjCenter2GripperBase_pose = [ObjCenter2Gripper_pos, pu.getQuaternionFromTwoVectors(InitGripperFaceDirection, ObjCenter2Gripper_dir)]
        ObjCenter2PreGripperBase_pose = [ObjCenter2PreGripper_pos, ObjCenter2GripperBase_pose[1]]

        return ObjCenter2PreGripperBase_pose, ObjCenter2GripperBase_pose
    

class RobotiqGraspLabeler:
    OPEN_POSITION = [0] * 6
    CLOSED_POSITION = 0.72 * np.array([1, 1, -1, 1, 1, -1])
    GROUPS = {
        'gripper': ['finger_joint', 'left_inner_knuckle_joint', 'left_inner_finger_joint', 'right_outer_knuckle_joint',
                    'right_inner_knuckle_joint', 'right_inner_finger_joint']
    }
    JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                        'jointReactionForces', 'appliedJointMotorTorque'])
    def __init__(self, asset_root, group_folder='group_objects', group_name='group0_dinning_table'):
        self.asset_root = asset_root
        self.group_folder = group_folder
        assert os.path.isdir(self.asset_root), f"Source Folder path '{self.asset_root}' does not exist."
        self.group_folder_path = os.path.join(self.asset_root, self.group_folder, group_name)


    # Start PyBullet in GUI mode.
    def _init_simulator(self):
        self.client_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")

    
    def _init_misc_variables(self):
        self.args = Namespace()
        self.prepare_area = [[-1000, -1000, 0.], [-1005, -1005, 50.]]
        self.obj_uni_names_dataset = {}
        self.obj_name_data = {}
        self.cur_objUniName = None
        self.rng = np.random.RandomState(0)
        self.args.max_num_urdf_points = 2048

    
    def reset(self, reset_all=True):
        self.cur_objUniName = self.obj_UniNames[0]
        # Place all objects to the prepare area
        for obj_name in self.obj_name_data.keys():
            self.place_obj2prepare_area(obj_name)

    
    def place_obj2prepare_area(self, obj_name):
        obj_id = self.obj_name_data[obj_name]["id"]
        pu.set_pose(body=obj_id, pose=(self.rng.uniform(*self.prepare_area), [0., 0., 0., 1.]), client_id=self.client_id)
        pu.set_mass(obj_id, mass=0., client_id=self.client_id)

    
    def update_objects_database(self):
        # Right now we only have two stages, object can not appear in both stages! We need to figure out how to deal with this problem
        # 0: "Table", "Bookcase", "Dishwasher", "Microwave", all storage furniture
        self.obj_uni_names_dataset = {}
        obj_categories = sorted(os.listdir(self.group_folder_path), key=natural_keys)
        for cate in obj_categories:
            obj_folder = os.path.join(self.group_folder_path, cate)
            obj_indexes = sorted(os.listdir(obj_folder), key=natural_keys)
            for idx in obj_indexes:
                obj_uni_name = f"{cate}_{idx}"
                obj_urdf_path = f"{self.group_folder_path}/{cate}/{idx}/mobility.urdf"
                obj_label_path = f"{self.group_folder_path}/{cate}/{idx}/label.json"
                assert os.path.exists(obj_urdf_path), f"Object {obj_uni_name} does not exist! Given path: {obj_urdf_path}"
                assert os.path.exists(obj_label_path), f"Object {obj_uni_name} does not exist! Given path: {obj_label_path}"
                self.obj_uni_names_dataset.update({
                    obj_uni_name: {
                        "urdf": obj_urdf_path, 
                        "label": read_json(obj_label_path), 
                        "urdfpath": obj_urdf_path, 
                        "labelpath": obj_label_path
                        }
                    })

    
    def load_objects(self):
        # Load objects
        self.obj_name_data = {} # The object can be queried or placed
        cate_uni_names = list(self.obj_uni_names_dataset.keys())

        for i, obj_uni_name in enumerate(cate_uni_names):
            obj_urdf_path = self.obj_uni_names_dataset[obj_uni_name]["urdf"]
            obj_label = self.obj_uni_names_dataset[obj_uni_name]["label"]
            Center2GripperBase_pose = obj_label["Center2GripperBase_pose"] if "Center2GripperBase_pose" in obj_label.keys() else None
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
                                                    "Center2GripperBase_pose": Center2GripperBase_pose,
                                                    }

                # If the object has joints, we need to set the joints to the lower limit
                self.obj_name_data[obj_uni_name]["joint_limits"] = self.set_obj_joints_to_lower_limit(object_id)
            
            except p.error as e:
                print(f"Failed to load object {obj_uni_name} | Error: {e}")
        
        self.obj_UniNames = list(self.obj_name_data.keys())

    
    def load_gripper(self):
        self.gripper_id = self.loadURDF("UR5_related/ur5_robotiq_description/urdf/robotiq_2f_85_gripper_visualization/urdf/robotiq_arg2f_85_model.urdf", 
                                        basePosition=[0, 0, 0.], 
                                        useFixedBase=True)

        pu.change_obj_color(self.gripper_id, rgba_color=[0., 0., 0., .2])
        gripper_links_num = pu.get_num_links(self.gripper_id, client_id=self.client_id)
        for link_id in [-1]+list(range(gripper_links_num)):
            p.setCollisionFilterGroupMask(self.gripper_id, link_id, 0, 0, physicsClientId=self.client_id)
        gripper_face_direction = [0., 0., 1.]

        joint_infos = [p.getJointInfo(self.gripper_id, joint_index, physicsClientId=self.client_id) \
                       for joint_index in range(p.getNumJoints(self.gripper_id, physicsClientId=self.client_id))]
        self.JOINT_INDICES_DICT = {entry[1].decode('ascii'): entry[0] for entry in joint_infos}
        self.GROUP_INDEX = {group_name: [self.JOINT_INDICES_DICT[joint_name] for joint_name in self.GROUPS[group_name]] \
                            for group_name in self.GROUPS}

    
    def disable_collisions(self, obj_id):
        obj_links_num = pu.get_num_links(obj_id, client_id=self.client_id)
        for link_id in [-1]+list(range(obj_links_num)):
            p.setCollisionFilterGroupMask(obj_id, link_id, 0, 0, physicsClientId=self.client_id)
    

    def enable_collisions(self, obj_id):
        obj_links_num = pu.get_num_links(obj_id, client_id=self.client_id)
        for link_id in [-1]+list(range(obj_links_num)):
            p.setCollisionFilterGroupMask(obj_id, link_id, 1, 1, physicsClientId=self.client_id)

    
    def open_gripper(self, realtime=False):
        waypoints = self.plan_gripper_joint_values(self.OPEN_POSITION)
        self.execute_gripper_plan(waypoints, realtime)


    def close_gripper(self, realtime=False):
        waypoints = self.plan_gripper_joint_values(self.CLOSED_POSITION)
        self.execute_gripper_plan(waypoints, realtime)


    def execute_gripper_plan(self, plan, realtime=False):
        """
        execute a discretized gripper plan (list of waypoints)
        """
        if plan is None: return
        for wp in plan:
            self.control_gripper_joints(wp)
            p.stepSimulation(physicsClientId=self.client_id)
            if realtime:
                time.sleep(1. / 240.)
        pu.step(2, self.client_id)


    def plan_gripper_joint_values(self, goal_joint_values, start_joint_values=None, duration=None):
        if start_joint_values is None:
            start_joint_values = self.get_gripper_joint_values()
        num_steps = 240 if duration is None else int(duration*240)
        discretized_plan = np.linspace(start_joint_values, goal_joint_values, num_steps)
        return discretized_plan
    

    def control_gripper_joints(self, joint_values, control_type='limited'):
        pu.control_joints(self.gripper_id, self.GROUP_INDEX['gripper'], joint_values, control_type, client_id=self.client_id)

    
    def get_gripper_joint_values(self):
        return [self.get_joint_state(i).jointPosition for i in self.GROUP_INDEX['gripper']]
    

    def get_joint_state(self, joint_index):
        return self.JointState(*p.getJointState(self.gripper_id, joint_index, physicsClientId=self.client_id))


    def prepare_cur_obj(self):
        self.cur_object_id = self.obj_name_data[self.cur_objUniName]["id"]
        self.cur_obj_urdf_path = self.obj_uni_names_dataset[self.cur_objUniName]["urdf"]
        self.cur_obj_bbox = self.obj_name_data[self.cur_objUniName]["bbox"]
        self.cur_obj_pc = self.obj_name_data[self.cur_objUniName]["pc"]
        self.cur_obj_mass = self.obj_name_data[self.cur_objUniName]["mass"]
        self.cur_obj_index = self.obj_UniNames.index(self.cur_objUniName)
        self.cur_rpy = [0., 0., 0.]
        self.old_gb_scaling = None
        self.placement_pos_world2center = [0.5, 0, self.cur_obj_bbox[9]+0.5]
        self.placement_pos_world2base = p.multiplyTransforms(self.placement_pos_world2center, [0, 0, 0, 1.], -self.cur_obj_bbox[:3], [0, 0, 0, 1.])[0]
        pu.set_pose(self.cur_object_id, (self.placement_pos_world2base, p.getQuaternionFromEuler(self.cur_rpy)))
        self.Center2GripperBase_pos_rpy = self.obj_name_data[self.cur_objUniName]["Center2GripperBase_pose"]
        if self.Center2GripperBase_pos_rpy is None: # Pos and RPY
            self.Center2GripperBase_pos_rpy = [[0, 0, 0.20], [0., 0., 0.]]

    
    def save_cur_obj_grasp_pose(self):
        self.obj_name_data[self.cur_objUniName]["Center2GripperBase_pose"] = self.Center2GripperBase_pos_rpy
        self.obj_uni_names_dataset[self.cur_objUniName]["label"]["Center2GripperBase_pose"] = self.Center2GripperBase_pos_rpy
        obj_label_path = self.obj_uni_names_dataset[self.cur_objUniName]["labelpath"]
        save_json(self.obj_uni_names_dataset[self.cur_objUniName]["label"], obj_label_path)


    def loadURDF(self, urdf_path, basePosition=None, baseOrientation=None, globalScaling=1.0, useFixedBase=False):
        basePosition = basePosition if basePosition is not None else [0., 0., 0.]
        baseOrientation = baseOrientation if baseOrientation is not None else p.getQuaternionFromEuler([0., 0., 0.])
        return p.loadURDF(urdf_path, basePosition=basePosition, baseOrientation=baseOrientation, 
                          globalScaling=globalScaling, useFixedBase=useFixedBase, physicsClientId=self.client_id)
    

    def keyboard_callback(self):
        events = p.getKeyboardEvents()
        for k, v in events.items():
            if v & p.KEY_WAS_TRIGGERED:
                C2GB_pos = self.Center2GripperBase_pos_rpy[0]
                C2GB_rpy = self.Center2GripperBase_pos_rpy[1]
                if k == ord('r'):
                    reset_key = input("Are you sure to reset the simulation? (y/n)")
                    if reset_key == 'y':            
                        resume_key = input("Resume the last object? (y/n)")
                        if resume_key == 'y':
                            self.reset(reset_all=False)
                            print(f"Object index keeps moving on {self.cur_obj_index}.")
                        else:
                            self.reset()
                            print("Simulation reset totally.")
                elif k == p.B3G_UP_ARROW:
                    self.place_obj2prepare_area(self.cur_objUniName)
                    cur_obj_index = self.obj_UniNames.index(self.cur_objUniName)
                    cur_obj_index = max(0, cur_obj_index - 1)
                    self.cur_objUniName = self.obj_UniNames[cur_obj_index]
                    self.prepare_cur_obj()
                    print(f"Object changed to {self.cur_obj_urdf_path}.")
                elif k == p.B3G_DOWN_ARROW:
                    self.place_obj2prepare_area(self.cur_objUniName)
                    cur_obj_index = self.obj_UniNames.index(self.cur_objUniName)
                    cur_obj_index = min(len(self.obj_UniNames)-1, cur_obj_index+1)
                    self.cur_objUniName = self.obj_UniNames[cur_obj_index]
                    self.prepare_cur_obj()
                    print(f"Object changed to {self.cur_obj_urdf_path}.")
                
                elif k == ord('q'):
                    C2GB_rpy[0] = (C2GB_rpy[0] + pi/36) % (2*pi)
                elif k == ord('e'):
                    C2GB_rpy[0] = (C2GB_rpy[0] - pi/36) % (2*pi)
                elif k == ord('a'):
                    C2GB_rpy[1] = (C2GB_rpy[1] + pi/36) % (2*pi)
                elif k == ord('d'):
                    C2GB_rpy[1] = (C2GB_rpy[1] - pi/36) % (2*pi)
                elif k == ord('z'):
                    C2GB_rpy[2] = (C2GB_rpy[2] + pi/36) % (2*pi)
                elif k == ord('c'):
                    C2GB_rpy[2] = (C2GB_rpy[2] - pi/36) % (2*pi)
                
                elif k == ord('y'):
                    C2GB_pos[0] += 0.01
                elif k == ord('i'):
                    C2GB_pos[0] -= 0.01
                elif k == ord('h'):
                    C2GB_pos[1] += 0.01
                elif k == ord('k'):
                    C2GB_pos[1] -= 0.01
                elif k == ord('n'):
                    C2GB_pos[2] += 0.01
                elif k == ord(','):
                    C2GB_pos[2] -= 0.01

                elif k == ord('s'):
                    self.save_cur_obj_grasp_pose()
                    print(f"Saved grasp pose for {self.cur_obj_urdf_path}.")

                elif k == ord('m'):
                    self.enable_collisions(self.gripper_id)
                    self.close_gripper(realtime=True)
                    pu.set_mass(self.cur_object_id, mass=self.cur_obj_mass)
                    pu.step_real(2, self.client_id)
                    self.open_gripper(realtime=False)
                    self.disable_collisions(self.gripper_id)
                    pu.set_mass(self.cur_object_id, mass=0.)
                    pu.set_pose(self.cur_object_id, (self.placement_pos_world2base, p.getQuaternionFromEuler(self.cur_rpy)))
                    print("Grasping and releasing done.")


    def start(self):
        self._init_simulator()
        self._init_misc_variables()
        self.update_objects_database()
        self.load_objects()
        self.load_gripper()
        self.reset()
        self.prepare_cur_obj()
        while True:
            self.update_gripper_pose()
            self.keyboard_callback()
            p.stepSimulation()
            time.sleep(1./240.)


    def set_obj_joints_to_lower_limit(self, obj_id):
        joints_num = pu.get_num_joints(obj_id, client_id=self.client_id)
        if joints_num > 0:
            joints_limits = np.array([pu.get_joint_limits(obj_id, joint_i, client_id=self.client_id) for joint_i in range(joints_num)])
            pu.set_joint_positions(obj_id, list(range(joints_num)), joints_limits[:, 0], client_id=self.client_id)
            pu.control_joints(obj_id, list(range(joints_num)), joints_limits[:, 0], client_id=self.client_id)
            return joints_limits


    def convert_BaseLinkPC_2_BboxCenterPC(self, pc, Base2BboxCenter):
        BboxCenter2Base = p.invertTransform(Base2BboxCenter[:3], [0., 0., 0., 1.])
        return se3_transform_pc(*BboxCenter2Base, pc)
    

    def get_obj_pc_from_id(self, obj_id, num_points=1024, use_worldpos=False):
        return pu.get_obj_pc_from_id(obj_id, num_points=num_points, use_worldpos=use_worldpos,
                                      rng=self.rng, client_id=self.client_id).astype(np.float32)
    

    def update_gripper_pose(self):
        World2ObjCenter_pose = pu.get_body_pose(self.cur_object_id, self.client_id)
        ObjCenter2GripperBase_pose = [self.Center2GripperBase_pos_rpy[0], p.getQuaternionFromEuler(self.Center2GripperBase_pos_rpy[1])]
        World2GripperBase_pose = pu.multiply_multi_transforms(World2ObjCenter_pose, ObjCenter2GripperBase_pose)
        pu.set_pose(self.gripper_id, World2GripperBase_pose)
    


if __name__ == "__main__":
    asset_root = "assets"
    group_name = "group0_dinning_table"
    obj_labeler = RobotiqGraspLabeler(asset_root, group_name=group_name)
    obj_labeler.start()