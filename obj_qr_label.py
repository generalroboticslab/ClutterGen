import pybullet as p
import pybullet_data
import json
import os
import shutil
import pybullet_utils as pu
from numpy import pi
import numpy as np
from utils import get_on_bbox, get_in_bbox
import time


VelThreshold = [0.005, np.pi]
AccThreshold = [1., np.pi]

class ObjectLabeler:
    def __init__(self, source_folder_path, overwrite=False):
        self.overwrite = overwrite
        self.source_folder_path = source_folder_path
        assert os.path.isdir(self.source_folder_path), f"Source Folder path '{self.source_folder_path}' does not exist."
        self.target_folder_path = os.path.join(os.path.dirname(self.source_folder_path), 'selected_obj')
        if self.overwrite:
            self.target_folder_path = self.source_folder_path
        os.makedirs(self.target_folder_path, exist_ok=True)


    # Start PyBullet in GUI mode.
    def _init_simulator(self):
        self.id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(enableFileCaching=0)


    def _init_buttons(self):
        p.removeAllUserParameters()
        # Create GUI elements.
        self.on_button_param = p.addUserDebugParameter("ON", 1, 0, 1)
        self.in_button_param = p.addUserDebugParameter("IN", 1, 0, 1)
        self.None_button_param = p.addUserDebugParameter("None", 1, 0, 1)
        self.save_button_param = p.addUserDebugParameter("Save", 1, 0, 1)
        self.remove_button_param = p.addUserDebugParameter("Remove", 1, 0, 1)
        self.global_scaling_param = p.addUserDebugParameter("Global Scaling", 0, 1, 0.5)
        # self.object_stage_param = p.addUserDebugParameter("Object Stage", 1, 10, 1)
        self.x_rot_param = p.addUserDebugParameter("X Rotation", 1, 0, 1)
        self.y_rot_param = p.addUserDebugParameter("Y Rotation", 1, 0, 1)
        self.z_rot_param = p.addUserDebugParameter("Z Rotation", 1, 0, 1)


    def reset_sim(self, reset_all=True):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF("plane.urdf")
        self.cur_object_id = None
        pu.change_obj_color(self.plane_id, rgba_color=[1., 1., 1., 0.2])
        if reset_all:
            self._read_source_folder_path()
            self._init_buttons()
            self._reset_buffer()
        self.load_cur_obj_and_pc()


    def _reset_buffer(self):
        self.on_button_read = 1
        self.in_button_read = 1
        self.None_button_read = 1
        self.save_button_read = 1
        self.remove_button_read = 1
        self.reset_button_read = 1
        self.x_rot_read = 1
        self.y_rot_read = 1
        self.z_rot_read = 1
        self.old_gb_scaling = None


    def _read_source_folder_path(self):
        self.category_lst = os.listdir(self.source_folder_path)
        self.category_lst.sort()
        self.max_cate_index = len(self.category_lst) - 1
        self.cur_cate_index = 0
        self.cur_obj_index = 0
        self.cur_category = self.category_lst[self.cur_cate_index]
        self.update_cur_cate_path()
        self.update_cur_obj_path()


    def _reset_meta_data(self):
        self.meta_data = {
            'queried_region': None,
            'globalScaling': 1.,
            "ori_rpy": [0, 0, 0.],
        }

    
    def update_cur_cate_path(self):
        self.cur_cate_path = os.path.join(self.source_folder_path, self.cur_category)
        self.cur_obj_lst = os.listdir(self.cur_cate_path)
        self.cur_obj_lst.sort()
        self.max_obj_index = len(self.cur_obj_lst) - 1
        self.cur_obj_index = 0


    def update_cur_obj_path(self):
        self.cur_obj_folder = os.path.join(self.cur_cate_path, self.cur_obj_lst[self.cur_obj_index])
        self.cur_obj_urdf_path = os.path.join(self.cur_obj_folder, 'mobility.urdf')
        self.cur_rpy = [0, 0, 0.]
        if os.path.isfile(os.path.join(self.cur_obj_folder, 'label.json')):
            with open(os.path.join(self.cur_obj_folder, 'label.json'), 'r') as f:
                self.meta_data = json.load(f)
                print(self.meta_data['ori_rpy'])
        else:
            self._reset_meta_data()

        
    def load_cur_obj_and_pc(self):
        try:
            new_object_id = p.loadURDF(self.cur_obj_urdf_path, basePosition=[0, 0, 0.], globalScaling=self.meta_data["globalScaling"])
            if hasattr(self, "cur_object_id") and self.cur_object_id is not None: p.removeBody(self.cur_object_id)
            self.cur_object_id = new_object_id
            self.cur_obj_pc = pu.get_obj_pc_from_id(self.cur_object_id)
            self.cur_obj_bbox = pu.get_obj_axes_aligned_bbox_from_pc(self.cur_obj_pc)
            
            p.changeDynamics(self.cur_object_id, -1, mass=0.)
            obj_joints_num = pu.get_num_joints(self.cur_object_id)
            if obj_joints_num > 0: # If the object is not movable, we need to control its joints to make it movable below each reload urdf!
                joints_limits = np.array([pu.get_joint_limits(self.cur_object_id, joint_i) for joint_i in range(obj_joints_num)])
                pu.set_joint_positions(self.cur_object_id, list(range(obj_joints_num)), joints_limits[:, 0])
                pu.control_joints(self.cur_object_id, list(range(obj_joints_num)), joints_limits[:, 0])

            print(f"Current category: {self.category_lst[self.cur_cate_index]}; Current cur_object_id: {self.cur_obj_index}")
        except Exception as e:
            print(f"Failed to load {self.cur_obj_urdf_path}. Error: {e}")


    def save_label(self):
        self.meta_data['ori_rpy'] = [(self.meta_data['ori_rpy'][i]+self.cur_rpy[i])%(2*pi) for i in range(len(self.cur_rpy))]
        # Change the first link's rotation in urdf file; we can not change partNet's urdf file directly.
        pu.modify_specific_link_in_urdf(self.cur_obj_urdf_path, self.meta_data['ori_rpy'], firstlink=True)
        obj_source_folder_path = os.path.dirname(self.cur_obj_urdf_path)
        json_file_path = os.path.join(obj_source_folder_path, 'label.json')
        with open(json_file_path, 'w') as f:
            json.dump(self.meta_data, f, indent=4)
            print(f"Labeling saved to {json_file_path}.")

        if not self.overwrite:
            obj_target_folder_path = os.path.join(self.target_folder_path, self.cur_category, os.path.basename(obj_source_folder_path))
            if os.path.exists(obj_target_folder_path):
                shutil.rmtree(obj_target_folder_path)
            shutil.copytree(obj_source_folder_path, obj_target_folder_path)
            print(f"{obj_source_folder_path} copied to {obj_target_folder_path}.")


    def button_callback(self):
        on_button_read = p.readUserDebugParameter(self.on_button_param)
        in_button_read = p.readUserDebugParameter(self.in_button_param)
        None_button_read = p.readUserDebugParameter(self.None_button_param)
        save_button_read = p.readUserDebugParameter(self.save_button_param)
        remove_button_read = p.readUserDebugParameter(self.remove_button_param)
        x_rot_read = p.readUserDebugParameter(self.x_rot_param)
        y_rot_read = p.readUserDebugParameter(self.y_rot_param)
        z_rot_read = p.readUserDebugParameter(self.z_rot_param)
        global_scaling_read = p.readUserDebugParameter(self.global_scaling_param)
        if self.old_gb_scaling != global_scaling_read:
            self.old_gb_scaling = global_scaling_read
        #     self.meta_data['globalScaling'] = global_scaling_read
        #     self.load_cur_obj_and_pc()
            print(f"Object Size changed to X: {self.cur_obj_bbox[7]*2*100}cm, Y: {self.cur_obj_bbox[8]*2*100}cm, Z: {self.cur_obj_bbox[9]*2*100}cm.")

        
        if self.on_button_read != on_button_read:
            self.on_button_read = on_button_read
            self.meta_data['queried_region'] = 'on'
            self.cur_obj_pc = pu.get_obj_pc_from_id(self.cur_object_id)
            self.cur_obj_bbox = pu.get_obj_axes_aligned_bbox_from_pc(self.cur_obj_pc)
            queried_region = get_on_bbox(self.cur_obj_bbox, 0.1)
            if hasattr(self, 'queried_region_id') and self.queried_region_id is not None: p.removeBody(self.queried_region_id)
            self.queried_region_id = pu.draw_box_body(queried_region[:3], queried_region[3:7], queried_region[7:], rgba_color=[1, 0, 0, 0.5])
            print("The object is labeled as 'on'.")
        
        if self.in_button_read != in_button_read:
            self.in_button_read = in_button_read
            self.meta_data['queried_region'] = 'in'
            self.cur_obj_pc = pu.get_obj_pc_from_id(self.cur_object_id)
            self.cur_obj_bbox = pu.get_obj_axes_aligned_bbox_from_pc(self.cur_obj_pc)
            queried_region = get_in_bbox(self.cur_obj_bbox)
            if hasattr(self, 'queried_region_id') and self.queried_region_id is not None: p.removeBody(self.queried_region_id)
            self.queried_region_id = pu.draw_box_body(queried_region[:3], queried_region[3:7], queried_region[7:], rgba_color=[1, 0, 0, 0.5])
            print("The object is labeled as 'in'.")
        
        if self.None_button_read != None_button_read:
            self.None_button_read = None_button_read
            self.meta_data['queried_region'] = 'None'
            if hasattr(self, 'queried_region_id') and self.queried_region_id is not None: p.removeBody(self.queried_region_id)
            self.queried_region_id = None
            print("The object is labeled as 'None'.")
        
        if self.save_button_read != save_button_read:
            self.save_button_read = save_button_read
            self.save_label()

        if self.remove_button_read != remove_button_read:
            self.remove_button_read = remove_button_read
            obj_source_folder_path = os.path.dirname(self.cur_obj_urdf_path)
            obj_target_folder_path = os.path.join(self.target_folder_path, self.cur_category, os.path.basename(obj_source_folder_path))
            if os.path.exists(obj_target_folder_path):
                shutil.rmtree(obj_target_folder_path)
                print(f"{obj_target_folder_path} removed.")
                if len(os.listdir(self.cur_cate_path)) == 0:
                    shutil.rmtree(self.cur_cate_path)
                    print(f"{self.cur_cate_path} removed.")
            else:
                print(f"{obj_target_folder_path} does not exist.")
        
        if self.x_rot_read != x_rot_read:
            self.cur_rpy[0] = (self.cur_rpy[0] + pi/2) % (2*pi)
            self.x_rot_read = x_rot_read
            print(f"X rotation changed to {self.cur_rpy[0]}.")
        if self.y_rot_read != y_rot_read:
            self.cur_rpy[1] = (self.cur_rpy[1] + pi/2) % (2*pi)
            self.y_rot_read = y_rot_read
            print(f"Y rotation changed to {self.cur_rpy[1]}.")
        if self.z_rot_read != z_rot_read:
            self.cur_rpy[2] = (self.cur_rpy[2] + pi/2) % (2*pi)
            self.z_rot_read = z_rot_read
            print(f"Z rotation changed to {self.cur_rpy[2]}.")
        
        try:
            obj_pose = pu.get_body_pose(self.cur_object_id)
            pu.set_pose(self.cur_object_id, (obj_pose[0], p.getQuaternionFromEuler(self.cur_rpy)))
        except:
            print("Failed to set object pose.")


    def keyboard_callback(self):
        events = p.getKeyboardEvents()
        for k, v in events.items():
            if v & p.KEY_WAS_TRIGGERED:
                if k == ord('r'):
                    reset_key = input("Are you sure to reset the simulation? (y/n)")
                    if reset_key == 'y':            
                        resume_key = input("Resume the last object? (y/n)")
                        if resume_key == 'y':
                            self.reset_sim(reset_all=False)
                            print(f"Object index keeps moving on {self.cur_obj_index}.")
                        else:
                            self.reset_sim()
                            print("Simulation reset totally.")
                elif k == p.B3G_UP_ARROW:
                    self.cur_obj_index = max(0, self.cur_obj_index - 1)
                    self.update_cur_obj_path()
                    self.load_cur_obj_and_pc()
                    self.old_gb_scaling = None
                    print(f"Object changed to {self.cur_obj_urdf_path}.")
                elif k == p.B3G_DOWN_ARROW:
                    self.cur_obj_index = min(self.max_obj_index, self.cur_obj_index + 1)
                    self.update_cur_obj_path()
                    self.load_cur_obj_and_pc()
                    self.old_gb_scaling = None
                    print(f"Object changed to {self.cur_obj_urdf_path}.")
                elif k == p.B3G_LEFT_ARROW:
                    self.cur_cate_index = max(0, self.cur_cate_index - 1)
                    self.cur_category = self.category_lst[self.cur_cate_index]
                    self.update_cur_cate_path()
                    self.update_cur_obj_path()
                    self.load_cur_obj_and_pc()
                    self.old_gb_scaling = None
                    print(f"Object changed to {self.cur_obj_urdf_path}.")
                elif k == p.B3G_RIGHT_ARROW:
                    self.cur_cate_index = min(self.max_cate_index, self.cur_cate_index + 1)
                    self.cur_category = self.category_lst[self.cur_cate_index]
                    self.update_cur_cate_path()
                    self.update_cur_obj_path()
                    self.load_cur_obj_and_pc()
                    self.old_gb_scaling = None
                    print(f"Object changed to {self.cur_obj_urdf_path}.")
                
                elif k == ord('q'):
                    self.cur_rpy[0] = (self.cur_rpy[0] + pi/180) % (2*pi)
                    print(f"Current Rotation is X: {self.cur_rpy[0]}, Y: {self.cur_rpy[1]}, Z: {self.cur_rpy[2]}.")
                elif k == ord('e'):
                    self.cur_rpy[0] = (self.cur_rpy[0] - pi/180) % (2*pi)
                    print(f"Current Rotation is X: {self.cur_rpy[0]}, Y: {self.cur_rpy[1]}, Z: {self.cur_rpy[2]}.")
                elif k == ord('a'):
                    self.cur_rpy[1] = (self.cur_rpy[1] + pi/180) % (2*pi)
                    print(f"Current Rotation is X: {self.cur_rpy[0]}, Y: {self.cur_rpy[1]}, Z: {self.cur_rpy[2]}.")
                elif k == ord('d'):
                    self.cur_rpy[1] = (self.cur_rpy[1] - pi/180) % (2*pi)
                    print(f"Current Rotation is X: {self.cur_rpy[0]}, Y: {self.cur_rpy[1]}, Z: {self.cur_rpy[2]}.")
                elif k == ord('z'):
                    self.cur_rpy[2] = (self.cur_rpy[2] + pi/180) % (2*pi)
                    print(f"Current Rotation is X: {self.cur_rpy[0]}, Y: {self.cur_rpy[1]}, Z: {self.cur_rpy[2]}.")
                elif k == ord('c'):
                    self.cur_rpy[2] = (self.cur_rpy[2] - pi/180) % (2*pi)
                    print(f"Current Rotation is X: {self.cur_rpy[0]}, Y: {self.cur_rpy[1]}, Z: {self.cur_rpy[2]}.")
                
                elif k == ord('m'):
                    self.placement_pos_world2center = [0, 0, self.cur_obj_bbox[9]]
                    self.placement_pos_world2base = p.multiplyTransforms(self.placement_pos_world2center, [0, 0, 0, 1.], -self.cur_obj_bbox[:3], [0, 0, 0, 1.])[0]
                    pu.set_pose(self.cur_object_id, (self.placement_pos_world2base, p.getQuaternionFromEuler(self.cur_rpy)))
                    p.changeDynamics(self.cur_object_id, -1, mass=0.1)
                    self.prev_obj_vel = pu.getObjVelocity(self.cur_object_id, to_array=True)
                    first_stable = False
                    for i in range(240):
                        p.stepSimulation()
                        obj_vel = pu.getObjVelocity(self.cur_object_id, to_array=True)
                        if (self.vel_checker(obj_vel) and self.acc_checker(self.prev_obj_vel, obj_vel)):
                            if not first_stable:  
                                first_stable = True
                                print(f"Stable Steps: {i}.")
                            # break
                        time.sleep(1./240.)
                    p.changeDynamics(self.cur_object_id, -1, mass=0.)
                    pu.set_pose(self.cur_object_id, ([0, 0, 0.], p.getQuaternionFromEuler(self.cur_rpy)))
                
                elif k == ord('k'):
                    global_scaling_read = input("Please input the global scaling factor: ")
                    try:
                        global_scaling_read = float(global_scaling_read)
                        assert global_scaling_read > 0
                        self.meta_data["globalScaling"] = global_scaling_read
                        self.load_cur_obj_and_pc()
                        print(f"Object Size changed to X: {self.cur_obj_bbox[7]*2*100}cm, Y: {self.cur_obj_bbox[8]*2*100}cm, Z: {self.cur_obj_bbox[9]*2*100}cm.")
                    except:
                        print(f"Invalid input: {global_scaling_read}.")
                
                elif k == ord(','):
                    obj_index = input(f"Please input the object index (Max {self.max_obj_index}): ")
                    try:
                        obj_index = int(obj_index)
                        if obj_index >= 0 and obj_index <= self.max_obj_index:
                            self.cur_obj_index = obj_index
                            self.update_cur_obj_path()
                            self.load_cur_obj_and_pc()
                            self.old_gb_scaling = None
                            print(f"Object changed to {self.cur_obj_urdf_path}.")
                        else:
                            print(f"Invalid input: {obj_index}.")
                    except:
                        print(f"Invalid input: {obj_index}.")
                
                elif k == ord('.'):
                    cate_index = input(f"Please input the category index (Max {self.max_cate_index}): ")
                    try:
                        cate_index = int(cate_index)
                        if cate_index >= 0 and cate_index <= self.max_cate_index:
                            self.cur_cate_index = cate_index
                            self.cur_category = self.category_lst[self.cur_cate_index]
                            self.update_cur_obj_path()
                            self.load_cur_obj_and_pc()
                            self.old_gb_scaling = None
                            print(f"Object changed to {self.cur_obj_urdf_path}.")
                        else:
                            print(f"Invalid input: {cate_index}.")
                    except:
                        print(f"Invalid input: {cate_index}.")

    
    # Utils
    def vel_checker(self, obj_vel):
        # print(f"Vel Checker: {obj_vel[:3].__abs__() < VelThreshold[0]}, {obj_vel[3:].__abs__() < VelThreshold[1]}")
        # print(f"{obj_vel[3:].__abs__()}")
        return ((obj_vel[:3].__abs__() < VelThreshold[0]).all() \
                    and (obj_vel[3:].__abs__() < VelThreshold[1]).all())


    def acc_checker(self, obj_vel, prev_obj_vel):
        acc = np.abs((obj_vel - prev_obj_vel) / (1/240))
        # print(f"Acc Checker: {acc[:3].__abs__() < AccThreshold[0]}, {acc[3:].__abs__() < AccThreshold[1]}")
        return (acc[:3] < AccThreshold[0]).all() \
                    and (acc[3:] < AccThreshold[1]).all()

    
    
    def start(self):
        self._init_simulator()
        self.reset_sim()
        while True:
            self.keyboard_callback()
            self.button_callback()
            # pc = pu.get_obj_pc_from_id(self.cur_object_id)
            # pu.visualize_pc(pc)
        

if __name__ == '__main__':
    source_folder_path = "assets/PartNet/partNet_raw"

    labeler = ObjectLabeler(source_folder_path, overwrite=False)
    labeler.start()