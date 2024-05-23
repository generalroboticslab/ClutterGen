import sys
import os
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the this package path to sys.path
sys.path.append(current_dir)

from socket_client import SocketClient


class FrankaPandaCtrl:
    def __init__(self):
        self.client = SocketClient()
        self.client.send_message(["START", None])
        self.num_joints = 7
        self.HOME = [0.054289627618698016, -0.24698193331114637, -0.1353490755215156, -1.4598274921017433, -0.10732393256213577, 1.2574428409277323, 0.6534634736124315]

        """
        All message from client to server type is [command, data]. Length of the message is 2.
        It seems that we can not send multiple data in one message easily.
        """
        
        
    def reset(self):
        self.home_gripper()
        self.home_arm()


    def set_joint_state(self, joint_state):
        assert len(joint_state) == self.num_joints
        command_joint_state = ["set_joint_state"] + [joint_state]
        self.client.send_message(command_joint_state)
        # message = self.client.receive_message()

    
    def set_eef_pose(self, pose):
        """
        pose: [x, y, z, qx, qy, qz, qw]
        """
        assert len(pose) == 7
        command_pose = ["set_eef_pose"] + [pose]
        self.client.send_message(command_pose)
        # message = self.client.receive_message()

    
    def plan_eef_cartesian_path(self, waypoints):
        """
        waypoints: [[x, y, z, qx, qy, qz, qw], ...]
        """
        for waypoint in waypoints:
            assert len(waypoint) == 7, f"waypoint should have length 7, but got {len(waypoint)}."
        command_waypoints = ["plan_eef_cartesian_path"] + [waypoints]
        self.client.send_message(command_waypoints)
        path_len = self.client.receive_message()
        path_len = int(eval(path_len))
        return path_len
    

    def plan_exec_eef_cartesian_path(self, waypoints):
        """
        No return value or message back. This is for the short path planning and execution.
        """
        for waypoint in waypoints:
            assert len(waypoint) == 7, f"waypoint should have length 7, but got {len(waypoint)}."
        command_waypoints = ["plan_exec_eef_cartesian_path"] + [waypoints]
        self.client.send_message(command_waypoints)
    
    
    def plan_eef_cartesian_path_compare(self, multi_waypoints):
        """
        waypoints: [[x, y, z, qx, qy, qz, qw], ...]
        """
        for waypoints in multi_waypoints:
            for waypoint in waypoints:
                assert len(waypoint) == 7, f"waypoint should have length 7, but got {len(waypoint)}."
        command_multi_waypoints = ["plan_eef_cartesian_path_compare"] + [multi_waypoints]
        self.client.send_message(command_multi_waypoints)
        lst_msg = self.client.receive_message()
        shortest_index, path_len = eval(lst_msg)
        if shortest_index is None:
            return None, 0
        return int(shortest_index), int(path_len)
    

    def execute_plan(self):
        command_execute = ["execute_plan", None]
        self.client.send_message(command_execute)

    
    def check_plan_loaded(self):
        command_check = ["check_plan_loaded", None]
        self.client.send_message(command_check)
        message = self.client.receive_message()
        return message


    def clean_plan(self):
        command_clean = ["clean_plan", None]
        self.client.send_message(command_clean)
        # message = self.client.receive_message()


    def small_lift(self, cur_eef_pose=None, height=0.02):
        """
        cur_eef_pose: [x, y, z, qx, qy, qz, qw]
        """
        cur_eef_pose = self.get_eef_pose() if cur_eef_pose is None else cur_eef_pose
        cur_eef_pose[2] += height
        return self.plan_exec_eef_cartesian_path([cur_eef_pose])

    
    def open_gripper(self):
        command_open_gripper = ["open_gripper", None]
        self.client.send_message(command_open_gripper)
        # message = self.client.receive_message()


    def close_gripper(self):
        command_close_gripper = ["close_gripper", None]
        self.client.send_message(command_close_gripper)
        # message = self.client.receive_message()


    def home_gripper(self):
        command_home_gripper = ["home_gripper", None]
        self.client.send_message(command_home_gripper)
        # message = self.client.receive_message()


    def home_arm(self):
        command_home_arm = ["home_arm", self.HOME]
        self.client.send_message(command_home_arm)
        # message = self.client.receive_message()


    def stop(self):
        command_stop = ["STOP", None]
        self.client.send_message(command_stop)

    """
    Get functions
    """
    def get_eef_pose(self):
        command_get_eef_pose = ["get_eef_pose", None]
        self.client.send_message(command_get_eef_pose)
        eef_pose_str = self.client.receive_message()
        eef_pose_raw = eval(eef_pose_str)
        return eef_pose_raw
    

    def get_joint_state(self):
        command_get_joint_state = ["get_joint_state", None]
        self.client.send_message(command_get_joint_state)
        joint_state_str = self.client.receive_message()
        joint_state_raw = eval(joint_state_str)
        return joint_state_raw
    

    """
    Utils functions
    """
    def disconnect(self):
        self.client.disconnect()
    

if __name__=="__main__":
    # Panda eef to hand <origin rpy="0 0 -0.785398163397" xyz="0 0 0"/>
    franka_panda_ctrl = FrankaPandaCtrl()
    franka_panda_ctrl.reset()
    # cur_eef_pose = franka_panda_ctrl.get_eef_pose()
    # cur_joint_state = franka_panda_ctrl.get_joint_state()
    # franka_panda_ctrl.home_gripper()
    # franka_panda_ctrl.close_gripper()
    # cur_eef_pose = franka_panda_ctrl.get_eef_pose()
    # franka_panda_ctrl.home_arm()
    # path_len = franka_panda_ctrl.plan_eef_cartesian_path([cur_eef_pose])
    # franka_panda_ctrl.execute_plan()
    # franka_panda_ctrl.clean_plan()