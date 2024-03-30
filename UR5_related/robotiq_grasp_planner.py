import pybullet as p
import pybullet_utils_cust as pu


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