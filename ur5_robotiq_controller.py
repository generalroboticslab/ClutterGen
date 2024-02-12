### Ur5 Setup
import os, math, time
from collections import namedtuple
import pybullet as p
import numpy as np
import pybullet_utils as pu
import pybullet_data


def normalize_angle_positive(angle):
    two_pi = 2.0 * np.pi
    return ((angle % two_pi) + two_pi) % two_pi


def normalize_angle(angle):
    angle_normalized = normalize_angle_positive(angle)
    if angle_normalized > np.pi:
        angle_normalized -= 2 * np.pi
    return angle_normalized


def load_ur_robotiq_robot(robot_initial_pose=[[0., 0., 0.], [0., 0., 0., 1.]], 
                          client_id=0):
    # load robot
    urdf_dir = 'UR5_related/ur5_robotiq_description/urdf'
    urdf_filepath = os.path.join(urdf_dir, 'ur5_robotiq.urdf')
    xacro_filepath = os.path.join(urdf_dir, 'ur5_robotiq_robot.xacro')
    if not os.path.exists(urdf_filepath): # generate urdf
        raise ValueError(f"URDF file {urdf_filepath} is not exist!")
        cmd = 'rosrun xacro xacro --inorder {} -o {}'.format(xacro_filepath, urdf_filepath)
        os.system(cmd)
        robotiq_description_dir = '../ur5_robotiq_meta_package/robotiq_2finger_grippers/robotiq_2f_85_gripper_visualization'
        sed_cmd = "sed -i 's|{}|{}|g' {}".format('package://robotiq_2f_85_gripper_visualization',
                                                 robotiq_description_dir, urdf_filepath)
        os.system(sed_cmd)
        ur5_description_dir = '../ur5_robotiq_meta_package/universal_robot/ur_description'
        sed_cmd = "sed -i 's|{}|{}|g' {}".format('package://ur_description', ur5_description_dir, urdf_filepath)
        os.system(sed_cmd)
        # adjust the gripper effort for stable grasping in pybullet
        sed_cmd = "sed -i 's|{}|{}|g' {}".format('limit effort="1000"', 'limit effort="200"', urdf_filepath)
        os.system(sed_cmd)

    robot_id = p.loadURDF(urdf_filepath, 
                          basePosition=robot_initial_pose[0], 
                          baseOrientation=robot_initial_pose[1],
                          flags=p.URDF_USE_SELF_COLLISION, 
                          physicsClientId=client_id)
    return robot_id, urdf_filepath


Motion = namedtuple('Motion', ['position_trajectory', 'time_trajectory', 'velocity_trajectory'])


class UR5RobotiqPybulletController(object):
    JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                         'qIndex', 'uIndex', 'flags',
                                         'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                         'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                         'parentFramePos', 'parentFrameOrn', 'parentIndex'])

    JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                           'jointReactionForces', 'appliedJointMotorTorque'])

    LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                         'localInertialFramePosition', 'localInertialFrameOrientation',
                                         'worldLinkFramePosition', 'worldLinkFrameOrientation'])

    # movable joints for each moveit group
    GROUPS = {
        'arm': ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
        'gripper': ['finger_joint', 'left_inner_knuckle_joint', 'left_inner_finger_joint', 'right_outer_knuckle_joint',
                    'right_inner_knuckle_joint', 'right_inner_finger_joint']
    }
    HOME = [0, -0.8227210029571718, -0.130, -0.660, 0, 1.62]
    HOME = [0, -1.15, 0.9, -0.660, 0, 0.0]
    HOME = [0, -1.5, 0.23, -0.660, 0, 0.0]
    OPEN_POSITION = [0] * 6
    CLOSED_POSITION = 0.72 * np.array([1, 1, -1, 1, 1, -1])

    JOINT_INDICES_DICT = {}
    EE_LINK_NAME = 'ee_link'

    TIP_LINK = "ee_link"
    BASE_LINK = "base_link"
    ARM = "manipulator"
    GRIPPER = "gripper"
    ARM_JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                       'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    GRIPPER_JOINT_NAMES = ['finger_joint']

    # this is read from moveit_configs joint_limits.yaml
    MOVEIT_ARM_MAX_VELOCITY = [3.14, 3.14, 3.14, 3.14, 3.14, 3.14]


    def __init__(self, robot_id, rng=None, client_id=0):
        self.id = robot_id
        self.rng = rng # RRT planner random
        self.pose_rng_seed = 234
        self.initial_pose_rng = np.random.default_rng(self.pose_rng_seed)
        self.client_id = client_id
        self.initial_joint_values = self.HOME
        self.num_arm_joints = len(self.GROUPS["arm"])
        self.num_gripper_joints = len(self.GROUPS["gripper"])
        joint_infos = [p.getJointInfo(robot_id, joint_index, physicsClientId=self.client_id) \
                       for joint_index in range(p.getNumJoints(robot_id, physicsClientId=self.client_id))]
        self.base_link_initial_pose = pu.get_link_pose(self.id, -1, client_id=self.client_id)

        self.JOINT_INDICES_DICT = {entry[1].decode('ascii'): entry[0] for entry in joint_infos}
        self.GROUP_INDEX = {key: [self.JOINT_INDICES_DICT[joint_name] for joint_name in self.GROUPS[key]] \
                            for key in self.GROUPS}
        self.EEF_LINK_INDEX = pu.link_from_name(robot_id, self.EE_LINK_NAME, client_id=self.client_id)

        self.arm_max_joint_velocities = [pu.get_max_velocity(self.id, j_id, client_id=self.client_id) for j_id in self.GROUP_INDEX['arm']] # not use?
        self.attach_cid = None
        self.attach_object_id = None

        # visualize
        self.initial_visualize = False

        self.reset(initial=True)
        self.reset_inner_rng() # above reset func would use the rng so we need to reset it back.


    def reset(self, input_joint_values=None, initial=False):
        if input_joint_values is not None:
            self.start_joint_values = input_joint_values
        else:
            if initial: self.start_joint_values = self.initial_joint_values
            else: self.start_joint_values = self.random_initial(box_ext=0.15)
        self.set_arm_joints(self.start_joint_values)
        self.set_gripper_joints(self.OPEN_POSITION)
        self.arm_discretized_plan = None
        self.gripper_discretized_plan = None
        self.arm_wp_target_index = 0
        self.gripper_wp_target_index = 0
        self.detach()
        return self.start_joint_values


    def attach_object(self, target_id):
        target_pose = pu.get_body_pose(target_id, client_id=self.client_id)
        eef_pose = self.get_eef_pose()
        eef_P_world = p.invertTransform(eef_pose[0], eef_pose[1])
        eef_P_target = p.multiplyTransforms(
            eef_P_world[0], eef_P_world[1], 
            target_pose[0], target_pose[1],
        )
        self.attach_cid = p.createConstraint(
            parentBodyUniqueId=target_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.id,
            childLinkIndex=self.EEF_LINK_INDEX,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=eef_P_target[0],
            childFrameOrientation=eef_P_target[1],
            physicsClientId=self.client_id)
        self.attach_object_id = target_id


    def detach(self):
        if self.attach_cid is not None:
            p.removeConstraint(self.attach_cid, physicsClientId=self.client_id)
            self.attach_cid = None
            self.attach_object_id = None


    def update_arm_motion_plan(self, arm_discretized_plan):
        self.arm_discretized_plan = arm_discretized_plan
        self.arm_wp_target_index = 1


    def update_gripper_motion_plan(self, gripper_discretized_plan):
        self.gripper_discretized_plan = gripper_discretized_plan
        self.gripper_wp_target_index = 1


    def set_arm_joints(self, joint_values):
        pu.set_joint_positions(self.id, self.GROUP_INDEX['arm'], joint_values, client_id=self.client_id)
        pu.control_joints(self.id, self.GROUP_INDEX['arm'], joint_values, client_id=self.client_id)


    def control_arm_joints(self, joint_values, control_type='hard', client_id=0):
        pu.control_joints(self.id, self.GROUP_INDEX['arm'], joint_values, control_type, client_id=self.client_id)


    def set_gripper_joints(self, joint_values):
        pu.set_joint_positions(self.id, self.GROUP_INDEX['gripper'], joint_values, client_id=self.client_id)
        pu.control_joints(self.id, self.GROUP_INDEX['gripper'], joint_values, client_id=self.client_id)


    def control_gripper_joints(self, joint_values, control_type='limited'):
        pu.control_joints(self.id, self.GROUP_INDEX['gripper'], joint_values, control_type, client_id=self.client_id)


    def close_gripper(self, realtime=False):
        waypoints = self.plan_gripper_joint_values(self.CLOSED_POSITION)
        self.execute_gripper_plan(waypoints, realtime)


    def plan_gripper_joint_values(self, goal_joint_values, start_joint_values=None, duration=None):
        if start_joint_values is None:
            start_joint_values = self.get_gripper_joint_values()
        num_steps = 240 if duration is None else int(duration*240)
        discretized_plan = np.linspace(start_joint_values, goal_joint_values, num_steps)
        return discretized_plan
    

    def get_jacobian_pybullet(self, arm_joint_values):

        gripper_joint_values = self.get_gripper_joint_values()
        current_positions = arm_joint_values + gripper_joint_values

        zero_vec = [0.0] * len(current_positions)
        jac_t, jac_r = p.calculateJacobian(self.id, 
                                           self.EEF_LINK_INDEX, (0, 0, 0),
                                           current_positions, 
                                           zero_vec, 
                                           zero_vec, 
                                           physicsClientId=self.client_id)
        jacobian = np.concatenate((np.array(jac_t)[:, :6], np.array(jac_r)[:, :6]), axis=0)
        return jacobian


    def get_arm_fk_pybullet(self, joint_values): # pybullet need to set joint then set back which pollutes visualization
        return pu.forward_kinematics(self.id, 
                                     self.GROUP_INDEX['arm'], 
                                     joint_values, 
                                     self.EEF_LINK_INDEX, 
                                     client_id=self.client_id)


    def get_arm_fk_ikfast(self, joint_values): # fk_ikfast position is right but orientation is wrong -> need further fix
        ee_pose = self.ur5_kin.forward(joint_values)
        ee_pose = np.asarray(ee_pose).reshape(3, 4)  # 3x4 rigid transformation matrix
        ee_matrix = np.concatenate([ee_pose, [[0, 0, 0, 1]]], axis=0)
        ee_orientation = self.quaternion_from_matrix(ee_matrix)
        ee_position = ee_pose[:, 3].tolist()
        ee_pose = (ee_position, ee_orientation)
        base_correction = (self.base_link_initial_pose[0], [0.0, 0.0, 1.0, 0.0]) # why correction?
        ee_2_wrist3 = ([0.0, 0.0, 0.0], [-0.5, 0.5, -0.5, 0.5])  # also could use pybullet ee_2_wrist3 = p.multiplyTransform(p.invertTransform(wrist3), ee_pose)
        ee_pose_adjust = pu.multiply_multi_transforms(base_correction, ee_pose, p.invertTransform(*ee_2_wrist3))
        return ee_pose_adjust
    

    def quaternion_from_matrix(self, matrix, isprecise=True):
        """Return quaternion from rotation matrix.
          If isprecise is True, the input matrix is assumed to be a precise rotation
          matrix and a faster algorithm is used.
          """
        M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
        if isprecise:
            q = np.empty((4,))
            t = np.trace(M)
            if t > M[3, 3]:
                q[0] = t
                q[3] = M[1, 0] - M[0, 1]
                q[2] = M[0, 2] - M[2, 0]
                q[1] = M[2, 1] - M[1, 2]
            else:
                i, j, k = 0, 1, 2
                if M[1, 1] > M[0, 0]:
                    i, j, k = 1, 2, 0
                if M[2, 2] > M[i, i]:
                    i, j, k = 2, 0, 1
                t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
                q[i] = t
                q[j] = M[i, j] + M[j, i]
                q[k] = M[k, i] + M[i, k]
                q[3] = M[k, j] - M[j, k]
                q = q[[3, 0, 1, 2]]
            q *= 0.5 / math.sqrt(t * M[3, 3])
        else:
            m00 = M[0, 0]
            m01 = M[0, 1]
            m02 = M[0, 2]
            m10 = M[1, 0]
            m11 = M[1, 1]
            m12 = M[1, 2]
            m20 = M[2, 0]
            m21 = M[2, 1]
            m22 = M[2, 2]
            # symmetric matrix K
            K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0], 
                          [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                          [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                          [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
            K /= 3.0
            # quaternion is eigenvector of K that corresponds to largest eigenvalue
            w, V = np.linalg.eigh(K)
            q = V[[3, 0, 1, 2], np.argmax(w)]
        if q[0] < 0.0:
            np.negative(q, q)
        return q


    def get_arm_ik(self, pose_2d, avoid_collisions=False, solver='ikpybullet'): # default use ikfast, lift use pybullet
        if solver == 'ikpybullet':
            return self.get_arm_ik_pybullet(pose_2d, avoid_collisions=avoid_collisions)
        else: return self.get_ik_fast(pose_2d, avoid_collisions=avoid_collisions)


    def get_arm_ik_pybullet(self, pose_2d, arm_joint_values=None, gripper_joint_values=None, avoid_collisions=True):
        gripper_joint_values = self.get_gripper_joint_values()
        arm_joint_values = self.get_arm_joint_values()

        joint_values = p.calculateInverseKinematics(self.id,
                                                    self.EEF_LINK_INDEX,  # self.JOINT_INDICES_DICT[self.EEF_LINK],
                                                    pose_2d[0],
                                                    pose_2d[1],
                                                    currentPositions=list(arm_joint_values) + gripper_joint_values,
                                                    physicsClientId=self.client_id
                                                    # maxNumIterations=100,
                                                    # residualThreshold=.01
                                                    )
        ik_result = list(joint_values[:6])
        # handle joint limit violations. 
        # TODO: confirm that this logic makes sense

        # Clamp joint values to be within joint limits (only for prismatic joints it seems)
        # for i in range(len(self.GROUP_INDEX['arm'])):
        #     if pu.violates_limit(self.id, self.GROUP_INDEX['arm'][i], ik_result[i], client_id=self.client_id):
        #         lower, upper = pu.get_joint_limits(self.id, self.GROUP_INDEX['arm'][i], client_id=self.client_id)
        #         if ik_result[i] < lower and ik_result[i] + 2*np.pi > upper:
        #             ik_result[i] = lower
        #         if ik_result[i] > upper and ik_result[i] - 2*np.pi < lower:
        #             ik_result[i] = upper
        #         if ik_result[i] < lower:
        #             ik_result[i] += 2 * np.pi
        #         if ik_result[i] > upper:
        #             ik_result[i] -= 2 * np.pi

        ik_result = self.convert_range(ik_result)
        if avoid_collisions:
            if not self.collision_fn_notarget(ik_result): return np.array(ik_result)
        else:
            return np.array(ik_result)
        return None


    def get_ik_fast(self, eef_pose, avoid_collisions=True, arm_joint_values=None, ignore_last_joint=True,
                    use_grasp_roll_duality=True): # no ik would return None; ik_fast return joint angle within [-pi, pi] !!!
        arm_joint_values = self.get_arm_joint_values()
        ik_results = self.get_ik_fast_full(eef_pose)
        if avoid_collisions:
            # avoid all collision
            collision_free = [not self.collision_fn_full(ik) for ik in ik_results]
            ik_results = np.array(ik_results)[np.where(collision_free)]
        else:
            # only allow collision with target object
            collision_free = [not self.collision_fn_notarget(ik) for ik in ik_results]
            ik_results = np.array(ik_results)[np.where(collision_free)]
        if not ik_results.any():
            return None
        if arm_joint_values is not None:
            if ignore_last_joint:
                jv_dists = np.linalg.norm(ik_results[:, :-1] - np.array(arm_joint_values)[:-1], axis=1)
            else:
                jv_dists = np.linalg.norm(ik_results - np.array(arm_joint_values), axis=1)
            ik_result = ik_results[np.argsort(jv_dists)[0]]
            if use_grasp_roll_duality:
                # parrallel jaw grasp roll duality
                reference = normalize_angle(arm_joint_values[-1])
                original = normalize_angle(ik_result[-1])
                dual = normalize_angle(ik_result[-1] + np.pi)
                if np.abs(dual - reference) < np.abs(original - reference):
                    ik_result[-1] = dual
        else:
            ik_result = ik_results[0]
        ik_result = np.array(self.convert_range(ik_result))
        return ik_result # TODO why I use numpy array return rather than list??


    def get_ik_fast_full(self, eef_pose):

        # base_2_shoulder = gu.get_transform('base_link', 'shoulder_link')
        # base_2_shoulder = ([0.0, 0.0, 0.089159], [0.0, 0.0, 1.0, 0.0])
        # the z-offset (0.089159) is from kinematics_file config in ur_description
        base_correction = (self.base_link_initial_pose[0], [0.0, 0.0, 1.0, 0.0]) # why correction? Because ikfast assume your base is [0, 0, 0], [0, 0, 0, 1]

        # ee_2_wrist3 = gu.get_transform('ee_link', 'wrist_3_link')
        ee_2_wrist3 = ([0.0, 0.0, 0.0], [-0.5, 0.5, -0.5, 0.5])  # also could use pybullet ee_2_wrist3 = p.multiplyTransform(p.invertTransform(wrist3), ee_pose)

        wrist_3_pose_in_shoulder = pu.multiply_multi_transforms(p.invertTransform(*base_correction), eef_pose, ee_2_wrist3)

        wrist_3_pose_in_shoulder = pu.pose2d_matrix(wrist_3_pose_in_shoulder)[:3]
        joint_configs = self.ur5_kin.inverse(wrist_3_pose_in_shoulder.reshape(-1).tolist())

        n_joints = self.ur5_kin.getDOF()
        n_solutions = int(len(joint_configs) / n_joints)
        joint_configs = np.asarray(joint_configs).reshape(n_solutions, n_joints)
        return joint_configs


    def plan_arm_joint_values(self, goal_joint_values, start_joint_values=None, maximum_planning_time=0.05,
                              previous_discretized_plan=None, start_joint_velocities=None, planner='birrt'):
        start_joint_values = self.get_arm_joint_values() if start_joint_values is None else start_joint_values
        # goal_joint_values are within [-pi, pi], so the arm joint_values should be within that range as well
        start_joint_values_converted = UR5RobotiqPybulletController.convert_range(start_joint_values)
        goal_joint_values_converted = UR5RobotiqPybulletController.convert_range(goal_joint_values)
        seed_trajectory = None

        RRT_plan, iter_time = self.rrt.plan_motion(start_joint_values_converted, goal_joint_values_converted,
                                                   maximum_planning_time=maximum_planning_time,
                                                   start_joint_velocities=start_joint_velocities,
                                                   seed_trajectory=seed_trajectory,
                                                   planner=planner)

        if not RRT_plan or len(RRT_plan) == 0: return None, iter_time
        else: return RRT_plan, iter_time  # already discreate plan


    def plan_arm_joint_values_simple(self, goal_joint_values, start_joint_values=None, duration=None):
        """ Linear interpolation between joint_values """
        start_joint_values = self.get_arm_joint_values() if start_joint_values is None else start_joint_values
        start_joint_values_converted = UR5RobotiqPybulletController.convert_range(start_joint_values)
        goal_joint_values_converted = UR5RobotiqPybulletController.convert_range(goal_joint_values)

        diffs = self.arm_difference_fn(goal_joint_values_converted, start_joint_values_converted)
        steps = np.abs(np.divide(diffs, self.MOVEIT_ARM_MAX_VELOCITY)) * 240 # joint distance / max_velocity = required time
        num_steps = int(max(steps))
        if duration is not None:
            num_steps = max(int(duration * 240), steps)     # this should ensure that it satisfies the max velocity of the end-effector

        goal_joint_values = np.array(start_joint_values_converted) + np.array(diffs)
        waypoints = np.linspace(start_joint_values_converted, goal_joint_values_converted, num_steps).tolist()
        return waypoints


    def plan_straight_line(self, eef_pose, start_joint_values=None, ee_step=0.05,
                           jump_threshold=3.0, avoid_collisions=False):
        if start_joint_values is None: start_joint_values = self.get_arm_joint_values()
        start_joint_values_converted = self.convert_range(start_joint_values)
        discretized_plan = self.rrt.plan_straight_line(start_joint_values_converted, eef_pose, ee_step=ee_step, 
                                                       avoid_collisions=avoid_collisions)
        return discretized_plan


    def plan_cartesian_control(self, x=0.0, y=0.0, z=0.0, frame="world"):
        """
        Only for small motion, do not check friction
        :param frame: "eef" or "world"
        """
        if frame == "eef":
            pose_2d = self.get_eef_pose()
            pose_2d_new = pu.multiply_multi_transforms(pose_2d, ((x, y, z), (0, 0, 0, 1)))
        elif frame == "world":
            pose_2d_new = self.get_eef_pose()
            pose_2d_new[0][0] += x
            pose_2d_new[0][1] += y
            pose_2d_new[0][2] += z
        else:
            raise TypeError("not supported frame: {}".format(frame))
        discretized_plan = self.plan_straight_line(pose_2d_new, ee_step=0.005, avoid_collisions=False)
        return discretized_plan


    @staticmethod
    def convert_range(joint_values):
        """ Convert continuous joint to range [-pi, pi] """
        circular_idx = set([0, 1, 2, 3, 4, 5]) # [1, 2, 3, 4, 5, 6] joints
        new_joint_values = []
        for i, v in enumerate(joint_values):
            if i in circular_idx:
                new_joint_values.append(pu.wrap_angle(v))
            else:
                new_joint_values.append(v)
        return new_joint_values


    @staticmethod
    def process_plan(moveit_plan, start_joint_values):
        """
        convert position trajectory to work with current joint values
        :param moveit_plan: MoveIt plan
        :return plan: Motion
        """
        diff = np.array(start_joint_values) - np.array(moveit_plan.joint_trajectory.points[0].positions)
        for p in moveit_plan.joint_trajectory.points:
            p.positions = (np.array(p.positions) + diff).tolist()
        plan = UR5RobotiqPybulletController.extract_plan(moveit_plan)
        return plan


    @staticmethod
    def discretize_plan(motion_plan): # not use velocity trajectory
        """ return np array """
        discretized_plan = np.zeros((0, 6))
        for i in range(len(motion_plan.position_trajectory) - 1):
            num_steps = (motion_plan.time_trajectory[i + 1] - motion_plan.time_trajectory[i]) * 240
            segment = np.linspace(motion_plan.position_trajectory[i], motion_plan.position_trajectory[i + 1], num_steps)
            if i + 1 == len(motion_plan.position_trajectory) - 1:
                discretized_plan = np.vstack((discretized_plan, segment))
            else:
                discretized_plan = np.vstack((discretized_plan, segment[:-1]))
        return discretized_plan


    @staticmethod
    def process_discretized_plan(discretized_plan, start_joint_values):
        """
        convert discretized plan to work with current joint values
        :param discretized_plan: discretized plan, list of waypoints
        :return plan: Motion
        """
        diff = np.array(start_joint_values) - np.array(discretized_plan[0])
        new_discretized_plan = []
        for wp in discretized_plan:
            new_discretized_plan.append((np.array(wp) + diff).tolist())
        return new_discretized_plan


    @staticmethod
    def extract_plan(moveit_plan):
        """
        Extract np arrays of position, velocity and time trajectories from moveit plan,
        and return Motion object
        """
        points = moveit_plan.joint_trajectory.points
        position_trajectory = []
        velocity_trajectory = []
        time_trajectory = []
        for p in points:
            position_trajectory.append(list(p.positions))
            velocity_trajectory.append(list(p.velocities))
            time_trajectory.append(p.time_from_start.to_sec())
        return Motion(np.array(position_trajectory), np.array(time_trajectory), np.array(velocity_trajectory))


    # execution
    def execute_arm_plan(self, plan, realtime=False, sleep=None):
        """
        execute a discretized arm plan (list of waypoints)
        """
        index = 0
        for wp in plan:
            index += 1; print(index)
            self.control_arm_joints(wp)
            p.stepSimulation(physicsClientId=self.client_id)
            # too short time to get stable performance for control
            if realtime:
                time.sleep(1. / 240.)
            if not realtime and sleep:
                time.sleep(sleep)
        pu.step(2, client_id=self.client_id)


    def execute_gripper_plan(self, plan, realtime=False):
        """
        execute a discretized gripper plan (list of waypoints)
        """
        for wp in plan:
            self.control_gripper_joints(wp)
            p.stepSimulation(physicsClientId=self.client_id)
            if realtime:
                time.sleep(1. / 240.)
        pu.step(2, client_id=self.client_id)


    def step(self, fix_turn_around=True):
        """ step the robot for 1/240 second """
        # calculate the latest conf and control array
        if self.arm_discretized_plan is None:
            self.control_arm_joints(self.start_joint_values)   # no grasp pose just not move or fall down
        elif self.arm_wp_target_index == len(self.arm_discretized_plan):
            self.control_arm_joints(self.arm_discretized_plan[-1])  # stay pose, do not fall down!
        else:
            if fix_turn_around:
                # if current joint is not in [-pi, pi] but next step must be within the range,
                # we need to fix current angle use set; Only if we want to monitor pybullet joints output, we use raw get_arm_joint_values
                # Guess that pybullet joints range is in [0, 2pi]? This fix should be embedde in control_arm_joints! TODO
                cur_joint = np.array(self.get_arm_joint_values())
                if (np.abs(cur_joint-np.array(self.arm_discretized_plan[self.arm_wp_target_index])) > np.pi).any():
                    self.set_arm_joints([pu.wrap_angle(jv) for jv in cur_joint])
            self.control_arm_joints(self.arm_discretized_plan[self.arm_wp_target_index])
            self.arm_wp_target_index += 1

        if self.gripper_discretized_plan is None:
            self.control_gripper_joints(self.OPEN_POSITION)    # keep original pose
        elif self.gripper_wp_target_index == len(self.gripper_discretized_plan):
            self.control_gripper_joints(self.gripper_discretized_plan[-1])
        else:
            self.control_gripper_joints(self.gripper_discretized_plan[self.gripper_wp_target_index])
            self.gripper_wp_target_index += 1


    def equal_conf(self, conf1, conf2, tol=0):
        adapted_conf2 = self.adapt_conf(conf2, conf1) # convert conf2 range to -pi~pi
        return np.allclose(conf1, adapted_conf2, atol=tol)


    def adapt_conf(self, conf2, conf1):
        """ adapt configuration 2 to configuration 1"""
        diff = self.arm_difference_fn(conf2, conf1)
        adapted_conf2 = np.array(conf1) + np.array(diff)
        return adapted_conf2.tolist()


    def reset_joint_values(self, joint_indices, joint_values):
        for i, v in zip(joint_indices, joint_values):
            p.resetJointState(self.id, i, v, physicsClientId=self.client_id)


    def reset_arm_joint_values(self, joint_values):
        self.reset_joint_values(self.GROUP_INDEX['arm'], joint_values)


    def reset_gripper_joint_values(self, joint_values):
        self.reset_joint_values(self.GROUP_INDEX['gripper'], joint_values)


    def set_group_joint_values(self, group_joint_indices, joint_values):
        p.setJointMotorControlArray(self.id, group_joint_indices, p.POSITION_CONTROL, joint_values,
                                    forces=[500] * len(joint_values), physicsClientId=self.client_id)


    def set_arm_joint_values(self, joint_values):
        self.set_group_joint_values(self.GROUP_INDEX['arm'], joint_values)


    def set_gripper_joint_values(self, joint_values=(0,)*6):
        self.set_group_joint_values(self.GROUP_INDEX['gripper'], joint_values)


    def get_joint_state(self, joint_index):
        return self.JointState(*p.getJointState(self.id, joint_index, physicsClientId=self.client_id))


    def get_arm_joint_values(self, convert_range=True):
        if convert_range:
            return self.convert_range([self.get_joint_state(i).jointPosition for i in self.GROUP_INDEX['arm']])
        else:
            return [self.get_joint_state(i).jointPosition for i in self.GROUP_INDEX['arm']]


    def get_gripper_joint_values(self):
        return [self.get_joint_state(i).jointPosition for i in self.GROUP_INDEX['gripper']]


    def get_eef_pose(self):
        return pu.get_link_pose(self.id, self.EEF_LINK_INDEX, client_id=self.client_id)


    def move_gripper_joint_values(self, joint_values, duration=1.0, num_steps=10):
        """ this method has nothing to do with moveit """
        start_joint_values = self.get_gripper_joint_values()
        goal_joint_values = joint_values
        position_trajectory = np.linspace(start_joint_values, goal_joint_values, num_steps)
        for i in range(num_steps):
            p.setJointMotorControlArray(self.id, self.GROUP_INDEX['gripper'], p.POSITION_CONTROL,
                                        position_trajectory[i], forces=[200] * len(joint_values), physicsClientId=self.client_id)
            time.sleep(duration / num_steps)


    # Reset inertial rng generator
    def reset_inner_rng(self):
        self.initial_pose_rng = np.random.default_rng(self.pose_rng_seed)


    # random select start space
    def random_initial(self, box_ext=0.003):
        rng = self.initial_pose_rng if self.initial_pose_rng else np.random
        if hasattr(self, 'box_id'): pu.remove_body(self.box_id, client_id=self.client_id) # remove last box_id to save memory
        position, orientation = self.get_arm_fk_pybullet(self.initial_joint_values)
        if self.initial_visualize:
            self.box_id = pu.draw_box_body(position, halfExtents=[box_ext] * 3, rgba_color=[1, 0, 0, 0.7], client_id=self.client_id)
        upper_bound = np.array(position) + box_ext
        lower_bound = np.array(position) - box_ext
        res = None; reset_try_times = 0; maximum_reset_try = 200; 
        while res is None:
            new_position = rng.uniform(lower_bound, upper_bound).tolist()
            new_orientation = p.getQuaternionFromEuler(rng.uniform(-np.pi, np.pi, size=3).tolist())
            res = self.get_arm_ik([new_position, new_orientation], avoid_collisions=False)
            reset_try_times += 1
            if reset_try_times >= maximum_reset_try:
                res = np.array(self.initial_joint_values)
                print(f"Can not reset UR5 collision-free to position: {new_position}, orientation: {orientation}!")
                # np.save(f"position_{new_position}_orientation_{orientation}_is_stuck", new_position+orientation)
                break
        return res.tolist()



if __name__=="__main__":
    from pynput import keyboard
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)
    p.loadURDF("plane.urdf", physicsClientId=client_id)
    robot_id, urdf_path = load_ur_robotiq_robot(client_id=client_id)
    robot = UR5RobotiqPybulletController(robot_id, rng=None, client_id=client_id)
    robot.reset()
    
    eef_pose = robot.get_eef_pose()
    with keyboard.Events() as events:
        while True:
            key = None
            event = events.get(0.0001)
            if event is not None:
                if isinstance(event, events.Press) and hasattr(event.key, 'char'):
                    key = event.key.char
            if key is not None:
                if key == 's':
                    eef_pose[0][0] += 0.01
                elif key == 'w':
                    eef_pose[0][0] -= 0.01
                elif key == 'a':
                    eef_pose[0][1] += 0.01
                elif key == 'd':
                    eef_pose[0][1] -= 0.01
                print(eef_pose)
            joint_values = robot.get_arm_ik(eef_pose)
            robot.control_arm_joints(joint_values)
            p.stepSimulation(physicsClientId=client_id)
            time.sleep(1./240.)
