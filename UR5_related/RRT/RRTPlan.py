import numpy as np
from math import ceil
import pybullet_utils as pu
from UR5_related.RRT.rrt import rrt
from UR5_related.RRT.rrt_star import rrt_star, rrt_star_connect
from UR5_related.RRT.rrt_connect import birrt, NODE_T
from itertools import combinations, product
import time

class RRTPlanner: # could multi
    """
    :parameter controllers: UR5RobotiqPybulletController Object, could be multi but need a list to cover
    """
    def __init__(self, controllers, client_id=0):
        self.num_robots = len(controllers)
        self.ur5_controller = controllers[0] # we use only one ur5 RRT implementation to speed up motion planning
        
        self.controllers = list(controllers)
        self.robot_ids = [controller.id for controller in controllers]
        self.dof = sum([len(controller.GROUP_INDEX['arm']) for controller in controllers]) # whether gripper also needs?
        self.client_id = client_id

        self.min_velocity = [] # compute min_velocity of all joints' max velocity
        min_velocity = []
        for controller in controllers:
            for index in controller.GROUP_INDEX['arm']:
                min_velocity.append(pu.get_max_velocity(controller.id, index, client_id=self.client_id))
            self.min_velocity.append(min(min_velocity))

        self.resolutions = 1 / 240 * self.min_velocity[0] if self.min_velocity[0] < 4.8 else 0.02 # adjust resolution to fit velocity
        self.resolutions *= 0.8  # smooth the resolutions to give some space to avoid collision
        self.extend_fn = self.get_extend_fn(self.resolutions)

    
    def set_joint_positions(self, joint_values):
        assert len(joint_values) == self.dof
        self.ur5_controller.set_arm_joints(joint_values)
    
    def get_joint_positions(self):
        return self.ur5_controller.get_arm_joint_values()
    
    def difference_fn(self, q1, q2):
        return self.ur5_controller.arm_difference_fn(q1, q2)

    def distance_fn(self, q1, q2):
        diff = np.array(self.difference_fn(q2, q1))
        return np.sqrt(np.dot(diff, diff))
    
    def sample_fn(self):
        return self.ur5_controller.arm_sample_fn()
    
    def get_extend_fn(self, resolutions=None):  # not include q1 in waypoints
        if resolutions is None:
            resolutions = 0.05 * np.ones(self.dof)

        def fn(q1, q2):
            diffs = self.difference_fn(q2, q1)
            steps = np.abs(np.divide(diffs, resolutions))
            num_steps = ceil(max(steps)) # pick the maximum steps which is most long trajectory
            waypoints = (np.linspace(0, diffs, num=num_steps) + np.array(q1)).tolist() 
            return waypoints
        return fn
    
    
    def get_collision_fn(self, obstacles=[], attachments=[], self_collisions=True,
                         disabled_targets=[], disabled_self_collisions=set()):
        # check_link_pairs is a 2d list
        check_link_pairs = []
        if self_collisions:
            check_link_pairs += pu.get_self_link_pairs(self.ur5_controller.id, self.ur5_controller.GROUP_INDEX['arm'], disabled_self_collisions, client_id=self.client_id)
        moving_bodies = self.robot_ids + [attachment for attachment in attachments]
        if len(obstacles) == 0: obstacles = list(set(pu.get_bodies(client_id=self.client_id)) - set(moving_bodies))
        if len(disabled_targets) > 0: [obstacles.remove(disabled_t) for disabled_t in disabled_targets if disabled_t in obstacles] # multi target needs to use loop
        print(f'obstacles: {obstacles}; All ids {pu.get_bodies(client_id=self.client_id)}')
        check_body_pairs = list(product(moving_bodies, obstacles)) + list(combinations(moving_bodies, 2))
        def collision_fn(q): # set may consume time? reset ought to make outside
            col_FLAG = False
            ### Ur5 arm joints are all rotable joints / no circular limits ###
            # if pu.violates_limits(self.ur5_controller.id, self.ur5_controller.GROUP_INDEX['arm'], q, client_id=self.client_id):
            #     return True
            cur_joints = self.get_joint_positions()
            self.set_joint_positions(q)

            for link1, link2 in check_link_pairs:
                if pu.pairwise_link_collision(self.ur5_controller.id, link1, self.ur5_controller.id, link2, client_id=self.client_id):
                    print(f'Self Collision!! {link1}/{link2}')
                    col_FLAG = True; break

            if not col_FLAG:
                for pair in check_body_pairs:
                    if pu.pairwise_collision(*pair, client_id=self.client_id):
                        print(f'collision with obstacles: {pair[0]} and {pair[1]}')
                        col_FLAG = True; break

            self.set_joint_positions(cur_joints) # reset joints! maybe not need during planning, 
                                                 # but has no good solution (only a bunch of joints check together)
            return col_FLAG
        return collision_fn


    def forward_kinematics(self, q):
        """ return a list of eef poses """
        poses = []
        split_q = split(q, self.num_robots)
        for ctrl, q_ in zip(self.controllers, split_q):
            poses.append(ctrl.get_arm_fk_pybullet(q_))
        return poses


    def plan_motion(self,
                    start_conf,
                    goal_conf,
                    planner='birrt',
                    greedy=True,
                    goal_tolerance=0.001,
                    goal_bias=0.2,
                    resolutions=0.01,
                    iterations=100000,
                    restarts=1,
                    obstacles=[],
                    attachments=[],
                    self_collisions=True,
                    disabled_collisions=set(),
                    maximum_planning_time=0.05,
                    start_joint_velocities=None,
                    seed_trajectory=None,
                    start_check=False,
                    visualize=False):

        # get some functions, consume more time
        self.collision_fn_full = self.ur5_controller.collision_fn_full  # full collision as robot collision check

        if planner == 'birrt':
            for i in range(restarts):
                path_conf, planning_time = birrt(start_conf=start_conf,
                                                goal_conf=goal_conf,
                                                distance=self.distance_fn,
                                                sample=self.sample_fn,
                                                extend=self.extend_fn,
                                                collision=self.collision_fn_full,
                                                iterations=iterations,
                                                visualize=visualize,
                                                fk=self.forward_kinematics,
                                                greedy=greedy,
                                                maximum_planning_time=maximum_planning_time,
                                                start_check=start_check)
                if path_conf is None:
                    # if planning_time == NODE_T: print('!!!Start or Goal Collision!!!')
                    # elif planning_time >= maximum_planning_time: print('!!!Time Use Up!!!')
                    # else: print('!!!Collision During RRT Plan!!!')
                    if visualize: pu.remove_all_markers(client_id=self.client_id)
                else:
                    break
            
            ### Smooth or desample path to strictly match the maximum joint velocity ###
            # if path_conf is not None and resolutions / self.min_velocity[0] > 1 / 240:  # one step time
            #     path_conf = self.smooth(path_conf, resolutions)
            # if path_conf is not None and int((1/240) / (resolutions/self.min_velocity[0])) >= 2:  # desample path to fit velocity
            #     path_conf = self.desample(path_conf, resolutions)

        elif planner == 'birrt_star':
            for i in range(restarts):
                iter_start = time.time()
                path_conf, planning_time = rrt_star_connect(start_conf=start_conf,
                                                            goal_conf=goal_conf,
                                                            distance=self.distance_fn,
                                                            sample=self.sample_fn,
                                                            extend=self.extend_fn,
                                                            collision=self.collision_fn_full,
                                                            iterations=iterations,
                                                            radius=0.15,
                                                            visualize=False,
                                                            fk=self.forward_kinematics,
                                                            group=True,
                                                            greedy=greedy,
                                                            maximum_planning_time=maximum_planning_time,
                                                            start_check=start_check)
                if path_conf is None:
                    if visualize: pu.remove_all_markers(client_id=self.client_id)
                else:
                    break
        else:
            raise ValueError('planner must be in \'rrt\' or \'birrt\'')

        return path_conf, planning_time

    def smooth(self, path, resolutions, real_time = 1 / 240):
        need_time = resolutions / self.min_velocity[0] # now only have one arm
        need_steps = ceil(need_time / real_time) + 1 # np.linspace include start and end point
        new_path = []
        for i in range(len(path)-1):
            new_path += np.linspace(path[i], path[i+1], need_steps)[:-1].tolist()
        new_path.append(path[-1])
        return new_path

    def desample(self, path, resolutions, real_time = 1 / 240):
        need_time = resolutions / self.min_velocity[0]
        jump_step = int(real_time / need_time)
        new_path = path[0::jump_step]
        if new_path[-1] != path[-1]: # miss goal
            new_path.append(path[-1])
        return new_path

    def plan_straight_line(self, start_joint_values, end_eef_pose, ee_step=0.05, jump_threshold=3.0,
                           avoid_collisions=False, smooth=True):
        """
        :param start_joint_values: start joint values
        :param end_eef_pose: goal end effector pose
        :param ee_step: float. The distance in meters to interpolate the path.
        :param jump_threshold: The maximum allowable distance in the arm's
            configuration space allowed between two poses in the path. Used to
            prevent "jumps" in the IK solution.
        :param avoid_collisions: bool. Whether to check for obstacles or not. DO not check collision still outputs bad resutls ik_fast
        :return:
        """
        controller = self.controllers[0] # focus on ur5
        start_eef_pose = controller.get_arm_fk_pybullet(start_joint_values) # need use fk_pybullet, fk_ikfast's orientation is wrong!!
        start_eef_pose = pu.merge_pose_2d(start_eef_pose); end_eef_pose = pu.merge_pose_2d(end_eef_pose)
        start_eef_pose = np.array(start_eef_pose); end_eef_pose = np.array(end_eef_pose)
        steps = ceil(np.linalg.norm(end_eef_pose[:3] - start_eef_pose[:3]) / ee_step)
        discretized_plan = np.linspace(start_eef_pose, end_eef_pose, steps)
        self.discretized_plan = discretized_plan
        arm_motion_plan = []
        for pose in discretized_plan:
            pose = pu.split_7d(pose)
            res = controller.get_arm_ik(pose, avoid_collisions=avoid_collisions, solver='ikpybullet')
            if res is None: return None # collision happens
            arm_motion_plan.append(res)
        if smooth:  # insert plan to make sure each point could get at 1/240 time based on smallest velocity
            worst_resolutions = max(map(max, np.diff(arm_motion_plan, axis=0)))
            arm_motion_plan = self.smooth(arm_motion_plan, worst_resolutions)
        return arm_motion_plan

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]