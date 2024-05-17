import time
from random import random

from robot_related.RRT.rrt_utils import irange, argmin
import pybullet_utils_cust as pu


class TreeNode(object):

    def __init__(self, config, parent=None):
        # configuration
        self.config = config
        # parent configuration
        self.parent = parent

    def retrace(self):
        """
        Get a list of nodes from start to itself
        :return: a list of nodes
        """
        sequence = []
        node = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]

    def __str__(self):
        return 'TreeNode(' + str(self.config) + ')'

    __repr__ = __str__


def configs(nodes):
    """
    Get the configurations of nodes
    :param nodes: array type pf nodes
    :return: a list of configurations
    """
    if nodes is None:
        return None
    return list(map(lambda n: n.config, nodes))


def rrt(start,
        goal_sample,
        distance,
        sample,
        extend,
        collision,
        goal_test=lambda q: False,
        iterations=2000,
        goal_probability=.2,
        greedy=True,
        visualize=False,
        fk=None,
        group=False,
        maximum_planning_time=None):
    """
    RRT algorithm
    :param start: start configuration
    :param goal_sample: goal configuration
    :param distance: distance function distance(q1, q2). Takes two confugurations, returns a number of distance
    :param sample: sampling function sample(). Takes nothing, returns a sample of configuration
    :param extend: extend function extend(q1, q2). Extends the tree from q1 towards q2.
    :param collision: collision checking function collision(q). Check whether collision exists for configuration q
    :param goal_test: function goal_test(q) to test if q is the goal configuration
    :param iterations: number of iterations to extend tree towards an sample
    :param goal_probability: bias probability to set the sample to goal
    :param visualize: whether draw nodes and lines for the tree
    :param fk: forward kinematics function, return pose 2d or a list of pose 2d. This is necessary when visualize is True
    :param group: whether this is a group of arms
    :return: a list of configurations
    """

    if collision(start):
        print("rrt fails, start configuration has collision")
        return None
    nodes = [TreeNode(start)] # 12 dimensions node vector
    start_time = time.time()
    for i in irange(iterations):
        goal = random() < goal_probability or i == 0
        s = goal_sample if goal else sample()
        if goal: print('Now is Goal')

        last = argmin(lambda n: distance(n.config, s), nodes)
        for q in extend(last.config, s):
            if collision(q): break
            if visualize:
                assert fk is not None, 'please provide a fk when visualizing'
                if group:
                    for pose_now, pose_prev in zip(fk(q), fk(last.config)):
                        pu.draw_line(pose_prev[0], pose_now[0], rgb_color=(0, 1, 0), width=1)
                else:
                    p_now = fk(q)[0]
                    p_prev = fk(last.config)[0]
                    pu.draw_line(p_prev, p_now, rgb_color=(0, 1, 0), width=1)
            last = TreeNode(q, parent=last)
            nodes.append(last)
            if goal_test(last.config):
                return configs(last.retrace())
            # if not greedy: break # only extend one resolution point? logistic error? Greedy means one direction until collision
        if maximum_planning_time and time.time() - start_time > maximum_planning_time: break
    print('Only Get Enough close Point')
    return configs(last.retrace())
    # return None