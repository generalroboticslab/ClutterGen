from random import random
from time import time
import pybullet_utils as pu
import time
from UR5_related.RRT.rrt_utils import INF, argmin
import numpy as np

NODE_T = 0.001 # one node extends time
class OptimalNode(object):

    def __init__(self, config, parent=None, d=0, path=[], iteration=None, visualize=False, group=True, fk=None, rgb_color=(0, 1, 0)):
        self.config = config
        self.parent = parent
        self.children = set()
        self.d = d
        self.path = path    # the path does not include itself or its parent
        self.visualize = visualize
        self.fk = fk
        self.group = group
        self.marker_ids = []
        self.solution_marker_ids = []
        self.rgb_color = rgb_color
        if self.visualize and parent is not None:
            self.draw_path()
        if parent is not None:
            self.cost = parent.cost + d
            self.parent.children.add(self)
        else:
            self.cost = d
        self.solution = False
        self.creation = iteration # record create this node's time
        self.last_rewire = iteration # record last modify this node's time

    def set_solution(self, solution): # label the solution path
        if self.solution is solution:
            return
        self.solution = solution
        # visualize
        if self.visualize and self.parent is not None:
            if solution is True:
                self.draw_solution_path()
            else:
                # used to a solution, now is not a solution node
                self.remove_solution_path()
        if self.parent is not None:
            self.parent.set_solution(solution)

    def retrace(self):
        if self.parent is None:
            return self.path + [self.config]
        return self.parent.retrace() + self.path + [self.config]

    def rewire(self, parent, d, path, iteration=None):
        if self.solution:
            self.parent.set_solution(False)
        self.parent.children.remove(self)
        self.parent = parent
        self.parent.children.add(self)
        if self.solution:
            self.parent.set_solution(True)
        self.d = d
        self.path = path
        # visualize
        if self.visualize and parent is not None:
            self.remove_path()
            self.draw_path()
        self.update()
        self.last_rewire = iteration

    def update(self):
        self.cost = self.parent.cost + self.d
        for n in self.children:
            n.update()

    def clear(self):
        self.node_handle = None
        self.edge_handle = None

    def remove_solution_path(self):
        for i in self.solution_marker_ids:
            pu.remove_marker(i)
        self.solution_marker_ids = []

    def remove_path(self):
        for i in self.marker_ids:
            pu.remove_marker(i)
        self.marker_ids = []

    def draw_path(self):
        assert self.fk is not None, 'please provide a fk when visualizing'
        if self.group:
            for q_prev, q_now in zip([self.parent.config] + self.path, self.path + [self.config]):
                for pose_now, pose_prev in zip(self.fk(q_prev), self.fk(q_now)):
                    self.marker_ids.append(pu.draw_line(pose_prev[0], pose_now[0], rgb_color=self.rgb_color, width=1))
        else:
            for q_prev, q_now in zip([self.parent.config] + self.path, self.path + [self.config]):
                p_now = self.fk(q_prev)[0]
                p_prev = self.fk(q_now.config)[0]
                self.marker_ids.append(pu.draw_line(p_prev, p_now, rgb_color=self.rgb_color, width=1))

    def draw_solution_path(self):
        assert self.fk is not None, 'please provide a fk when visualizing'
        if self.group:
            for q_prev, q_now in zip([self.parent.config] + self.path, self.path + [self.config]):
                for pose_now, pose_prev in zip(self.fk(q_prev), self.fk(q_now)):
                    self.solution_marker_ids.append(pu.draw_line(pose_prev[0], pose_now[0], rgb_color=(1, 0, 0), width=6))
        else:
            for q_prev, q_now in zip([self.parent.config] + self.path, self.path + [self.config]):
                p_now = self.fk(q_prev)[0]
                p_prev = self.fk(q_now.config)[0]
                self.solution_marker_ids.append(pu.draw_line(p_prev, p_now, rgb_color=(1, 0, 0), width=width))

    def __str__(self):
        return self.__class__.__name__ + '(' + str(self.config) + ')'

    __repr__ = __str__


class extend_path:
    def __init__(self, maximum_node, num_node=0):
        self.maximum_node = maximum_node
        self.num_node = num_node
        self.full = False

    def safe_path(self, sequence, collision, greedy=False):
        path = []
        if self.num_node >= self.maximum_node: self.full = True; return path
        for q in sequence:
            self.num_node += 1
            if collision(q):
                break
            path.append(q)

            if self.num_node >= self.maximum_node: self.full = True; break
            if not greedy: break
        return path


def rrt_star(start,
             goal,
             distance,
             sample,
             extend,
             collision,
             radius,
             goal_probability,
             informed,
             visualize,
             fk,
             group,
             max_time=INF,
             max_iterations=INF):
    if collision(start) or collision(goal):
        return None
    nodes = [OptimalNode(start)]
    goal_n = None
    t0 = time()
    it = 0
    while (time() - t0) < max_time and it < max_iterations:
        do_goal = goal_n is None and (it == 0 or random() < goal_probability)
        s = goal if do_goal else sample()
        # Informed RRT*
        if informed and goal_n is not None and distance(start, s) + distance(s, goal) >= goal_n.cost:
            continue
        # if it % 100 == 0:
        #     print(it, time() - t0, goal_n is not None, do_goal, (goal_n.cost if goal_n is not None else INF))
        it += 1

        nearest = argmin(lambda n: distance(n.config, s), nodes)
        path = safe_path(extend(nearest.config, s), collision) # greedy extension to the farest node
        if len(path) == 0:
            continue

        new = OptimalNode(path[-1], parent=nearest, d=distance(nearest.config, path[-1]), path=path[:-1], iteration=it,
                          visualize=visualize, group=group, fk=fk)
        # if safe and do_goal:
        if do_goal and distance(new.config, goal) < 1e-6:
            goal_n = new
            goal_n.set_solution(True)
        # TODO - k-nearest neighbor version
        neighbors = list(filter(lambda n: distance(n.config, new.config) < radius, nodes))
        nodes.append(new)

        # check if we should change new's parent to a neighbor
        for n in neighbors:
            d = distance(n.config, new.config)
            if n.cost + d < new.cost:
                path = safe_path(extend(n.config, new.config), collision)
                if len(path) != 0 and distance(new.config, path[-1]) < 1e-6:
                    new.rewire(n, d, path[:-1], iteration=it)

        # check if we should change a neighbor's parent to new
        for n in neighbors:  # TODO - avoid repeating work
            d = distance(new.config, n.config)
            if new.cost + d < n.cost:
                path = safe_path(extend(new.config, n.config), collision)
                if len(path) != 0 and distance(n.config, path[-1]) < 1e-6:
                    n.rewire(new, d, path[:-1], iteration=it)
    if goal_n is None:
        return None
    return goal_n.retrace()



def rrt_star_connect(start_conf,
                     goal_conf,
                     distance,
                     sample,
                     extend,
                     collision,
                     radius,
                     visualize,
                     greedy,
                     fk,
                     group,
                     iterations=10000,
                     maximum_planning_time=None,
                     start_check=True):
    """
        return a list of path (joint values), including start point and end point
    """
    min_cost = INF
    if maximum_planning_time is not None: maximum_planning_nodes = int(maximum_planning_time / NODE_T)  # transfer maximum_planning_time to maximum_planning_nodes
    else: maximum_planning_nodes = 1e5  # maximum
    iterations = maximum_planning_nodes  # the worst case is each iteration it will collision check once, should be same as nodes num

    if np.allclose(maximum_planning_nodes, 0): # give up this planning, (might be no path..)
        return None, 0

    if collision(start_conf):
        print('start conf collision:')
        if start_check: return None, NODE_T  # give start node check
    if collision(goal_conf):
        print('goal conf collision:')
        return None, NODE_T * 2  # give start, end nodes check

    # direct path-----------------------------------------
    last1 = OptimalNode(start_conf)
    tree1 = [last1]; node_num = 2; break_Flag = False # include start and end point in compute time
    for q in extend(start_conf, goal_conf):
        node_num += 1
        if collision(q):
            break_Flag = True; break
        last1 = OptimalNode(q, parent=last1, d=distance(q, last1.config))
        tree1.append(last1)
        if maximum_planning_nodes and node_num >= maximum_planning_nodes:
            print('Direct time:', time_consume(node_num)); break_Flag = True; break
    if not break_Flag:
        # print('Direct Path')
        return last1.retrace(), time_consume(node_num)
    #------------------------------------------------------
    tree1, tree2 = tree1, [OptimalNode(goal_conf)]
    new1 = last1
    new2 = None
    path_expander = extend_path(maximum_node=maximum_planning_nodes, num_node=node_num)

    safe_path = path_expander.safe_path

    for it in range(iterations):
        if len(tree1) > len(tree2):
            tree1, tree2 = tree2, tree1
        s = sample()

        # tree 1
        nearest1 = argmin(lambda n: distance(n.config, s), tree1)
        path = safe_path(extend(nearest1.config, s), collision, greedy=greedy)
        if path_expander.full: break
        if len(path) == 0:
            continue
        new1 = OptimalNode(path[-1], parent=nearest1, d=distance(
            nearest1.config, path[-1]), path=path[:-1], iteration=it, group=group, fk=fk)
        neighbors1 = list(filter(lambda n: distance(n.config, new1.config) < radius, tree1))
        tree1.append(new1)

        for n in neighbors1:
            d = distance(n.config, new1.config)
            if n.cost + d < new1.cost:
                path = safe_path(extend(n.config, new1.config), collision)
                if len(path) != 0 and distance(new1.config, path[-1]) < 1e-6:
                    new1.rewire(n, d, path[:-1], iteration=it)
                if path_expander.full: break
        if path_expander.full: break
        for n in neighbors1:  # TODO - avoid repeating work
            d = distance(new1.config, n.config)
            if new1.cost + d < n.cost:
                path = safe_path(extend(new1.config, n.config), collision)
                if len(path) != 0 and distance(n.config, path[-1]) < 1e-6:
                    n.rewire(new1, d, path[:-1], iteration=it)
                if path_expander.full: break
        if path_expander.full: break

        # tree 2
        nearest2 = argmin(lambda n: distance(n.config, new1.config), tree2)
        path = safe_path(extend(nearest2.config, new1.config), collision, greedy=greedy)
        if path_expander.full: break
        if len(path) == 0:
            continue
        new2 = OptimalNode(path[-1], parent=nearest2, d=distance(nearest2.config, path[-1]), path=path[:-1], iteration=it, group=group, fk=fk)
        neighbors2 = list(filter(lambda n: distance(n.config, new2.config) < radius, tree2))
        tree2.append(new2)

        for n in neighbors2:
            d = distance(n.config, new2.config)
            if n.cost + d < new2.cost:
                path = safe_path(extend(n.config, new2.config), collision)
                if len(path) != 0 and distance(new2.config, path[-1]) < 1e-6:
                    new2.rewire(n, d, path[:-1], iteration=it)
                if path_expander.full: break
        if path_expander.full: break

        for n in neighbors2:  # TODO - avoid repeating work
            d = distance(new2.config, n.config)
            if new2.cost + d < n.cost:
                path = safe_path(extend(new2.config, n.config), collision)
                if len(path) != 0 and distance(n.config, path[-1]) < 1e-6:
                    n.rewire(new2, d, path[:-1], iteration=it)
                if path_expander.full: break
        if path_expander.full: break

        # solved
        if np.allclose(new1.config, new2.config):
            new1.set_solution(True)
            new2.set_solution(True)
            min_cost = new1.cost + new2.cost
            path1, path2 = new1.retrace(), new2.retrace()
            if path1[0] != start_conf:
                path1, path2 = path2, path1
            return path1[:-1] + path2[::-1], time_consume(path_expander.num_node)

    if visualize:  # visualize tree shape
        for node in tree1:
            if node.parent is not None:
                start, end = fk(node.config)[0][0], fk(node.parent.config)[0][0] # compute fk
                pu.draw_line(start, end)
        for node in tree2:
            if node.parent is not None:
                start, end = fk(node.config)[0][0], fk(node.parent.config)[0][0] # compute fk
                pu.draw_line(start, end)

    if tree1[0].config == start_conf: res_tree = tree1
    elif tree2[0].config == start_conf: res_tree = tree2
    # return random path
    return res_tree[-1].retrace(), time_consume(path_expander.num_node)

    # return closest path

    # nearest = argmin(lambda n: distance(n.config, goal_conf), res_tree)
    # path = nearest.retrace()
    # return path, time_consume(path_expander.num_node)

def time_consume(num_node):
    return num_node * NODE_T

def pose_difference(pose1, pose2):
    pose1, pose2 = np.array(pose1), np.array(pose2)
    return np.linalg.norm(pose1-pose2)