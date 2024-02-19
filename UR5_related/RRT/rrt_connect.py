import time
import numpy as np
from UR5_related.RRT.smoothing import smooth_path
from UR5_related.RRT.rrt import TreeNode, configs
from UR5_related.RRT.rrt_utils import irange, argmin
import pybullet_utils as pu

NODE_T = 0.001 # one node extends time
def rrt_connect(q1,
                q2,
                distance,
                sample,
                extend,
                collision,
                iterations,
                visualize,
                fk,
                greedy, # greedy is usually true to save time
                maximum_planning_nodes,
                seed_nodes=None):
    if maximum_planning_nodes <= 0: return None, 0  # Budget nodes already used up
    root1, root2 = TreeNode(q1), TreeNode(q2)
    if seed_nodes: nodes1, nodes2 = seed_nodes, [root2] # seed_nodes: q1 does not create here!! could not use object equal to check!
    else: nodes1, nodes2 = [root1], [root2]

    node_num = 0 # start planning
    for _ in irange(iterations):
        if len(nodes1) > len(nodes2):  # why swap: seems to make the tree balance?
            nodes1, nodes2 = nodes2, nodes1
        s = sample()
        last1 = argmin(lambda n: distance(n.config, s), nodes1)
        for q in extend(last1.config, s):
            node_num += 1
            if collision(q): break
            last1 = TreeNode(q, parent=last1)
            nodes1.append(last1)
            if not greedy: break
            if time_budget_used_up(node_num, maximum_planning_nodes): break

        last2 = argmin(lambda n: distance(n.config, last1.config), nodes2)
        if time_budget_used_up(node_num, maximum_planning_nodes): break # put here make sure the function has last2
        for q in extend(last2.config, last1.config):
            node_num += 1
            if collision(q): break
            last2 = TreeNode(q, parent=last2)
            nodes2.append(last2)
            if not greedy: break
            if time_budget_used_up(node_num, maximum_planning_nodes): break

        if np.allclose(last2.config, last1.config, atol=1e-3, rtol=0): # success plan
            path1, path2 = last1.retrace(), last2.retrace()
            if path1[0].config != root1.config:
                path1, path2 = path2, path1
            return configs(path1[:-1] + path2[::-1]), time_consume(node_num)
        
        if time_budget_used_up(node_num, maximum_planning_nodes): break

    if visualize:  # visualize tree shape
        for node in nodes1:
            if node.parent is not None:
                start, end = fk(node.config)[0][0], fk(node.parent.config)[0][0] # compute fk
                pu.draw_line(start, end)
        for node in nodes2:
            if node.parent is not None:
                start, end = fk(node.config)[0][0], fk(node.parent.config)[0][0] # compute fk
                pu.draw_line(start, end)

    # # return random path
    # path1, path2 = last1.retrace(), last2.retrace()
    # if path1[0].config == root1.config: return configs(path1), time_consume(node_num)
    # elif path2[0].config == root1.config: return configs(path2), time_consume(node_num)

    ## return cloeset path
    # if nodes1[0].config == root1.config: root_path = nodes1
    # elif nodes2[0].config == root1.config: root_path = nodes2
    # goal_pos = fk(q2)[0][0]
    # cloeset_node = root_path[np.argmin([pose_difference(fk(node.config)[0][0], goal_pos) for node in root_path])]
    # return configs(cloeset_node.retrace()), time_consume(node_num)

    ## return no path, if plan unsuccessfully, would return no path
    return None, time_consume(node_num)


def direct_path(q1, q2, extend, collision, maximum_planning_nodes=None):
    last1 = TreeNode(q1)
    path = [last1]; node_num = 2 # include start and end point in compute time
    for q in extend(q1, q2):
        node_num += 1
        if collision(q):
            return path, False, node_num # used budget nodes
        last1 = TreeNode(q, parent=last1)
        path.append(last1)
        if maximum_planning_nodes and node_num >= maximum_planning_nodes: 
            # print('Direct time:', time_consume(node_num));
            break
    return path, True, node_num


def birrt(start_conf,
          goal_conf,
          distance,
          sample,
          extend,
          collision,
          iterations,
          visualize,
          fk,
          greedy,
          hard_model=False, # hardmodel: rrt will output None only if the path is exactly ok
          maximum_planning_time=None,
          start_check=False):
    """
    return a list of path (joint values), including start point and end point
    """
    if maximum_planning_time is not None: maximum_planning_nodes = int(maximum_planning_time / NODE_T) # transfer maximum_planning_time to maximum_planning_nodes
    else: maximum_planning_nodes = 4e3 # maximum planning nodes
    iterations = maximum_planning_nodes # the worst case is each iteration it will collision check once, should be same as nodes num

    if np.allclose(maximum_planning_nodes, 0): # give up this planning, time budget is close to 0
        return None, 0
    if collision(start_conf):
        print('<<<<<start conf collision>>>>>')
        if start_check: return None, time_consume(1) # give start node check
    if collision(goal_conf):
        print('<<<<<goal conf collision>>>>>')
        return None, time_consume(2)  # give end node check
    
    # If we only use no collision check direct_path to speed up RRT planner
    path, over, used_nodes = direct_path(start_conf, goal_conf, extend, collision, maximum_planning_nodes=maximum_planning_nodes)

    if over: # plan success or time use up
        path_pos = configs(path)
        if hard_model and path_pos[-1] != goal_conf: return None # Hard Model: only use direct path for planning not RRT
        return path_pos, time_consume(used_nodes)
    direct_path_time = time_consume(used_nodes)
    
    # iterations does not control the real time budget, it only needs to bigger than maximum_planning_nodes
    path, rrt_time = rrt_connect(start_conf,
                                 goal_conf,
                                 distance,
                                 sample,
                                 extend,
                                 collision,
                                 iterations,
                                 visualize,
                                 fk,
                                 greedy,
                                 maximum_planning_nodes-used_nodes,
                                 path) # continue to plan use direct path as a seed trajectory

    if path:
        if hard_model and path[-1] != goal_conf: return None, direct_path_time + rrt_time
        else: return path, direct_path_time + rrt_time
    else: return None, direct_path_time + rrt_time

def time_consume(num_node):
    return num_node * NODE_T

def pose_difference(pose1, pose2):
    pose1, pose2 = np.array(pose1), np.array(pose2)
    return np.linalg.norm(pose1-pose2)

def time_budget_used_up(used_nodes, maximum_planning_nodes=None):
    if maximum_planning_nodes is None: return False
    elif used_nodes >= maximum_planning_nodes: return True
    else: return False
