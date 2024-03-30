from collections import namedtuple
import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict, deque, namedtuple
from itertools import product, combinations, count
import open3d as o3d
from math import ceil, sqrt, sin, cos
import xml.etree.ElementTree as ET


INF = np.inf
PI = np.pi
CIRCULAR_LIMITS = -PI, PI
MAX_DISTANCE = 0


def configure_pybullet(rendering=False, debug=False, yaw=50.0, pitch=-35.0, dist=1.2, target=(0.0, 0.0, 0.0)):
    """
    This function is likely to be called multiples times to initiate multiple connections
    Note that Only one local in-process GUI/GUI_SERVER connection allowed.
    Pass in the client ID is important because otherwise they all operate on the first connection

    you can use p.GUI in different python files
    you can use p.DIRECT in a same python file
    """
    if not rendering:
        client_id = p.connect(p.DIRECT)
    else:
        client_id = p.connect(p.GUI)  # be careful about GUI vs GUI_SERVER, GUI_SERVER should only be used with shared memory
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    if not debug:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client_id)
    reset_camera(yaw=yaw, pitch=pitch, dist=dist, target=target, client_id=client_id)
    p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=client_id)
    p.resetSimulation(physicsClientId=client_id)
    p.setGravity(0, 0, -9.8)
    return client_id


def step(duration=1.0, client_id=0):
    for i in range(int(duration * 240)):
        p.stepSimulation(physicsClientId=client_id)


def step_real(duration=1.0, client_id=0):
    for i in range(int(duration * 240)):
        p.stepSimulation(physicsClientId=client_id)
        time.sleep(1.0 / 240.0)


def split_7d(pose):
    return [list(pose[:3]), list(pose[3:])]


def merge_pose_2d(pose):
    return pose[0] + pose[1]


def get_euler_from_quaternion(quaternion):
    return list(p.getEulerFromQuaternion(quaternion))


def get_quaternion_from_euler(euler):
    return list(p.getQuaternionFromEuler(euler))


# Constraints

ConstraintInfo = namedtuple('ConstraintInfo', ['parentBodyUniqueId', 'parentJointIndex',
                                               'childBodyUniqueId', 'childLinkIndex', 'constraintType',
                                               'jointAxis', 'jointPivotInParent', 'jointPivotInChild',
                                               'jointFrameOrientationParent', 'jointFrameOrientationChild',
                                               'maxAppliedForce'])


def remove_all_constraints(client_id=0):
    for cid in get_constraint_ids(client_id):
        p.removeConstraint(cid, physicsClientId=client_id)


def get_constraint_ids(client_id=0):
    """
    getConstraintUniqueId will take a serial index in range 0..getNumConstraints,  and reports the constraint unique id.
    Note that the constraint unique ids may not be contiguous, since you may remove constraints.
    """
    return sorted([p.getConstraintUniqueId(i, physicsClientId=client_id) for i in range(p.getNumConstraints(physicsClientId=client_id))])


def get_constraint_info(constraint, client_id=0):
    # there are four additional arguments
    return ConstraintInfo(*p.getConstraintInfo(constraint, physicsClientId=client_id)[:11])


# Joints

JOINT_TYPES = {
    p.JOINT_REVOLUTE: 'revolute',  # 0
    p.JOINT_PRISMATIC: 'prismatic',  # 1
    p.JOINT_SPHERICAL: 'spherical',  # 2
    p.JOINT_PLANAR: 'planar',  # 3
    p.JOINT_FIXED: 'fixed',  # 4
    p.JOINT_POINT2POINT: 'point2point',  # 5
    p.JOINT_GEAR: 'gear',  # 6
}


def get_num_joints(body, client_id=0):
    return p.getNumJoints(body, physicsClientId=client_id)


def get_joints(body, client_id=0):
    return list(range(get_num_joints(body, client_id=client_id)))


def get_joint(body, joint_or_name):
    if type(joint_or_name) is str:
        return joint_from_name(body, joint_or_name)
    return joint_or_name


JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])


def get_joint_info(body, joint, client_id=0):
    return JointInfo(*p.getJointInfo(body, joint, physicsClientId=client_id))


def get_joints_info(body, joints, client_id=0):
    return [JointInfo(*p.getJointInfo(body, joint, physicsClientId=client_id)) for joint in joints]


def get_joint_name(body, joint, client_id=0):
    return get_joint_info(body, joint, physicsClientId=client_id).jointName.decode('UTF-8')


def get_joint_names(body, client_id=0):
    return [get_joint_name(body, joint, client_id) for joint in get_joints(body, client_id)]


def joint_from_name(body, name, client_id=0):
    for joint in get_joints(body, client_id):
        if get_joint_name(body, joint, client_id) == name:
            return joint
    raise ValueError(body, name)


def has_joint(body, name, client_id=0):
    try:
        joint_from_name(body, name, client_id=client_id)
    except ValueError:
        return False
    return True


def joints_from_names(body, names, client_id=0):
    return tuple(joint_from_name(body, name, client_id=client_id) for name in names)


JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                       'jointReactionForces', 'appliedJointMotorTorque'])


def get_joint_state(body, joint, client_id=0):
    return JointState(*p.getJointState(body, joint, physicsClientId=client_id))


def get_joint_position(body, joint, client_id=0):
    return get_joint_state(body, joint, client_id=client_id).jointPosition


def get_joint_torque(body, joint, client_id=0):
    return get_joint_state(body, joint, client_id=client_id).appliedJointMotorTorque


def get_joint_positions(body, joints=None, client_id=0):
    return list(get_joint_position(body, joint, client_id) for joint in joints)


def set_joint_position(body, joint, value, client_id=0):
    p.resetJointState(body, joint, value, physicsClientId=client_id)


def set_joint_positions(body, joints, values, client_id=0):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        set_joint_position(body, joint, value, client_id=client_id)


def get_configuration(body, client_id=0):
    return get_joint_positions(body, get_movable_joints(body, client_id), client_id)


def set_configuration(body, values, client_id=0):
    set_joint_positions(body, get_movable_joints(body, client_id=client_id), values, client_id=client_id)


def get_full_configuration(body, client_id=0):
    # Cannot alter fixed joints
    return get_joint_positions(body, get_joints(body), client_id=client_id)


def get_joint_type(body, joint, client_id=0):
    return get_joint_info(body, joint, client_id=client_id).jointType


def is_movable(body, joint, client_id=0):
    return get_joint_type(body, joint, client_id=client_id) != p.JOINT_FIXED


def get_movable_joints(body, client_id=0):  # 45 / 87 on pr2
    return [joint for joint in get_joints(body, client_id=client_id) if is_movable(body, joint,client_id=client_id)]


def joint_from_movable(body, index, client_id=0):
    return get_joints(body, client_id)[index]


def is_circular(body, joint, client_id=0):
    # Do not understand what this means
    joint_info = get_joint_info(body, joint, client_id=client_id)
    if joint_info.jointType == p.JOINT_FIXED:
        return False
    return joint_info.jointUpperLimit >= joint_info.jointLowerLimit  # reversed?


def get_joint_limits(body, joint, client_id=0):
    """
    Obtain the limits of a single joint
    :param body: int
    :param joint: int
    :return: (int, int), lower limit and upper limit
    """
    joint_info = get_joint_info(body, joint, client_id=client_id)
    return joint_info.jointLowerLimit, joint_info.jointUpperLimit


def get_joints_limits(body, joints, client_id=0):
    """
    Obtain the limits of a set of joints
    :param body: int
    :param joints: array type
    :return: a tuple of 2 arrays - lower limit and higher limit
    """
    lower_limit = []
    upper_limit = []
    for joint in joints:
        lower_limit.append(get_joint_info(body, joint, client_id=client_id).jointLowerLimit)
        upper_limit.append(get_joint_info(body, joint, client_id=client_id).jointUpperLimit)
    return lower_limit, upper_limit


def get_min_limit(body, joint, client_id=0):
    return get_joint_limits(body, joint, client_id=client_id)[0]


def get_max_limit(body, joint, client_id=0):
    return get_joint_limits(body, joint, client_id=client_id)[1]


def get_max_velocity(body, joint, client_id=0):
    return get_joint_info(body, joint, client_id=client_id).jointMaxVelocity


def get_max_force(body, joint, client_id=0):
    return get_joint_info(body, joint, client_id=client_id).jointMaxForce


def get_joint_q_index(body, joint, client_id=0):
    return get_joint_info(body, joint, client_id=client_id).qIndex


def get_joint_v_index(body, joint, client_id=0):
    return get_joint_info(body, joint, client_id=client_id).uIndex


def get_joint_axis(body, joint, client_id=0):
    return get_joint_info(body, joint, client_id=client_id).jointAxis


def get_joint_parent_frame(body, joint, client_id=0):
    joint_info = get_joint_info(body, joint, client_id=client_id)
    return joint_info.parentFramePos, joint_info.parentFrameOrn


def violates_limit(body, joint, value, client_id=0):
    if not is_circular(body, joint, client_id=client_id):
        lower, upper = get_joint_limits(body, joint, client_id=client_id)
        if (value < lower) or (upper < value):
            return True
    return False


def violates_limits(body, joints, values, client_id=0):
    return any(violates_limit(body, joint, value, client_id=client_id) for joint, value in zip(joints, values))


def wrap_angle(theta): # change to -np.pi ~ np.pi
    return (theta + np.pi) % (2 * np.pi) - np.pi


def circular_difference(theta2, theta1):
    return wrap_angle(theta2 - theta1)


def wrap_joint(body, joint, value, client_id=0):
    if is_circular(body, joint, client_id=client_id):
        return wrap_angle(value)
    return value


def get_difference_fn(body, joints, client_id=0):
    def fn(q2, q1):
        difference = []
        for joint, value2, value1 in zip(joints, q2, q1):
            difference.append(circular_difference(value2, value1)
                              if is_circular(body, joint, client_id=client_id) else (value2 - value1))
        return list(difference)

    return fn


def get_refine_fn(body, joints, num_steps=0, client_id=0): # no use
    difference_fn = get_difference_fn(body, joints, client_id=client_id)
    num_steps = num_steps + 1

    def fn(q1, q2):
        q = q1
        for i in range(num_steps):
            q = tuple((1. / (num_steps - i)) * np.array(difference_fn(q2, q)) + q)
            yield q

    return fn


# Body and base

BodyInfo = namedtuple('BodyInfo', ['base_name', 'body_name'])


# Bodies

def get_bodies(client_id=0):
    return [p.getBodyUniqueId(i, physicsClientId=client_id)
            for i in range(p.getNumBodies(physicsClientId=client_id))]


def get_body_info(body, client_id=0):
    return BodyInfo(*p.getBodyInfo(body, physicsClientId=client_id))


def get_base_name(body, client_id=0):
    return get_body_info(body, client_id=client_id).base_name.decode(encoding='UTF-8')


def get_body_name(body, client_id=0):
    return get_body_info(body, client_id).body_name.decode(encoding='UTF-8')


def get_name(body, client_id=0):
    name = get_body_name(body, client_id=client_id)
    if name == '':
        name = 'body'
    return '{}{}'.format(name, int(body))


def has_body(name, client_id=0):
    try:
        body_from_name(name, client_id=client_id)
    except ValueError:
        return False
    return True


def body_from_name(name, client_id=0):
    for body in get_bodies(client_id=client_id):
        if get_body_name(body, client_id=client_id) == name:
            return body
    raise ValueError(name)


def remove_body(body, client_id=0):
    return p.removeBody(body, physicsClientId=client_id)


def get_body_quat(body, client_id=0):
    return get_body_pose(body, client_id=client_id)[1]  # [x,y,z,w]


def set_pose(body, pose, client_id=0):
    (point, quat) = pose
    p.resetBasePositionAndOrientation(body, point, quat, physicsClientId=client_id)


def set_center_pose(body, center_pose, Center2Base_pose, client_id=0):
    Base_pose = multiply_multi_transforms(center_pose, Center2Base_pose)
    (point, quat) = Base_pose
    p.resetBasePositionAndOrientation(body, point, quat, physicsClientId=client_id)


def is_rigid_body(body, client_id=0):
    for joint in get_joints(body, client_id=client_id):
        if is_movable(body, joint, client_id=client_id):
            return False
    return True


def is_fixed_base(body):
    return get_mass(body) == STATIC_MASS


def dump_body(body):
    print('Body id: {} | Name: {} | Rigid: {} | Fixed: {}'.format(
        body, get_body_name(body), is_rigid_body(body), is_fixed_base(body)))
    for joint in get_joints(body):
        print('Joint id: {} | Name: {} | Type: {} | Circular: {} | Limits: {}'.format(
            joint, get_joint_name(body, joint), JOINT_TYPES[get_joint_type(body, joint)],
            is_circular(body, joint), get_joint_limits(body, joint)))
    print('Link id: {} | Name: {} | Mass: {}'.format(-1, get_base_name(body), get_mass(body)))
    for link in get_links(body):
        print('Link id: {} | Name: {} | Parent: {} | Mass: {}'.format(
            link, get_link_name(body, link), get_link_name(body, get_link_parent(body, link)),
            get_mass(body, link)))
        # print(get_joint_parent_frame(body, link))
        # print(map(get_data_geometry, get_visual_data(body, link)))
        # print(map(get_data_geometry, get_collision_data(body, link)))


def dump_world():
    for body in get_bodies():
        dump_body(body)
        print()


def remove_all_bodies():
    for i in get_body_ids():
        p.removeBody(i)


def get_body_infos():
    """ Return all body info in a list """
    return [get_body_info(i) for i in get_body_ids()]


def get_body_names():
    """ Return all body names in a list """
    return [bi.body_name for bi in get_body_infos()]


def get_body_id(name):
    return get_body_names().index(name)


def get_body_ids():
    return sorted([p.getBodyUniqueId(i) for i in range(p.getNumBodies())])


def get_body_pose(body, client_id=0):
    raw = p.getBasePositionAndOrientation(body, physicsClientId=client_id)
    position = list(raw[0])
    orn = list(raw[1])
    return [position, orn]


def get_body_mesh_num(body, client_id=0):
    link_num = get_num_links(body, client_id=client_id) # add baselink
    num_mesh_parts = sum([len(get_link_collision_shape(body, link_index, client_id=client_id)) for link_index in range(-1, link_num-1)])
    return num_mesh_parts


# Control

def control_joint(body, joint, value, client_id=0):
    return p.setJointMotorControl2(bodyUniqueId=body,
                                   jointIndex=joint,
                                   controlMode=p.POSITION_CONTROL,
                                   targetPosition=value,
                                   targetVelocity=0,
                                   maxVelocity=get_max_velocity(body, joint),
                                   force=get_max_force(body, joint),
                                   physicsClientId=client_id)


def control_joints(body, joints, positions, control_type='hard', client_id=0):
    forces = [get_max_force(body, joint) // 15 for joint in joints] if control_type == 'limited' else [100000] * len(joints)
    return p.setJointMotorControlArray(bodyUniqueId=body, 
                                       jointIndices=joints, 
                                       controlMode=p.POSITION_CONTROL,
                                       targetPositions=positions,
                                       targetVelocities=[0.0] * len(joints),
                                       forces=forces, physicsClientId=client_id)


def forward_kinematics(body, joints, positions, eef_link=None, client_id=0):
    eef_link = get_num_joints(body, client_id=client_id) - 1 if eef_link is None else eef_link
    old_positions = get_joint_positions(body, joints, client_id=client_id)
    set_joint_positions(body, joints, positions, client_id=client_id)
    eef_pose = get_link_pose(body, eef_link, client_id=client_id)
    set_joint_positions(body, joints, old_positions)
    return eef_pose


def inverse_kinematics(body, eef_link, position, orientation=None):
    if orientation is None:
        jv = p.calculateInverseKinematics(bodyUniqueId=body,
                                          endEffectorLinkIndex=eef_link,
                                          targetPosition=position,
                                          residualThreshold=1e-3)
    else:
        jv = p.calculateInverseKinematics(bodyUniqueId=body,
                                          endEffectorLinkIndex=eef_link,
                                          targetPosition=position,
                                          targetOrientation=orientation,
                                          residualThreshold=1e-3)
    return jv


# Links

BASE_LINK = -1
STATIC_MASS = 0

get_num_links = get_num_joints
get_links = get_joints


def get_link_name(body, link, client_id=0):
    if link == BASE_LINK:
        return get_base_name(body, client_id=client_id)
    return get_joint_info(body, link, client_id=client_id).linkName.decode('UTF-8')


def get_link_parent(body, link, client_id=0):
    if link == BASE_LINK:
        return None
    return get_joint_info(body, link, client_id=client_id).parentIndex


def link_from_name(body, name, client_id=0):
    if name == get_base_name(body, client_id=client_id):
        return BASE_LINK
    for link in get_joints(body, client_id=client_id):
        if get_link_name(body, link, client_id=client_id) == name:
            return link
    raise ValueError(body, name)


def has_link(body, name):
    try:
        link_from_name(body, name)
    except ValueError:
        return False
    return True


LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                     'localInertialFramePosition', 'localInertialFrameOrientation',
                                     'worldLinkFramePosition', 'worldLinkFrameOrientation'])


def get_link_state(body, link, client_id=0):
    return LinkState(*p.getLinkState(body, link, physicsClientId=client_id))


def get_link_collision_shape(body, link=-1, client_id=0):
    return p.getCollisionShapeData(body, link, physicsClientId=client_id)


def get_com_pose(body, link, client_id=0):  # COM = center of mass
    if link == BASE_LINK:
        return get_body_pose(body, client_id=client_id)
    link_state = get_link_state(body, link, client_id=client_id)
    return list(link_state.linkWorldPosition), list(link_state.linkWorldOrientation)


def get_link_inertial_pose(body, link, client_id=0):
    if link == BASE_LINK:
        return get_body_pose(body, client_id=client_id)
    link_state = get_link_state(body, link, client_id=client_id)
    return link_state.localInertialFramePosition, link_state.localInertialFrameOrientation


def get_link_pose(body, link, client_id=0):
    if link == BASE_LINK:
        return get_body_pose(body, client_id=client_id)
    # if set to 1 (or True), the Cartesian world position/orientation will be recomputed using forward kinematics.
    link_state = get_link_state(body, link, client_id=client_id)
    return [list(link_state.worldLinkFramePosition), list(link_state.worldLinkFrameOrientation)]


def get_all_link_parents(body, client_id=0):
    return {link: get_link_parent(body, link, client_id=client_id) for link in get_links(body, client_id=client_id)}


def get_all_link_children(body, client_id=0):
    children = {}
    for child, parent in get_all_link_parents(body, client_id=client_id).items():
        if parent not in children:
            children[parent] = []
        children[parent].append(child)
    return children


def get_link_children(body, link, client_id=0):
    children = get_all_link_children(body, client_id=client_id)
    return children.get(link, [])


def get_link_ancestors(body, link, client_id=0):
    parent = get_link_parent(body, link, client_id=client_id)
    if parent is None:
        return []
    return get_link_ancestors(body, parent, client_id=client_id) + [parent]


def get_joint_ancestors(body, link, client_id=0):
    return get_link_ancestors(body, link, client_id=client_id) + [link]


def get_movable_joint_ancestors(body, link):
    return list(filter(lambda j: is_movable(body, j), get_joint_ancestors(body, link)))


def get_link_descendants(body, link, client_id=0):
    descendants = []
    for child in get_link_children(body, link, client_id=client_id):
        descendants.append(child)
        descendants += get_link_descendants(body, child, client_id=client_id)
    return descendants


def are_links_adjacent(body, link1, link2, client_id=0):
    return (get_link_parent(body, link1, client_id=client_id) == link2) or \
           (get_link_parent(body, link2, client_id=client_id) == link1)


def get_adjacent_links(body):
    adjacent = set()
    for link in get_links(body):
        parent = get_link_parent(body, link)
        adjacent.add((link, parent))
        # adjacent.add((parent, link))
    return adjacent


def get_adjacent_fixed_links(body):
    return list(filter(lambda item: not is_movable(body, item[0]),
                       get_adjacent_links(body)))


def get_fixed_links(body):
    edges = defaultdict(list)
    for link, parent in get_adjacent_fixed_links(body):
        edges[link].append(parent)
        edges[parent].append(link)
    visited = set()
    fixed = set()
    for initial_link in get_links(body):
        if initial_link in visited:
            continue
        cluster = [initial_link]
        queue = deque([initial_link])
        visited.add(initial_link)
        while queue:
            for next_link in edges[queue.popleft()]:
                if next_link not in visited:
                    cluster.append(next_link)
                    queue.append(next_link)
                    visited.add(next_link)
        fixed.update(product(cluster, cluster))
    return fixed


DynamicsInfo = namedtuple('DynamicsInfo', ['mass', 'lateral_friction',
                                           'local_inertia_diagonal', 'local_inertial_pos', 'local_inertial_orn',
                                           'restitution', 'rolling_friction', 'spinning_friction',
                                           'contact_damping', 'contact_stiffness', 'body_type', 'collision_margin'])


def get_dynamics_info(body, link=BASE_LINK, client_id=0):
    return DynamicsInfo(*p.getDynamicsInfo(body, link, physicsClientId=client_id))


def get_mass(body, link=BASE_LINK, client_id=0):
    return get_dynamics_info(body, link, client_id=client_id).mass


def set_mass(body, mass, link=BASE_LINK, client_id=0):
    p.changeDynamics(body, link, mass=mass, physicsClientId=client_id)


def fix_base(body, client_id=0):
    set_mass(body, STATIC_MASS, BASE_LINK, client_id=client_id)


def get_joint_inertial_pose(body, joint):
    dynamics_info = get_dynamics_info(body, joint)
    return dynamics_info.local_inertial_pos, dynamics_info.local_inertial_orn


# Camera

CameraInfo = namedtuple('CameraInfo', ['width', 'height',
                                       'viewMatrix', 'projectionMatrix', 'cameraUp',
                                       'cameraForward', 'horizontal', 'vertical',
                                       'yaw', 'pitch', 'dist', 'target'])


def reset_camera(yaw=50.0, pitch=-35.0, dist=5.0, target=(0.0, 0.0, 0.0), client_id=0):
    p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=target, physicsClientId=client_id)


def get_camera():
    return CameraInfo(*p.getDebugVisualizerCamera())


# Visualization

def create_frame_marker(pose=((0, 0, 0), (0, 0, 0, 1)),
                        x_color=np.array([1, 0, 0]),
                        y_color=np.array([0, 1, 0]),
                        z_color=np.array([0, 0, 1]),
                        line_length=0.1,
                        line_width=2,
                        life_time=0,
                        replace_frame_id=None,
                        client_id=0):
    """
    Create a pose marker that identifies a position and orientation in space with 3 colored lines.
    """
    position = np.array(pose[0])
    orientation = np.array(pose[1])

    pts = np.array([[0, 0, 0], [line_length, 0, 0], [0, line_length, 0], [0, 0, line_length]])
    rotIdentity = np.array([0, 0, 0, 1])
    po, _ = p.multiplyTransforms(position, orientation, pts[0, :], rotIdentity)
    px, _ = p.multiplyTransforms(position, orientation, pts[1, :], rotIdentity)
    py, _ = p.multiplyTransforms(position, orientation, pts[2, :], rotIdentity)
    pz, _ = p.multiplyTransforms(position, orientation, pts[3, :], rotIdentity)

    if replace_frame_id is not None:
        x_id = p.addUserDebugLine(po, px, x_color, line_width, life_time, replaceItemUniqueId=replace_frame_id[0], physicsClientId=client_id)
        y_id = p.addUserDebugLine(po, py, y_color, line_width, life_time, replaceItemUniqueId=replace_frame_id[1], physicsClientId=client_id)
        z_id = p.addUserDebugLine(po, pz, z_color, line_width, life_time, replaceItemUniqueId=replace_frame_id[2], physicsClientId=client_id)
    else:
        x_id = p.addUserDebugLine(po, px, x_color, line_width, life_time, physicsClientId=client_id)
        y_id = p.addUserDebugLine(po, py, y_color, line_width, life_time, physicsClientId=client_id)
        z_id = p.addUserDebugLine(po, pz, z_color, line_width, life_time, physicsClientId=client_id)
    frame_id = (x_id, y_id, z_id)
    return frame_id


def create_arrow_marker(pose=((0, 0, 0), (0, 0, 0, 1)),
                        line_length=0.1,
                        line_width=2,
                        life_time=0,
                        color_index=0,
                        raw_color=None,
                        replace_frame_id=None,
                        client_id=0):
    """
    Create an arrow marker that identifies the z-axis of the end effector frame. Add a dot towards the positive direction.
    """

    color = raw_color if raw_color is not None else rgb_colors_1[color_index % len(rgb_colors_1)]

    po, _ = p.multiplyTransforms(pose[0], pose[1], [0, 0, 0], [0, 0, 0, 1])
    pz, _ = p.multiplyTransforms(pose[0], pose[1], [0, 0, line_length], [0, 0, 0, 1])
    pz_extend1, _ = p.multiplyTransforms(pz, pose[1], [0, line_length*0.2, -line_length*0.2], [0, 0, 0, 1])
    pz_extend2, _ = p.multiplyTransforms(pz, pose[1], [0, -line_length*0.2, -line_length*0.2], [0, 0, 0, 1])

    if replace_frame_id is not None:
        z_id = p.addUserDebugLine(po, pz, color, line_width, life_time,
                                  replaceItemUniqueId=replace_frame_id[0], physicsClientId=client_id)
        z_extend_id1 = p.addUserDebugLine(pz, pz_extend1, color, line_width, life_time,
                                          replaceItemUniqueId=replace_frame_id[1], physicsClientId=client_id)
        z_extend_id2 = p.addUserDebugLine(pz, pz_extend2, color, line_width, life_time,
                                          replaceItemUniqueId=replace_frame_id[2], physicsClientId=client_id)
    else:
        z_id = p.addUserDebugLine(po, pz, color, line_width, life_time, physicsClientId=client_id)
        z_extend_id1 = p.addUserDebugLine(pz, pz_extend1, color, line_width, life_time, physicsClientId=client_id)
        z_extend_id2 = p.addUserDebugLine(pz, pz_extend2, color, line_width, life_time, physicsClientId=client_id)
    frame_id = (z_id, z_extend_id1, z_extend_id2)
    return frame_id

def create_arrow_marker_points(pose=((0, 0, 0), (0, 0, 0, 1)),
                        line_length=0.1,
                        life_time=0,
                        color_index=0,
                        raw_color=None,
                        replace_markers_id=None,
                        client_id=0):
    """
    Create an arrow marker that identifies the z-axis of the end effector frame. Add a dot towards the positive direction.
    """
    if replace_markers_id: remove_path(replace_markers_id)

    color = raw_color if raw_color is not None else rgb_colors_1[color_index % len(rgb_colors_1)]
    color = color.tolist() + [1] # transfer to rgba list

    po, _ = p.multiplyTransforms(pose[0], pose[1], [0, 0, 0], [0, 0, 0, 1])
    pz, _ = p.multiplyTransforms(pose[0], pose[1], [0, 0, line_length], [0, 0, 0, 1])
    pz_extend1, _ = p.multiplyTransforms(pz, pose[1], [0, line_length*0.2, -line_length*0.2], [0, 0, 0, 1])
    pz_extend2, _ = p.multiplyTransforms(pz, pose[1], [0, -line_length*0.2, -line_length*0.2], [0, 0, 0, 1])

    num_points = 20
    line1 = np.linspace(po, pz, num_points).tolist()
    line2 = np.linspace(pz, pz_extend1, num_points).tolist()
    line3 = np.linspace(pz, pz_extend2, num_points).tolist()
    all_box = line1 + line2 + line3
    markers_id = []
    for pos in all_box:
        markers_id.append(draw_box_body(pos, rgba_color=color, client_id=client_id))
    return markers_id


def change_obj_color(obj_id, linklist=None, rgba_color=[0, 0, 0, 1], client_id=0):
    num_links = get_num_links(obj_id, client_id=client_id)
    linklist = list(range(num_links)) + [-1] if linklist is None else linklist
    for link in linklist:
        if link > num_links: continue
        p.changeVisualShape(obj_id, link, rgbaColor=rgba_color, physicsClientId=client_id)


class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)

    def get_rgb(self, val):
        return self.cmap(self.norm(val))[:3]


def rgb(value, minimum=-1, maximum=1):
    """ for the color map https://stackoverflow.com/questions/20792445/calculate-rgb-value-for-a-range-of-values-to-create-heat-map """
    assert minimum <= value <= maximum
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    return r / 255., g / 255., b / 255.


def plot_heatmap_bar(cmap_name, vmin=-1, vmax=-1):
    plt.figure(num=-1, figsize=(10, 2))
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cb = mpl.colorbar.ColorbarBase(plt.gca(), cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_label('Heatmap bar')
    plt.pause(0.001)


# https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
rgb_colors_255 = [(230, 25, 75),  # red
                  (60, 180, 75),  # green
                  (255, 225, 25),  # yello
                  (0, 130, 200),  # blue
                  (245, 130, 48),  # orange
                  (145, 30, 180),  # purple
                  (70, 240, 240),  # cyan
                  (240, 50, 230),  # magenta
                  (210, 245, 60),  # lime
                  (250, 190, 190),  # pink
                  (0, 128, 128),  # teal
                  (230, 190, 255),  # lavender
                  (170, 110, 40),  # brown
                  (255, 250, 200),  # beige
                  (128, 0, 0),  # maroon
                  (170, 255, 195),  # lavender
                  (128, 128, 0),  # olive
                  (255, 215, 180),  # apricot
                  (0, 0, 128),  # navy
                  (128, 128, 128),  # grey
                  (0, 0, 0),  # white
                  (255, 255, 255)]  # black

rgb_colors_1 = np.array(rgb_colors_255) / 255.


def draw_line(start_pos, end_pos, rgb_color=(1, 0, 0), width=15, lifetime=0, client_id=0):
    lid = p.addUserDebugLine(lineFromXYZ=start_pos,
                             lineToXYZ=end_pos,
                             lineColorRGB=rgb_color,
                             lineWidth=width,
                             lifeTime=lifetime,
                             physicsClientId=client_id)
    return lid


def draw_circle_around_z_axis(centre, radius, rgb_color=(1, 0, 0), width=3, lifetime=0, num_divs=100, client_id=0):
    points = np.array(centre) + radius * np.array(
        [(np.cos(ang), np.sin(ang), 0) for ang in np.linspace(0, 2 * np.pi, num_divs)])
    lids = []
    for i in range(len(points) - 1):
        start_pos = points[i]
        end_pos = points[i + 1]
        lid = p.addUserDebugLine(lineFromXYZ=start_pos,
                                 lineToXYZ=end_pos,
                                 lineColorRGB=rgb_color,
                                 lineWidth=width,
                                 lifeTime=lifetime,
                                 physicsClientId=client_id)
        lids.append(lid)
    return lids

def draw_sphere_body(position, radius=0.01, rgba_color=[0.5,0.5,1,0.8], client_id=0):
    vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba_color, physicsClientId=client_id)
    body_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id, physicsClientId=client_id)
    return body_id

def draw_box_body(position, orientation=[0, 0, 0, 1], halfExtents=[0.003, 0.003, 0.003], rgba_color=[1, 0, 0, 1], client_id=0):
    vs_id = p.createVisualShape(p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=rgba_color, physicsClientId=client_id)
    body_id = p.createMultiBody(basePosition=position, baseOrientation=orientation, 
                                baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id, physicsClientId=client_id)
    return body_id

def create_box_body(position, orientation=[0, 0, 0, 1], halfExtents=[0.003, 0.003, 0.003], rgba_color=[1, 0, 0, 1], mass=0., client_id=0):
    vs_id = p.createVisualShape(p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=rgba_color, physicsClientId=client_id)
    cs_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents, physicsClientId=client_id)
    box_id = p.createMultiBody(baseMass=mass, basePosition=position, baseOrientation=orientation,
                                      baseCollisionShapeIndex=cs_id, baseVisualShapeIndex=vs_id,
                                      physicsClientId=client_id)
    return box_id


def draw_path(eef_poses, client_id=0):
    markers_id = []
    markers_id.append(draw_sphere_body(eef_poses[0], radius=0.03, rgba_color=[1, 0, 0, 0.8], client_id=client_id)) # draw start point by red
    for i in range(1, len(eef_poses)-1):
        markers_id.append(draw_sphere_body(eef_poses[i], client_id=client_id))
    markers_id.append(draw_sphere_body(eef_poses[-1], radius=0.03, rgba_color=[0, 1, 0, 0.8], client_id=client_id)) # draw end point by green
    return markers_id

def draw_conveyor_path(discretized_trajectory, num_plot_points=None,
                       reachability_value=None, sphere_path=False, client_id=0):
    num_plot_points = num_plot_points if num_plot_points else len(discretized_trajectory)
    idx = np.linspace(0, len(discretized_trajectory) - 1, num_plot_points).astype(int)
    for i in range(len(idx) - 1):
        pos1 = discretized_trajectory[idx[i]][0]
        pos2 = discretized_trajectory[idx[i + 1]][0]
        if reachability_value is not None and reachability_value[idx[i]] and reachability_value[idx[i+1]]: piece_len_color = (reachability_value[idx[i]], 0, 0)
        else: piece_len_color = (0, 0, 0)
        draw_line(pos1, pos2, rgb_color=piece_len_color, client_id=client_id)
        if sphere_path:
            draw_sphere_body(pos1, radius=0.008, rgba_color=list(piece_len_color)+[0.8], client_id=client_id)  # to record images


def remove_path(marker_ids, client_id=0):
    for id in marker_ids:
        p.removeBody(id, physicsClientId=client_id)

def remove_marker(marker_id, client_id=0):
    p.removeUserDebugItem(marker_id, physicsClientId=client_id)


def remove_markers(marker_ids, client_id=0):
    for i in marker_ids:
        p.removeUserDebugItem(i, physicsClientId=client_id)


def remove_all_markers(client_id=0):
    p.removeAllUserDebugItems(physicsClientId=client_id)


# RRT

def all_collision(client_id=0, **kwargs):
    bodies = get_bodies(client_id=client_id)
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            if pairwise_collision(bodies[i], bodies[j], client_id=client_id, **kwargs):
                return True
    return False


def get_distance_fn(body, joints, weights=None, client_id=0):
    if weights is None:
        weights = 1 * np.ones(len(joints))
    difference_fn = get_difference_fn(body, joints, client_id=client_id)

    def fn(q1, q2):
        diff = np.array(difference_fn(q2, q1))
        return np.sqrt(np.dot(weights, diff * diff))
    return fn

def get_extend_fn(body, joints, resolutions=None, client_id=0):
    if resolutions is None:
        resolutions = 0.05 * np.ones(len(joints))
    difference_fn = get_difference_fn(body, joints, client_id=client_id)

    def fn(q1, q2):
        steps = np.abs(np.divide(difference_fn(q2, q1), resolutions))
        num_steps = ceil(max(steps))
        waypoints = []
        diffs = difference_fn(q2, q1)
        for i in range(1, num_steps + 1):
            waypoints.append(list((i / float(num_steps)) * np.array(diffs) + q1))
        return waypoints
    return fn

def get_goal_test_fn(goal_conf, atol=0.001, rtol=0):
    def fn(conf):
        return np.allclose(conf, goal_conf, atol=atol, rtol=rtol)
    return fn

def get_moving_links(body, moving_joints, client_id=0):
    moving_links = list(moving_joints)
    for link in moving_joints:
        moving_links += get_link_descendants(body, link, client_id=client_id)
    return list(set(moving_links))

def get_moving_pairs(body, moving_joints, client_id=0):
    moving_links = get_moving_links(body, moving_joints, client_id=client_id)
    for i in range(len(moving_links)):
        link1 = moving_links[i]
        ancestors1 = set(get_joint_ancestors(body, link1)) & set(moving_joints)
        for j in range(i + 1, len(moving_links)):
            link2 = moving_links[j]
            ancestors2 = set(get_joint_ancestors(body, link2)) & set(moving_joints)
            if ancestors1 != ancestors2:
                yield link1, link2

def get_self_link_pairs(body, joints, disabled_collisions=set(), client_id=0):
    moving_links = get_moving_links(body, joints, client_id=client_id)
    fixed_links = list(set(get_links(body)) - set(moving_links))
    check_link_pairs = list(product(moving_links, fixed_links))
    if True:
        check_link_pairs += list(get_moving_pairs(body, joints, client_id=client_id))
    else:
        check_link_pairs += list(combinations(moving_links, 2))
    check_link_pairs = list(filter(lambda pair: not are_links_adjacent(body, *pair, client_id=client_id), check_link_pairs))
    check_link_pairs = list(filter(lambda pair: (pair not in disabled_collisions) and
                                                (pair[::-1] not in disabled_collisions), check_link_pairs))
    return check_link_pairs

def get_sample_fn(body, joints, rng=None, client_id=0):
    rng = rng if rng else np.random
    def fn():
        values = []
        for joint in joints:
            limits = CIRCULAR_LIMITS if is_circular(body, joint, client_id=client_id) \
                else get_joint_limits(body, joint, client_id=client_id)
            values.append(rng.uniform(*limits))
        return list(values)
    return fn


def getObjVelocity(body, to_array=True, client_id=0):
    linear_v, ang_v = p.getBaseVelocity(body, physicsClientId=client_id)
    if to_array: return np.array(linear_v + ang_v)
    else: return list(linear_v + ang_v)


def pairwise_collision(body1, body2, max_distance=MAX_DISTANCE, client_id=0):  # 10000
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance, physicsClientId=client_id)) != 0  # getContactPoints

def pairwise_link_collision(body1, link1, body2, link2, max_distance=MAX_DISTANCE, client_id=0):  # 10000
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  linkIndexA=link1, linkIndexB=link2, physicsClientId=client_id)) != 0  # getContactPoints

def single_collision(body1, **kwargs):
    for body2 in get_bodies():
        if (body1 != body2) and pairwise_collision(body1, body2, **kwargs):
            return True
    return False

def pose_difference(pose1, pose2):
    pose1, pose2 = np.array(pose1), np.array(pose2)
    return np.linalg.norm(pose1-pose2)

def pose2d_matrix(pose2d):
    matrix = np.zeros((4, 4))
    position, orientation = pose2d
    orientation_m = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    matrix[:3, :3] = orientation_m
    matrix[:3, 3] = position
    matrix[3, 3] = 1
    return matrix

def multiply_multi_transforms(*args):
    assert len(args) >= 2, 'multi transforms need at least 2 pose'
    if len(args) == 2:
        pose1, pose2 = args
        return p.multiplyTransforms(*pose1, *pose2)
    elif len(args) > 2:
        pose1, pose2 = args[0], args[1]
        accumulate_pose = p.multiplyTransforms(*pose1, *pose2)
        for i in range(2, len(args)):
            accumulate_pose = p.multiplyTransforms(*accumulate_pose, *args[i])
        return accumulate_pose
    

def quat_apply(quat, vec):
    return p.multiplyTransforms([0, 0, 0.], quat, vec, [0, 0, 0, 1])[0]
    

def getQuaternionFromAxisAngle(axis, angle):
    # Normalize the axis
    axis_length = sqrt(sum([i**2 for i in axis]))
    normalized_axis = [i/axis_length for i in axis]
    
    # Calculate the sine and cosine of half the angle
    sin_theta_over_2 = sin(angle / 2.0)
    cos_theta_over_2 = cos(angle / 2.0)
    
    # Compute the quaternion
    w = cos_theta_over_2
    x = normalized_axis[0] * sin_theta_over_2
    y = normalized_axis[1] * sin_theta_over_2
    z = normalized_axis[2] * sin_theta_over_2
    
    return [x, y, z, w]


def getQuaternionFromTwoVectors(v0, v1):
    # Normalize input vectors
    v0_normalized = v0 / np.linalg.norm(v0)
    v1_normalized = v1 / np.linalg.norm(v1)
    
    # Compute the cross product and dot product
    cross_prod = np.cross(v0_normalized, v1_normalized)
    dot_prod = np.dot(v0_normalized, v1_normalized)
    
    # If the vectors are parallel, we need to pick an arbitrary axis
    if np.isclose(dot_prod, -1.0):
        # Vectors are antiparallel, pick an arbitrary perpendicular axis
        if abs(v0_normalized[0]) < abs(v0_normalized[1]) and abs(v0_normalized[0]) < abs(v0_normalized[2]):
            orthogonal_axis = np.array([1, 0, 0])
        elif abs(v0_normalized[1]) < abs(v0_normalized[2]):
            orthogonal_axis = np.array([0, 1, 0])
        else:
            orthogonal_axis = np.array([0, 0, 1])
        
        # Quaternion for 180 degrees rotation around the orthogonal axis
        q = np.array([*orthogonal_axis * np.sin(np.pi / 2), np.cos(np.pi / 2)])  # sin(π/2) = 1, cos(π/2) = 0
    elif np.isclose(dot_prod, 1.0):
        # Vectors are parallel, no rotation needed
        q = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        # General case for non-parallel vectors
        q = np.array([*cross_prod, 1.0 + dot_prod])
        q = q / np.linalg.norm(q)  # Normalize quaternion
    
    return q

######### RoboSensai Specified #########
def sample_pc_from_mesh(mesh_path, mesh_scaling=[1., 1., 1.], num_points=1024, visualize=False):
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # Sample a point cloud from the mesh
    point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

    # Convert the Open3D point cloud to a NumPy array
    point_cloud_np = np.asarray(point_cloud.points) * np.array(mesh_scaling)

    if visualize:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
        o3d.visualization.draw_geometries([pcd])

    return point_cloud_np


def get_link_pc_from_id(obj_id, link_index=-1, min_num_points=1024, use_worldpos=False, rng=None, client_id=0):
    # Pybullet bug: get link pose can only get the joint pose (it ignores the link offset!!)
    world2basejoint = get_body_pose(obj_id, client_id=client_id)
    world2linkjoint = get_link_pose(obj_id, link_index, client_id=client_id) 
    # if use absolute point position, we transform the point from world to link, point position in the worldframe
    # else we use relative point position, which is the point position in the baselink frame (So it will not change for movable objects)
    what2linkjoint = world2linkjoint if use_worldpos else \
                     p.multiplyTransforms(*p.invertTransform(*world2basejoint), *world2linkjoint)
    link_collision_infos = get_link_collision_shape(obj_id, link_index, client_id=client_id)
    link_pc = []
    for i, link_collision_info in enumerate(link_collision_infos):
        if len(link_collision_info)==0: 
            print(f'No collision shape for object {obj_id}, link {link_index}, {i}th part')
            continue

        ## ***Note: linkjoint2linkcenter_pos is already scaled by mesh_scale, so we don't need to scale it again!
        link_type, link_mesh_scale, link_mesh_path, linkjoint2linkcenter_pos, linkjoint2linkcenter_ori = link_collision_info[2:7]

        what2linkpart = p.multiplyTransforms(
            what2linkjoint[0], what2linkjoint[1], 
            linkjoint2linkcenter_pos, linkjoint2linkcenter_ori
        )
        if link_type == p.GEOM_MESH:
            linkpart_pc_local = sample_pc_from_mesh(link_mesh_path, link_mesh_scale, min_num_points)
        elif link_type == p.GEOM_BOX:
            # link_mesh_scale is object dimension if link_type is GEOM_BOX
            object_halfExtents = np.array(link_mesh_scale) / 2
            if rng is not None: linkpart_pc_local = rng.uniform(-object_halfExtents, object_halfExtents, size=(min_num_points, 3))
            else: linkpart_pc_local = np.uniform(-object_halfExtents, object_halfExtents, size=(min_num_points, 3))
        elif link_type == p.GEOM_CYLINDER:
            raise NotImplementedError
        
        linkpart_pc_world = [p.multiplyTransforms(what2linkpart[0], what2linkpart[1], point, [0, 0, 0, 1.])[0] for point in linkpart_pc_local]
        link_pc.extend(linkpart_pc_world)
    return np.array(link_pc)


# TODO: We need to sample each link's point cloud based on the size of the mesh! It is meaningless to oversampling each link with same amount of points then downsampling!
def get_obj_pc_from_id(obj_id, num_points=1024, use_worldpos=False, rng=None, client_id=0):
    # Set all joints to the lower limit before getting bbox and pc; Assumption is lower joints means the object is in the most compact shape!
    obj_joints_num = get_num_joints(obj_id, client_id=client_id)
    joints_lower_limit = [get_joint_limits(obj_id, joint_i, client_id=client_id)[0] for joint_i in range(obj_joints_num)]
    set_joint_positions(obj_id, list(range(obj_joints_num)), joints_lower_limit, client_id=client_id)
    control_joints(obj_id, list(range(obj_joints_num)), joints_lower_limit, client_id=client_id)
    # Get object point cloud
    obj_pc = []; link_num = get_num_links(obj_id, client_id=client_id) + 1 # add baselink
    num_mesh_parts = get_body_mesh_num(obj_id, client_id=client_id)
    min_num_points = ceil(num_points / max(num_mesh_parts, 1)) * 2
    for i, link_index in enumerate(range(-1, link_num-1)):
        link_pc = get_link_pc_from_id(obj_id, link_index, min_num_points=min_num_points, use_worldpos=use_worldpos, rng=rng, client_id=client_id)
        obj_pc.extend(link_pc)
    if rng is not None: return rng.permutation(obj_pc)[:num_points]
    else: return np.random.permutation(obj_pc)[:num_points]


def get_obj_axes_aligned_bbox_from_pc(obj_pc):
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(obj_pc)
    # Another is get_oriented_bounding_box, but no get_half_extent attributes
    bbox = o3d_pc.get_axis_aligned_bounding_box()
    return np.concatenate([bbox.get_center(), [0., 0., 0., 1.], bbox.get_half_extent()])


def visualize_pc(point_cloud_np, zoom=0.8):
    # Visualize the point cloud
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(point_cloud_np)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    bbox = o3d_pc.get_axis_aligned_bounding_box(); bbox.color = np.array([0, 0, 0.])
    # front means the direction pointing to the camera (facing the camera), up means the up translation between camera and world coordinate
    o3d.visualization.draw_geometries([o3d_pc, coord_frame, bbox], lookat=[0.0, 0.0, 0.0], 
                                      front=[-1.0, 0.0, 0.5], up=[0.0, 0.0, 1.0], zoom=zoom)
    

def visualize_pc_lst(pc_lst, zoom=0.8, color=None):
    # Visualize the point cloud with specific color
    if color is None: color = [None] * len(pc_lst)
    else: assert len(color) == len(pc_lst), 'Color list must have the same length as pc list'
    geometries = []
    for i, pc in enumerate(pc_lst):
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(pc)
        if color[i] is not None: 
            o3d_pc.paint_uniform_color(color[i])
        geometries.append(o3d_pc)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    geometries.append(coord_frame)
    o3d.visualization.draw_geometries(geometries, lookat=[0.0, 0.0, 0.0], 
                                      front=[-1.0, 0.0, 0.5], up=[0.0, 0.0, 1.0], zoom=zoom)
    

def get_pc_from_camera(width, height, view_matrix, proj_matrix, client_id=0):
    # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer

    # get a depth image
    # "infinite" depths will have a value close to 1
    image_arr = p.getCameraImage(width=width, 
                                 height=height, 
                                 viewMatrix=view_matrix, 
                                 projectionMatrix=proj_matrix,
                                 physicsClientId=client_id)
    depth = image_arr[3]

    # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    world2pixels = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), np.array(depth).reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    pixels = pixels[z < 0.99]
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(world2pixels, pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]

    return points


def compute_camera_matrix(cameraEyePosition, cameraTargetPosition, cameraUpVector=[0, 0, 1]):
    return p.computeViewMatrix(cameraEyePosition, cameraTargetPosition, cameraUpVector)


def modify_specific_link_in_urdf(file_path, new_rpy=[0., 0., 0.], new_scale=1.0, firstlink=True, specify_link=None):
    new_rpy = ' '.join([str(i) for i in new_rpy])
    new_scale = ' '.join([str(i) for i in [new_scale]*3])
    
    # Parse the URDF file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Find the specified link by name
    for link in root.findall('link'):
        if link.get('name') == specify_link or specify_link is None or firstlink:
            # Modify the origin's xyz attribute in visual and collision tags
            for origin_tag in link.findall('.//origin'):
                origin_tag.set('rpy', new_rpy)

            # # Modify the scale attribute in the geometry tags
            # for geometry_tag in link.findall('.//geometry/mesh'):
            #     geometry_tag.set('scale', new_scale)
                
            if firstlink: break

    # Write the modified tree back to a file
    tree.write(file_path, encoding='utf-8', xml_declaration=True)