from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import torch
import trimesh
import urdf_parser_py.urdf as urdf
from copy import deepcopy

DT = 1 / 60
gym = gymapi.acquire_gym()


def sim_sec(sim, seconds, dt=DT):
	for _ in range(int(seconds/dt)):
		gym.simulate(sim)
	gym.fetch_results(sim, True)
	# refresh tensors
	gym.refresh_actor_root_state_tensor(sim)
	gym.refresh_rigid_body_state_tensor(sim)
	gym.refresh_dof_state_tensor(sim)
	gym.refresh_net_contact_force_tensor(sim)
	gym.refresh_jacobian_tensors(sim)
	gym.refresh_mass_matrix_tensors(sim)


# Control
def set_pose(gym, sim, state_tensor, body_idxs, positions=None, orientations=None, linvels=None, angvels=None, verbose=False):
    if type(body_idxs) == list: # Multiple actors set_pose (required by GPU pipeline)
        for i, actor_body_idxs in enumerate(body_idxs):
            if positions is not None and positions[i] is not None:
                state_tensor[actor_body_idxs, :3] = positions[i]
            if orientations is not None and orientations[i] is not None:
                state_tensor[actor_body_idxs, 3:7] = orientations[i]
            if linvels is not None and linvels[i] is not None:
                state_tensor[actor_body_idxs, 7:10] = linvels[i]
            if angvels is not None and angvels[i] is not None:
                state_tensor[actor_body_idxs, 10:13] = angvels[i]
        body_idxs = torch.cat(body_idxs).unique()
    else: # Single actor set_pose
        if positions is not None:
            state_tensor[body_idxs, :3] = positions
        if orientations is not None:
            state_tensor[body_idxs, 3:7] = orientations
        if linvels is not None:
            state_tensor[body_idxs, 7:10] = linvels
        if angvels is not None:
            state_tensor[body_idxs, 10:13] = angvels

    # A better way to clone rather than torch.tensor(source_tensor)
    tensor_idxs = body_idxs.detach().clone().to(device=state_tensor.device, dtype=torch.int32)
    success = gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(state_tensor), \
                                                      gymtorch.unwrap_tensor(tensor_idxs), len(body_idxs))
    # success = gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(state_tensor))
    if verbose: print('Pose Set:', success)


def control_dof(gym, sim, dof_pos_tensor, env_idxs, control_signals, lower_limits=None, upper_limits=None, verbose=False):
    if lower_limits is not None: control_signals = torch.clamp(control_signals, min=lower_limits)
    if upper_limits is not None: control_signals = torch.clamp(control_signals, max=upper_limits)
    dof_pos_tensor[env_idxs, :] = control_signals
    success = gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_pos_tensor))
    # tensor_idxs = env_idxs.detach().clone().to(device=dof_pos_tensor.device, dtype=torch.int32)
    # success = gym.set_dof_position_target_tensor_indexed(sim, gymtorch.unwrap_tensor(dof_pos_tensor), \
    #                                                      gymtorch.unwrap_tensor(tensor_idxs), len(env_idxs))
    if verbose: print('Dof Set:', success)


def set_dof(gym, sim, dof_pos_tensor, dof_vel_tensor, env_idxs, target_pos, target_vel=None, lower_limits=None, upper_limits=None, verbose=False):
    if lower_limits is not None: target_pos = torch.clamp(target_pos, min=lower_limits)
    if upper_limits is not None: target_pos = torch.clamp(target_pos, max=upper_limits)
    if target_vel is not None: dof_vel_tensor[env_idxs, :] = target_vel
    dof_pos_tensor[env_idxs, :] = target_pos
    dof_tensor = torch.cat([dof_pos_tensor.view(-1, 1), dof_vel_tensor.view(-1, 1)], dim=1)
    success = gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_tensor))


# Query
def get_pose(state_tensor, body_idxs, decimals=4):
    poses = state_tensor[body_idxs, :7]
    velocity = state_tensor[body_idxs, 7:]
    if decimals is not None:
        poses = torch.round(poses, decimals=decimals)
        velocity = torch.round(velocity, decimals=decimals)
    return poses, velocity


def get_dof(dof_tensor, body_idxs):
    return dof_tensor[body_idxs, :]


def get_force(force_tensor, body_idxs):
    return force_tensor[body_idxs, :]


def get_contact_points(env, finger_index=[4, 5], filter_mask=None): # We assume only contact one thing for each finger at each step / And these two finger will not contact each other
    contacts = gym.get_env_rigid_contacts(env)
    finger_red_idx, finger_green_idx = finger_index
    finger_contact_info = [None, None]; detectable = True
    if filter_mask is not None: filter_start, filter_end = filter_mask
    for contact in contacts: # We are in the local frame, might need to transform to the world frame
        contact_body_idxs = [contact['body0'], contact['body1']]
        if finger_red_idx in contact_body_idxs:
            order_idx = contact_body_idxs.index(finger_red_idx)
            contact_pos, contact_normal = contact[f'localPos{order_idx}'], contact['normal']
            local_pose, normal_direction = cvtVoidXyzToList(contact_pos), cvtVoidXyzToList(contact_normal, (-1) ** order_idx)
            if filter_mask is not None:
                for i in range(len(local_pose)):
                    detectable = local_pose[i] <= min(filter_start[i], filter_end[i]) or local_pose[i] >= max(filter_start[i], filter_end[i])
                    if detectable == True: break # early stop, this contact point is out of filter area
            finger_contact_info[0] = [local_pose, normal_direction] if detectable else None
        if finger_green_idx in contact_body_idxs:
            order_idx = contact_body_idxs.index(finger_green_idx)
            contact_pos, contact_normal = contact[f'localPos{order_idx}'], contact['normal']
            local_pose, normal_direction = cvtVoidXyzToList(contact_pos), cvtVoidXyzToList(contact_normal, (-1) ** order_idx)
            if filter_mask is not None:
                for i in range(len(local_pose)):
                    detectable = local_pose[i] <= min(filter_start[i], filter_end[i]) or local_pose[i] >= max(filter_start[i], filter_end[i])
                    if detectable == True: break # early stop, this contact point is out of filter area
            finger_contact_info[1] = [local_pose, normal_direction] if detectable else None
        if None not in finger_contact_info: break # Two joints contact info have been filled
    return finger_contact_info


def get_contact_points_single(env, finger_index=4, filter_mask=None): # We assume only contact one thing for each finger at each step / And these two finger will not contact each other
    contacts = gym.get_env_rigid_contacts(env)
    finger_contact_info = None; detectable = True
    if filter_mask is not None: filter_start, filter_end = filter_mask
    for contact in contacts: # We are in the local frame, might need to transform to the world frame
        contact_body_idxs = [contact['body0'], contact['body1']]
        if finger_index in contact_body_idxs:
            order_idx = contact_body_idxs.index(finger_index)
            contact_pos, contact_normal = contact[f'localPos{order_idx}'], contact['normal']
            local_pose, normal_direction = cvtVoidXyzToList(contact_pos), cvtVoidXyzToList(contact_normal, (-1) ** order_idx)
            if filter_mask is not None:
                for i in range(len(local_pose)):
                    detectable = local_pose[i] <= min(filter_start[i], filter_end[i]) or local_pose[i] >= max(filter_start[i], filter_end[i])
                    if detectable == True: break # early stop, this contact point is out of filter area
            finger_contact_info = [local_pose, normal_direction] if detectable else None
        if finger_contact_info != None: break # Two joints contact info have been filled
    return finger_contact_info


def get_link_names(env, body_handle):
    return gym.get_actor_rigid_body_names(env, body_handle)
    

def get_num_of_actors(sim):
    return gym.get_sim_actor_count(sim)


def get_envs_images(sim, env_handles, camera_handles, image_type):
    images = []
    for env_handle, camera_handle in zip(env_handles, camera_handles):
        camera_image = gym.get_camera_image(sim, env_handle, camera_handle, image_type)
        image_height, image_width = camera_image.shape
        if image_type == gymapi.IMAGE_COLOR:
            camera_image = camera_image.reshape(image_height, image_width//4, 4)
            camera_image = camera_image[:, :, :3] # remove alpha channel
        images.append(camera_image)
    return np.stack(images)


def get_envs_images_tensor(sim, camera_tensors):
    gym.start_access_image_tensors(sim)
    rgbd_tensors = []
    for rgba_tensor, depth_tensor in camera_tensors:
        image_height, image_width = rgba_tensor.shape
        rgba_tensor = rgba_tensor.reshape(image_height, image_width//4, 4)
        rgb_tensor = rgba_tensor[:, :, :3]
        rgbd_tensor = torch.cat([rgb_tensor, depth_tensor.unsqueeze(-1)], dim=-1)
        rgbd_tensors.append(rgbd_tensor)

    rgbd_tensors = torch.stack(rgbd_tensors)
    gym.end_access_image_tensors(sim)
    return rgbd_tensors


def cvtVoidXyzToList(void_array, coef=1):
    return [coef*void_array['x'], coef*void_array['y'], coef*void_array['z']]


def transform2list(transform):
    return [[transform.p.x, transform.p.y, transform.p.z],
            [transform.r.x, transform.r.y, transform.r.z, transform.r.w]]


def change_rigid_color(env, body_handle, color=[0, 0, 0], link_id=0):
    color = gymapi.Vec3(*color)
    gym.set_rigid_body_color(env, body_handle, link_id, gymapi.MESH_VISUAL_AND_COLLISION, color)


def convert_time(relative_time):
    relative_time = int(relative_time)
    hours = relative_time // 3600
    left_time = relative_time % 3600
    minutes = left_time // 60
    seconds = left_time % 60
    return f'{hours}:{minutes}:{seconds}'


def inverse_transform(poses):
    inv_position = -poses[:, :3]
    inv_orientation = quat_conjugate(poses[:, 3:])
    return inv_position, inv_orientation


def orientation_error(desired, current=None):
    if current is None: 
        return desired[:, 0:3] * torch.sign(desired[:, 3]).unsqueeze(-1)
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def homo_transform(rot, translation, point): # 3D homogeneous transform
    return quat_apply(rot, point) + translation


def multiply_transform(translation1, rot1, translation2, rot2):
    # R1 * [R2X + T2] + T1 
    # quaternion order same as rotation matrix ??? https://math.stackexchange.com/questions/940159/multiplication-of-rotation-matrices-in-quaternion
    new_rot = quat_mul(rot1, rot2)
    new_translation = quat_apply(rot1, translation2) + translation1
    return torch.cat([new_rot, new_translation], dim=1)


def getMeshBbox(mesh_path, oriented_bbox=False, scale=None):
    mesh = trimesh.load_mesh(mesh_path)
    if scale is None:
        if oriented_bbox: return mesh.bounding_box_oriented.primitive.extents
        else: return mesh.bounding_box.primitive.extents
    else:
        if oriented_bbox: return [extend*scale[i] for i, extend in enumerate(mesh.bounding_box_oriented.primitive.extents)]
        else: return [extend*scale[i] for i, extend in enumerate(mesh.bounding_box.primitive.extents)]


def getUrdfStates(urdf_path):
    robot = urdf.Robot.from_xml_file(urdf_path)
    link_states = []
    # Iterate through links or joints to access attributes
    for link_name, link in robot.link_map.items():
        # Get scaling of the link
        link_geometry = deepcopy(link.collision.geometry) if link.collision is not None else print(f"Link {link_name} has no collision geometry")
        link_states.append((f"{link_name}", link_geometry)) 
    return link_states


def quat_from_euler(euler):
    # xyz
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]
    return quat_from_euler_xyz(roll, pitch, yaw)


def euler_from_quat(q):
    shape_q = q.shape
    q = q.reshape(-1, 4)
    roll, pitch, yaw = get_euler_xyz(q)
    euler = torch.stack([roll, pitch, yaw], dim=-1)
    return euler.reshape(*shape_q[:-1], 3)


def quat_bet_vects(v1, v2):
    # Compute the quaternion rotation from vector v1 to the vector v2
    v1, v2 = normalize(v1), normalize(v2)
    dot_prod = torch.dot(v1, v2)
    if (dot_prod < -0.999999): return quat_from_euler_xyz(0, 0, torch.pi)
    elif (dot_prod > 0.999999): return quat_from_euler_xyz(0, 0, 0)
    else:
        rot_axis = torch.cross(v1, v2)
        w = torch.tensor([1 + dot_prod], device=rot_axis.device)
        return normalize(torch.cat([rot_axis, w]))


def np_scale(x, lower=0, upper=255):
    shift_x = x - np.min(x)
    normalize_x = shift_x / (np.max(shift_x)+1e-15)
    return (upper - lower) * normalize_x + lower


@torch.jit.script
def euler_to_transform_matrix(roll, pitch, yaw):
    # Create rotation matrices
    R_roll = torch.stack([torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll),
                          torch.zeros_like(roll), torch.cos(roll), -torch.sin(roll),
                          torch.zeros_like(roll), torch.sin(roll), torch.cos(roll)], dim=1).view(-1, 3, 3)

    R_pitch = torch.stack([torch.cos(pitch), torch.zeros_like(pitch), torch.sin(pitch),
                           torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch),
                           -torch.sin(pitch), torch.zeros_like(pitch), torch.cos(pitch)], dim=1).view(-1, 3, 3)

    R_yaw = torch.stack([torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw),
                         torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw),
                         torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=1).view(-1, 3, 3)
    
    # Multiply the rotation matrices to get the combined rotation matrix
    R = torch.bmm(R_yaw, torch.bmm(R_pitch, R_roll))
    
    # Create a 4x4 identity matrix with batch dimensions
    identity = torch.eye(4).unsqueeze(0).expand(R.size(0), -1, -1).to(R.device)

    # Replace the upper-left 3x3 corner of the identity matrix with the rotation matrix
    transform_matrix = identity.clone()
    transform_matrix[:, :3, :3] = R
    
    return transform_matrix


@torch.jit.script
def ravel_multi_index(coords: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    r"""Converts a tensor of coordinate vectors into a tensor of flat indices.

    This is a `torch` implementation of `numpy.ravel_multi_index`.

    Args:
        coords: A tensor of coordinate vectors, (*, D).
        shape: The source shape.

    Returns:
        The raveled indices, (*,).
    """

    shape = coords.new_tensor((*shape, 1))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return (coords * coefs).sum(dim=-1)


@torch.jit.script
def unravel_index(indices: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    r"""Converts a tensor of flat indices into a tensor of coordinate vectors.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of flat indices, (*,).
        shape: The target shape.

    Returns:
        The unraveled coordinates, (*, D).
    """

    shape = indices.new_tensor((*shape, 1))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return torch.div(indices[..., None], coefs, rounding_mode='trunc') % shape[:-1]


@torch.jit.script
def torch_rand_float(lower, upper, shape):
    # type: (Tensor, Tensor, Tuple[int, int]) -> Tensor
    return torch.rand(*shape, device=lower.device) * (upper - lower) + lower


@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)
