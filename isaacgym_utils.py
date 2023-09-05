from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import torch
import trimesh

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
            if positions is not None:
                state_tensor[actor_body_idxs, :3] = positions[i]
            if orientations is not None:
                state_tensor[actor_body_idxs, 3:7] = orientations[i]
            if linvels is not None:
                state_tensor[actor_body_idxs, 7:10] = linvels[i]
            if angvels is not None:
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


def homo_transform(rot, translation, point): # 3D homogeneous transform
    return quat_apply(rot, point) + translation


def multiply_transform(translation1, rot1, translation2, rot2):
    # R1 * [R2X + T2] + T1 
    # quaternion order same as rotation matrix ??? https://math.stackexchange.com/questions/940159/multiplication-of-rotation-matrices-in-quaternion
    new_rot = quat_mul(rot1, rot2)
    new_translation = quat_apply(rot1, translation2) + translation1
    return torch.cat([new_rot, new_translation], dim=1)


def getMeshBbox(mesh_path, oriented_bbox=False):
    mesh = trimesh.load_mesh(mesh_path)
    if oriented_bbox: return mesh.bounding_box_oriented.primitive.extents
    else: return mesh.bounding_box.primitive.extents


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


@torch.jit.script
def torch_rand_float(lower, upper, shape):
    # type: (Tensor, Tensor, Tuple[int, int]) -> Tensor
    return torch.rand(*shape, device=lower.device) * (upper - lower) + lower


@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)
