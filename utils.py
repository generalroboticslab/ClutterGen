import json
from collections import OrderedDict
import os
import csv
import pprint
import numpy as np
import torch
import re
import h5py


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


def read_h5py_mem(h5py_path):
    with h5py.File(h5py_path, 'r') as f:
        return read_dataset_recursively(f)


def read_dataset_recursively(hdf5_group):
    """
    Recursively read datasets from an HDF5 group into a nested dictionary.

    :param hdf5_group: HDF5 group object to read the datasets from
    :return: Nested dictionary with the same structure as the HDF5 group/datasets
    """
    data_dict = {}
    for key, item in hdf5_group.items():
        if isinstance(item, h5py.Dataset):
            # Read the dataset and assign to the dictionary
            data_dict[key] = item[()]
        elif isinstance(item, h5py.Group):
            # Recursively read the group
            data_dict[key] = read_dataset_recursively(item)
    return data_dict


def create_dataset_recursively(hdf5_group, data_dict):
    """
    Recursively create datasets from a nested dictionary within an HDF5 group.

    :param hdf5_group: HDF5 group object to store the datasets
    :param data_dict: Nested dictionary containing the data to store
    """
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # Create a sub-group for nested dictionaries
            sub_group = create_or_update_group(hdf5_group, key)
            create_dataset_recursively(sub_group, value)
        else:
            # Create a dataset for non-dictionary items
            hdf5_group.create_dataset(key, data=value)


def create_or_update_group(parent, group_name):
    """Create or get a group in the HDF5 file."""
    return parent.require_group(group_name)


def save_h5py(data, h5py_path):
    with h5py.File(h5py_path, 'w') as f:
        for k, v in data.items():
            f.create_dataset(k, data=v)


def write_csv_line(result_file_path, result):
    """ write a line in a csv file; create the file and write the first line if the file does not already exist """
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(result)
    result = OrderedDict(result)
    file_exists = os.path.exists(result_file_path)
    with open(result_file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, result.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(result)


def dict2list(diction):
    """ convert a dictionary to a list of tuples """
    key_list, value_list = [], []
    for k, v in diction.items():
        if isinstance(v, dict):
            subkey_lst, subvalue_lst = dict2list(v)
            key_list.extend(subkey_lst)
            value_list.extend(subvalue_lst)
        else:
            key_list.append(k)
            value_list.append(v)
    return key_list, value_list


def get_on_bbox(bbox, z_half_extend:float):
    # scene_center_pos is the relative translation from the scene object's baselink to the center of the scene object's bounding box
    # All bbox given should be in the center frame (baselink is at the origin when import the urdf)
    SceneCenter_2_QRregionCenter = [0, 0, bbox[9]+z_half_extend]
    orientation = [0, 0, 0, 1.]
    QRregion_half_extents = bbox[7:10].copy()
    QRregion_half_extents[2] = z_half_extend
    return np.array([*SceneCenter_2_QRregionCenter, *orientation, *QRregion_half_extents])


def get_in_bbox(bbox, z_half_extend:float=None):
    if z_half_extend is None: z_half_extend = bbox[9]
    # Half extend should not be smaller than the original half extend
    z_half_extend = max(z_half_extend, bbox[9])
    # scene_center_pos is the relative translation from the scene object's baselink to the center of the scene object's bounding box
    # All bbox given should be in the center frame (baselink is at the origin when import the urdf)
    SceneCenter_2_QRregionCenter = [0, 0, z_half_extend-bbox[9]]
    orientation = [0, 0, 0, 1.]
    QRregion_half_extents = bbox[7:10].copy()
    QRregion_half_extents[2] = z_half_extend
    return np.array([*SceneCenter_2_QRregionCenter, *orientation, *QRregion_half_extents])


def pc_random_downsample(pc_array, num_points, autopad=False):
    """ Randomly downsample/shuffle a point cloud
        Args:
        pc_array: (N, 3) numpy array
        num_points: int
    """
    if num_points >= pc_array.shape[0]: 
        if autopad: # Pad the point cloud with zeros will make the next real scene points become sparse
            pc_array = np.concatenate([pc_array, np.zeros((num_points - pc_array.shape[0], 3))], axis=0) 
        return np.random.permutation(pc_array)
    else:
        return pc_array[np.random.choice(pc_array.shape[0], num_points, replace=False)]
    

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
        sampled_pc: sampled pointcloud data, [npoint, 3]
    """
    N, C = xyz.shape
    centroids = np.zeros(npoint, dtype=int)
    min_distance = np.ones(N) * 1e10 # min distance from the unsampled points to the sampled points
    farthest_idx = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest_idx
        centroid = xyz[farthest_idx, :]
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        mask = dist < min_distance
        min_distance[mask] = dist[mask]
        farthest_idx = np.argmax(min_distance, axis=-1)
    sampled_pc = xyz[centroids]
    return sampled_pc


def inverse_sigmoid(x):
    return torch.log(x / (1 - x + 1e-10))
    

def create_mesh_grid(action_ranges=[(0, 1)]*6, num_steps=[5]*6):
    assert len(action_ranges) == len(num_steps), "action_ranges and num_steps must have the same length"
    action_steps = [torch.linspace(start, end, num_steps[j]) for j, (start, end) in enumerate(action_ranges)]
    # Use torch.meshgrid with explicit indexing argument
    meshgrid_tensors = torch.meshgrid(*action_steps, indexing='ij')
    # Stack the meshgrid tensors along a new dimension to get the final meshgrid tensor
    meshgrid_tensor = torch.stack(meshgrid_tensors, dim=-1)
    return meshgrid_tensor


def tensor_memory_in_mb(tensor):
    # Calculate the memory occupied by the tensor
    num_elements = tensor.numel()
    element_size = tensor.element_size()
    total_memory_bytes = num_elements * element_size
    total_memory_mb = total_memory_bytes / (1024 ** 2)  # Convert bytes to MB
    return total_memory_mb


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def check_file_exist(file_path):
    if os.path.exists(file_path):
        response = input(f"Find existing file {file_path}! Whether remove or not (y/n):")
        if response == 'y' or response == 'Y': 
            os.remove(file_path)
        else: 
            raise Exception("Give up this evaluation because of exsiting file.")


### Multi-Envs Utils ###
def combine_envs_float_info2list(infos, key, env_ids=None):
    if env_ids is None: env_ids = range(len(infos))
    return [infos[id][key] for id in env_ids]


def combine_envs_dict_info2dict(infos, key, env_ids=None):
    if env_ids is None: env_ids = range(len(infos))
    merged_info = {}
    for id in env_ids:
        info_dict = infos[id][key]
        for k, v in info_dict.items():
            if k not in merged_info: 
                merged_info[k] = v
                continue
            cur_val, nums = merged_info[k]
            new_val, new_nums = v
            merged_info[k] = [(cur_val * nums + new_val * new_nums) / (nums + new_nums), nums + new_nums]
    return merged_info


# Transformation

def quaternions_to_euler_array(quaternions):
    """
    Convert an array of quaternions into Euler angles (roll, pitch, and yaw) using the ZYX convention.
    
    Parameters:
    quaternions: A numpy array of shape (N, 4) where each row contains the components of a quaternion [x, y, z, w]
    
    Returns:
    euler_angles: A numpy array of shape (N, 3) where each row contains the Euler angles [roll, pitch, yaw]
    """
    if quaternions.ndim == 1:
        quaternions = quaternions[np.newaxis, :]
        flatten_flag = True
    else:
        flatten_flag = False

    # Preallocate the output array
    euler_angles = np.zeros((quaternions.shape[0], 3))
    
    # Extract components
    w, x, y, z = quaternions[:, 3], quaternions[:, 0], quaternions[:, 1], quaternions[:, 2]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    # Combine the angles
    euler_angles[:, 0] = roll
    euler_angles[:, 1] = pitch
    euler_angles[:, 2] = yaw
    
    if flatten_flag:
        euler_angles = euler_angles.flatten()

    return euler_angles


def normalize(x, eps: float = 1e-9):
    # Normalize an array of vectors
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=eps, max=None)


def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = np.stack([x, y, z, w], axis=-1).reshape(shape)
    return quat


def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = np.cross(xyz, b, axis=-1) * 2
    return (b + a[:, 3:] * t + np.cross(xyz, t, axis=-1)).reshape(shape)


def tf_combine(t1, q1, t2, q2):
    q1 = normalize(q1)
    q2 = normalize(q2)
    return quat_apply(q1, t2) + t1, quat_mul(q1, q2)


def se3_transform_pc(t, q, pc):
    if not isinstance(pc, np.ndarray):
        pc = np.array(pc)
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    if not isinstance(q, np.ndarray):
        q = np.array(q)
    if len(t.shape) == 1:
        t = t[np.newaxis, :]
    if len(q.shape) == 1:
        q = q[np.newaxis, :]

    pc_shape = pc.shape
    t = t.repeat(pc_shape[0], axis=0)
    q = q.repeat(pc_shape[0], axis=0)
    return quat_apply(q, pc) + t
