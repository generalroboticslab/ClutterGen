import json
from collections import OrderedDict
import os
import csv
import pprint
import numpy as np
import torch



def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


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
    bbox = bbox.copy()
    # scene_center_pos is the relative translation from the object baselink to the center of the object bounding box
    # All bbox given should be in the baselink frame (baselink is at the origin)
    scene_center_pos, scene_half_extents = bbox[:3], bbox[7:10]
    scene_center_pos[2] += scene_half_extents[2] + z_half_extend
    scene_half_extents[2] = z_half_extend
    orientation = bbox[3:7]
    return np.array([*scene_center_pos, *orientation, *scene_half_extents])


def get_in_bbox(bbox, z_half_extend:float=None):
    bbox = bbox.copy()
    if z_half_extend is None: z_half_extend = bbox[9]
    # Half extend should not be smaller than the original half extend
    z_half_extend = max(z_half_extend, bbox[9])
    # scene_center_pos is the relative translation from the object baselink to the center of the object bounding box
    # You need to change the relative translation not the absolute translation (half extent in z-axis)
    scene_center_pos, scene_half_extents = bbox[:3], bbox[7:10]
    scene_center_pos[2] += z_half_extend - scene_half_extents[2]
    scene_half_extents[2] = z_half_extend
    orientation = bbox[3:7]
    return np.array([*scene_center_pos, *orientation, *scene_half_extents])


def pc_random_downsample(pc_array, num_points):
    """ Randomly downsample a point cloud
        if num_points >= pc_array.shape[0], pad the point cloud with zeros 
        Args:
        pc_array: (N, 3) numpy array
        num_points: int
    """
    if num_points >= pc_array.shape[0]: 
        return pc_array
    else:
        idx = np.random.choice(pc_array.shape[0], num_points, replace=False)
        return pc_array[idx]


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