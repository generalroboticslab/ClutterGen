import json
from collections import OrderedDict
import os
import csv
import pprint
import numpy as np



def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def write_json(data, json_path):
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
    # center_pos is the relative translation from the object baselink to the center of the object bounding box
    center_pos, half_extents = bbox[:3], bbox[7:10]
    center_pos[2] += half_extents[2] + z_half_extend
    half_extents[2] = z_half_extend
    orientation = bbox[3:7]
    return np.array([*center_pos, *orientation, *half_extents])


def get_in_bbox(bbox, z_half_extend:float=None):
    bbox = bbox.copy()
    if z_half_extend is None: z_half_extend = bbox[9]
    # center_pos is the relative translation from the object baselink to the center of the object bounding box
    # You need to change the relative translation not the absolute translation (half extent in z-axis)
    center_pos, half_extents = bbox[:3], bbox[7:10]
    center_pos[2] += z_half_extend - half_extents[2]
    half_extents[2] = z_half_extend
    orientation = bbox[3:7]
    return np.array([*center_pos, *orientation, *half_extents])


def pc_random_downsample(pc_array, num_points):
    """ Randomly downsample a point cloud """
    num_points = min(num_points, pc_array.shape[0])
    idx = np.random.choice(pc_array.shape[0], num_points, replace=False)
    return pc_array[idx]