import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory path
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def check_dict_same_len(dict):
    len_dict = None
    for k, v in dict.items():
        if len_dict is None:
            len_dict = len(v)
            continue
        assert len_dict == len(v), f"All datasets must have the same length {len_dict}, but {k} has length {len(v)}"
    return len_dict


def split_dataset(hdf5_filename, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_select=True):
    assert train_ratio + val_ratio + test_ratio == 1., "Ratio must sum to 1"
    train_filename = hdf5_filename.replace('.h5', '_train.h5')
    val_filename = hdf5_filename.replace('.h5', '_val.h5')
    test_filename = hdf5_filename.replace('.h5', '_test.h5')
    with h5py.File(hdf5_filename, 'r') as f:
        dataset_len = check_dict_same_len(f)
        train_len = int(train_ratio * dataset_len)
        val_len = int(val_ratio * dataset_len)
        test_len = dataset_len - train_len - val_len
        indices = np.arange(dataset_len)
        if random_select:
            np.random.shuffle(indices)
        
        # Indexing elements must be in increasing order for h5
        train_indices = np.sort(indices[:train_len])
        val_indices = np.sort(indices[train_len:train_len+val_len])
        test_indices = np.sort(indices[train_len+val_len:])
        train_data = {k: v[train_indices] for k, v in f.items()}
        val_data = {k: v[val_indices] for k, v in f.items()}
        test_data = {k: v[test_indices] for k, v in f.items()}

        with h5py.File(train_filename, "w") as train_f:
            for k, v in train_data.items():
                train_f.create_dataset(k, data=v)
        with h5py.File(val_filename, "w") as val_f:
            for k, v in val_data.items():
                val_f.create_dataset(k, data=v)
        with h5py.File(test_filename, "w") as test_f:
            for k, v in test_data.items():
                test_f.create_dataset(k, data=v)
    print(f"Split dataset into\n"
          f"train: {train_filename}\n" 
          f"val: {val_filename}\n"
          f"test: {test_filename}")


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_filename):
        self.hdf5_filename = hdf5_filename
        self.file = h5py.File(self.hdf5_filename, 'r')
        # Check that all datasets have the same length
        dataset_len = check_dict_same_len(self.file)
        self.length = dataset_len
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Directly access the data without reopening the file
        scene_pc = torch.tensor(self.file['scene_pc'][idx], dtype=torch.float32)
        qr_obj_pc = torch.tensor(self.file['qr_obj_pc'][idx], dtype=torch.float32)
        qr_obj_pose = torch.tensor(self.file['qr_obj_pose'][idx], dtype=torch.float32)
        # pu.visualize_pc(scene_pc)
        # pu.visualize_pc(qr_obj_pc)
        # print(qr_obj_pose)
        # transform_qr_obj_pc = np.array(
        #     [p.multiplyTransforms(
        #         qr_obj_pose[:3], qr_obj_pose[3:], 
        #         point, [0., 0., 0., 1.])[0] for point in qr_obj_pc]
        #     )
        # pu.visualize_pc(np.concatenate([scene_pc, transform_qr_obj_pc], axis=0))
        # Your data processing here...
        return {'scene_pc': scene_pc, 'qr_obj_pc': qr_obj_pc, 'qr_obj_pose': qr_obj_pose}
    
    def close(self):
        self.file.close()


if __name__=="__main__":
    dataset_path = "SP_Dataset/table_10_group0_dinning_table.h5"
    split_dataset(dataset_path, random_select=True)
    
    from utils import read_h5py_mem
    train_dataset_path = dataset_path.replace('.h5', '_train.h5')
    train_dataset = read_h5py_mem(train_dataset_path)
