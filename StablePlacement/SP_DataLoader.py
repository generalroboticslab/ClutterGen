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
from torch.utils.data import DataLoader

from utils import read_dataset_recursively, se3_transform_pc
import pybullet_utils as pu


def split_dataset(hdf5_filename, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_select=True):
    assert train_ratio + val_ratio + test_ratio == 1., "Ratio must sum to 1"
    train_filename = hdf5_filename.replace('.h5', '_train.h5')
    val_filename = hdf5_filename.replace('.h5', '_val.h5')
    test_filename = hdf5_filename.replace('.h5', '_test.h5')
    with h5py.File(hdf5_filename, 'r') as f:
        keys = list(f.keys())
        dataset_len = len(f)
        # dataset_len = check_dict_same_len(f)
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

        def copy_data(source_file, target_filename, indices):
            with h5py.File(target_filename, 'w') as target_file:
                # indice is the key of the original dataset, index is the index of the new dataset
                for index, indice in enumerate(indices):
                    key = keys[indice]
                    source_file.copy(source_file[key], target_file, name=str(index))

        copy_data(f, train_filename, train_indices)
        copy_data(f, val_filename, val_indices)
        copy_data(f, test_filename, test_indices)

        print(f"Split dataset into\n"
              f"train: {train_filename}\n"
              f"val: {val_filename}\n"
              f"test: {test_filename}")


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_filepath):
        self.hdf5_filepath = hdf5_filepath
        self.file = h5py.File(self.hdf5_filepath, 'r')
        # Check that all datasets have the same length
        self.length = len(self.file)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        idx = str(idx)
        data_group = self.file[idx]
        scene_pc = torch.tensor(data_group['scene_pc'][()], dtype=torch.float32)
        qr_obj_pc = torch.tensor(data_group['qr_obj_pc'][()], dtype=torch.float32)  # Ensure it's a single array
        qr_obj_pose = torch.tensor(data_group['qr_obj_pose'][()], dtype=torch.float32)
        qr_obj_name = data_group['qr_obj_name'][()].decode('utf-8')
        World2PlacedObj_poses = read_dataset_recursively(data_group['World2PlacedObj_poses'])

        # pu.visualize_pc(scene_pc)
        # pu.visualize_pc(qr_obj_pc)
        # print(qr_obj_pose)
        # transform_qr_obj_pc = se3_transform_pc(qr_obj_pose[:3], qr_obj_pose[3:], qr_obj_pc)
        # pu.visualize_pc(np.concatenate([scene_pc, transform_qr_obj_pc], axis=0))
        
        return {'scene_pc': scene_pc, 
                'qr_obj_pc': qr_obj_pc, 
                'qr_obj_pose': qr_obj_pose, 
                'qr_obj_name': qr_obj_name, 
                'World2PlacedObj_poses': World2PlacedObj_poses}


    def close(self):
        self.file.close()


def custom_collate(batch):
    batched_data = {}
    # Assuming other keys are numpy arrays that should be concatenated
    for k, v in batch[0].items():
        if isinstance(v, torch.Tensor):
            batched_data[k] = torch.stack([item[k] for item in batch])
        else:
            batched_data[k] = [item[k] for item in batch]
    return batched_data


if __name__=="__main__":
    dataset_path = "StablePlacement/SP_Dataset/table_10_group0_dinning_table.h5"
    split_dataset(dataset_path, random_select=True)
    
    from utils import read_h5py_mem
    train_dataset_path = dataset_path.replace('.h5', '_train.h5')
    train_dataset = read_h5py_mem(train_dataset_path)

    Dataset = HDF5Dataset(train_dataset_path)
    sp_train_dataloader = DataLoader(HDF5Dataset(train_dataset_path), batch_size=2, shuffle=True, collate_fn=custom_collate)
    sp_train_dataloader = iter(sp_train_dataloader)
    batch = next(sp_train_dataloader)

