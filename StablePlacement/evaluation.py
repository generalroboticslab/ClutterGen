import os
import sys
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory path
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
import datetime

import torch
from torch.utils.data import DataLoader
from DataLoader import HDF5Dataset
from sp_model import StablePlacementModel
from torch.optim import Adam
from torch.nn.functional import mse_loss

import pybullet as p
import pybullet_utils as pu
import numpy as np

from tqdm import tqdm
import argparse
from distutils.util import strtobool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='StablePlacement_EVAL')
    parser.add_argument('--main_dataset_path', type=str, default='SP_Dataset/table_10_group0_dinning_table.h5')
    parser.add_argument('--model_save_folder', type=str, default='SP_Model')
    parser.add_argument('--visualize_pc', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Save dataset or not')
    parser.add_argument('--checkpoint', type=str, default='StablePlacement_03-02_22:40')
    parser.add_argument('--index_episode', type=str, default="best")

    args = parser.parse_args()
    timer = '_' + '_'.join(str(datetime.datetime.now())[5:16].split())  # a time name file
    args.final_name = args.env_name + timer
    args.checkpoint_path = os.path.join(args.model_save_folder, args.checkpoint, args.index_episode)+".pth"

    return args


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StablePlacementModel(device=device).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()
    print("Model loaded from", args.checkpoint_path)

    main_dataset_path = args.main_dataset_path
    test_dataset_path = main_dataset_path.replace('.h5', '_test.h5')
    sp_test_dataloader = DataLoader(HDF5Dataset(test_dataset_path), batch_size=5, shuffle=False)

    test_loss = 0.
    with torch.no_grad():
        for batch in tqdm(sp_test_dataloader):
            scene_pc = batch['scene_pc'].to(device)
            qr_obj_pc = batch['qr_obj_pc'].to(device)
            qr_obj_pose = batch['qr_obj_pose'].to(device)
            # Forward pass
            pred_pose = model(scene_pc, qr_obj_pc)
            # Compute loss
            loss = mse_loss(pred_pose, qr_obj_pose)
            test_loss += loss.item()

            if args.visualize_pc:
                scene_pc = scene_pc.cpu().numpy()
                qr_obj_pc = qr_obj_pc.cpu().numpy()
                pred_pose = pred_pose.cpu().numpy()
                qr_obj_pose = qr_obj_pose.cpu().numpy()
                # Visualize the point cloud
                for i in range(len(scene_pc)):
                    transformed_pred_qr_obj_pc = np.array(
                        [p.multiplyTransforms(
                            pred_pose[i, :3], pred_pose[i, 3:], 
                            point, [0., 0., 0., 1.])[0] for point in qr_obj_pc[i]]
                        )
                    transformed_ground_truth_qr_obj_pc = np.array(
                        [p.multiplyTransforms(
                            qr_obj_pose[i, :3], qr_obj_pose[i, 3:], 
                            point, [0., 0., 0., 1.])[0] for point in qr_obj_pc[i]]
                        )
                    pu.visualize_pc_lst(
                        [scene_pc[i], transformed_pred_qr_obj_pc, transformed_ground_truth_qr_obj_pc], 
                        color=[[0, 0, 1], [1, 0, 0], [0, 1, 0]])

        test_loss /= len(sp_test_dataloader)
