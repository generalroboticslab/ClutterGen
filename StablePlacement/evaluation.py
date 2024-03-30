import os
import sys
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory path
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
import datetime
from tabulate import tabulate
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from SP_DataLoader import HDF5Dataset, custom_collate, create_subset_dataset
from sp_model import StablePlacementPolicy_Determ, StablePlacementPolicy_Beta, StablePlacementPolicy_Normal
from torch.optim import Adam
from torch.nn.functional import mse_loss

import pybullet_utils_cust as pu
from sp_utils import update_success_records, compute_success_records_summary
import numpy as np
from utils import se3_transform_pc, read_json, sorted_dict, generate_table

from tqdm import tqdm
import time
import argparse
from distutils.util import strtobool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='StablePlacement_EVAL')
    parser.add_argument('--main_dataset_path', type=str, default='StablePlacement/SP_Dataset/table_10_group0_dinning_table.h5')
    parser.add_argument('--model_save_folder', type=str, default='StablePlacement/SP_Result')
    parser.add_argument('--visualize_pc', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Save dataset or not')
    parser.add_argument('--checkpoint', type=str, default='StablePlacement_03-20_16:02_Deterministic_WeightedLoss_EntCoef0.0_weiDecay0.0')
    parser.add_argument('--index_episode', type=str, default="1000")
    
    parser.add_argument('--use_simulator', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Save dataset or not')
    parser.add_argument('--rendering', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--realtime', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--vel_threshold', type=float, default=[0.005, np.pi/36], nargs='+')
    parser.add_argument('--acc_threshold', type=float, default=[1., np.pi*2], nargs='+') 
    parser.add_argument('--env_json_name', type=str, default='Union_03-12_23:40Sync_Beta_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab60_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_entropy0.01_seed123456_EVAL_best_objRange_10_10')
    parser.add_argument('--use_robot_sp', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Visualize critic')

    parser.add_argument('--test_dataset', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Save dataset or not')


    args = parser.parse_args()
    timer = '_' + '_'.join(str(datetime.datetime.now())[5:16].split())  # a time name file
    args.final_name = args.env_name + timer
    args.checkpoint_path = os.path.join(args.model_save_folder, args.checkpoint, "Checkpoint", f"epoch_{args.index_episode}")+".pth"
    args.json_path = os.path.join(args.model_save_folder, args.checkpoint, "Json", f"{args.checkpoint}.json")

    # Keep the training args if evaluation args is None
    stored_args = args.__dict__.copy()  # store eval_args to avoid overwrite
    train_args = read_json(args.json_path)
    args.__dict__.update(train_args)
    args.__dict__.update(stored_args) # Merge train and eval args but keep eval args

    return args


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_simulator or args.use_robot_sp:
        from RoboSensai_bullet import RoboSensaiBullet
        from PPO.PPO_continuous_sg import Agent, update_tensor_buffer
        from utils import combine_envs_float_info2list
        import pybullet_utils_cust as pu
        # If use the simulator, we need to specify the .json name
        args.env_json_path = os.path.join("eval_res", args.object_pool_name, "Json", args.env_json_name+'.json')
        assert os.path.exists(args.env_json_path), f"Json file for create envs\n {args.env_json_path}\n does not exist"
        env_args = argparse.Namespace()
        env_args.__dict__.update(read_json(args.env_json_path))
        env_args.rendering = args.rendering
        env_args.realtime = args.realtime
        env_args.vel_threshold = args.vel_threshold
        env_args.acc_threshold = args.acc_threshold
        env_args.use_robot_sp = args.use_robot_sp
        envs = RoboSensaiBullet(env_args)
        # t_agent = Agent(envs).to(device)
        # t_agent.load_checkpoint(env_args.checkpoint_path, evaluate=True, map_location="cuda:0")
        # t_agent.pc_extractor.eval() # The PC extractor's BN layer was set to train so we keep it train first.

    if args.use_beta:
        model = StablePlacementPolicy_Beta(device=device).to(device)
    elif args.use_normal:
        model = StablePlacementPolicy_Normal(device=device).to(device)
    else:
        model = StablePlacementPolicy_Determ(device=device).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()
    print("Model loaded from", args.checkpoint_path)

    main_dataset_path = args.main_dataset_path
    test_dataset_path = main_dataset_path.replace('.h5', '_test.h5')
    sp_test_dataloader = DataLoader(HDF5Dataset(test_dataset_path), batch_size=1, shuffle=False, collate_fn=custom_collate)

    test_loss = 0.; success_sum = 0.
    cluttered_record = 0; cluttered_success = 0
    pred_success_records = {
        "num_objs_on_qr_scene_success": {},
        "num_objs_on_qr_scene_counts": {},
        "num_times_obj_get_qr_success": {},
        "num_times_obj_get_qr_counts": {},
    }
    gt_success_records = deepcopy(pred_success_records)
    rp_success_records = deepcopy(pred_success_records)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(sp_test_dataloader)):
            scene_pc = batch['scene_pc'].to(device)
            qr_obj_pc = batch['qr_obj_pc'].to(device)
            qr_obj_pose = batch['qr_obj_pose'].to(device)
            qr_obj_name = batch['qr_obj_name'][0]
            sp_placed_obj_poses = batch['World2PlacedObj_poses'][0]
            for obj_name, obj_pose in sp_placed_obj_poses.items():
                sp_placed_obj_poses[obj_name] = pu.split_7d(obj_pose)
            # Forward pass
            pred_qr_obj_pose, _ = model(scene_pc, qr_obj_pc)
            # Compute loss
            loss = mse_loss(pred_qr_obj_pose, qr_obj_pose)
            test_loss += loss.item()
                
            if args.use_simulator:
                # Evaluate the model
                # Scene and obj feature tensor are keeping updated inplace]
                    ################ agent evaluation ################
                # Test the ground truth is correct (the datapoint is valid)
                pred_success, _ = envs.stable_placement_eval_step(qr_obj_name, pred_qr_obj_pose, sp_placed_obj_poses)
                gt_success, _ = envs.stable_placement_eval_step(qr_obj_name, qr_obj_pose, sp_placed_obj_poses) # Copy to avoid in-place modification in the stable_placement_eval_step
                update_success_records(gt_success_records, qr_obj_name, sp_placed_obj_poses, gt_success)
                
                if not gt_success and not pred_success:
                    continue # Skip the prediction if the ground truth is not successful

                # Test the prediction
                update_success_records(pred_success_records, qr_obj_name, sp_placed_obj_poses, pred_success)

                if args.use_robot_sp:
                    # Evaluate the model
                    # Scene and obj feature tensor are keeping updated inplace]
                    ################ agent evaluation ################
                    # if qr_obj_name not in ['005_tomato_soup_can_0', '006_mustard_bottle_0', '007_tuna_fish_can_0', '009_gelatin_box_0', '010_potted_meat_can_0', '019_pitcher_base_0', '024_bowl_0', '025_mug_0', '029_plate_0', '030_fork_0']
                    envs.stable_placement_reset()
                    scene_pc, qr_obj_pc = envs.stable_placement_compute_observation(qr_obj_name, sp_placed_obj_poses)
                    scene_pc, qr_obj_pc = torch.from_numpy(scene_pc).to(device).unsqueeze(0).float(), torch.from_numpy(qr_obj_pc).to(device).unsqueeze(0).float()
                    pred_qr_obj_pose, _ = model(scene_pc, qr_obj_pc)
                    rp_success = envs.stable_placement_task_step(qr_obj_name, pred_qr_obj_pose, sp_placed_obj_poses)
                    envs.stable_placement_reset()
                    update_success_records(rp_success_records, qr_obj_name, sp_placed_obj_poses, rp_success)

            if args.visualize_pc:
                scene_pc_np = scene_pc.cpu().numpy()
                qr_obj_pc_np = qr_obj_pc.cpu().numpy()
                pred_qr_obj_pose_np = pred_qr_obj_pose.cpu().numpy()
                qr_obj_pose_np = qr_obj_pose.cpu().numpy()
                # Visualize the point cloud
                for i in range(len(scene_pc_np)):
                    transformed_pred_qr_obj_pc = se3_transform_pc(pred_qr_obj_pose_np[i][:3], pred_qr_obj_pose_np[i][3:], qr_obj_pc_np[i])
                    transformed_ground_truth_qr_obj_pc = se3_transform_pc(qr_obj_pose_np[i][:3], qr_obj_pose_np[i][3:], qr_obj_pc_np[i])
                    pu.visualize_pc_lst(
                        [scene_pc_np[i], transformed_pred_qr_obj_pc, transformed_ground_truth_qr_obj_pc], 
                         color=[[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        
        test_loss /= len(sp_test_dataloader)
        
        if args.use_simulator:
            compute_success_records_summary(gt_success_records)
            compute_success_records_summary(pred_success_records)
            compute_success_records_summary(rp_success_records)
        

    print(f"Test Loss: {test_loss:.4f}")
    if args.use_simulator:
        # Assuming you have these variables from your environment
        gt_success_records = sorted_dict(gt_success_records)
        pred_success_records = sorted_dict(pred_success_records)
        rp_success_records = sorted_dict(rp_success_records)

        gt_qr_scene_success_misc = generate_table(gt_success_records, "num_objs_on_qr_scene_success_rate", "num_objs_on_qr_scene_counts", "Ground Truth QR Scene Success")
        gt_qr_obj_success_misc = generate_table(gt_success_records, "num_times_obj_get_qr_success_rate", "num_times_obj_get_qr_counts", "Ground Truth QR Objects Success")
        pred_qr_scene_success_misc = generate_table(pred_success_records, "num_objs_on_qr_scene_success_rate", "num_objs_on_qr_scene_counts", "Predicted QR Scene Success")
        pred_qr_obj_success_misc = generate_table(pred_success_records, "num_times_obj_get_qr_success_rate", "num_times_obj_get_qr_counts", "Predicted QR Objects Success")
        rp_qr_scene_success_misc = generate_table(rp_success_records, "num_objs_on_qr_scene_success_rate", "num_objs_on_qr_scene_counts", "Robot Sensai QR Scene Success")
        rp_qr_obj_success_misc = generate_table(rp_success_records, "num_times_obj_get_qr_success_rate", "num_times_obj_get_qr_counts", "Robot Sensai QR Objects Success")

        print(f"Ground Truth Avg Success Rate: {gt_success_records['Avg_success_rate']} / Data Points: {gt_success_records['recorded_num_data_points']}")
        print(tabulate(gt_qr_scene_success_misc, headers="firstrow", tablefmt="grid"))
        print(tabulate(gt_qr_obj_success_misc, headers="firstrow", tablefmt="grid"))
        print(f"Prediction Avg Success Rate On Test Dataset (after filtering): {pred_success_records['Avg_success_rate']} / Data Points: {pred_success_records['recorded_num_data_points']}")
        print(tabulate(pred_qr_scene_success_misc, headers="firstrow", tablefmt="grid"))
        print(tabulate(pred_qr_obj_success_misc, headers="firstrow", tablefmt="grid"))
        print(f"Robot Place Prediction Success Rate On Test Dataset (after filtering): {rp_success_records['Avg_success_rate']} / Data Points: {rp_success_records['recorded_num_data_points']}")
        print(tabulate(rp_qr_scene_success_misc, headers="firstrow", tablefmt="grid"))
        print(tabulate(rp_qr_obj_success_misc, headers="firstrow", tablefmt="grid"))