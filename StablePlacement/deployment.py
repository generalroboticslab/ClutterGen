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
from StablePlacement.sp_dataloader import HDF5Dataset, custom_collate
from sp_model import get_sp_model
from torch.optim import Adam
from torch.nn.functional import mse_loss
import numpy as np

import pybullet_utils_cust as pu
from utils import se3_transform_pc, read_json, sorted_dict, generate_table

from tqdm import tqdm
import time
from tabulate import tabulate
import argparse
from distutils.util import strtobool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='StablePlacement_EVAL')
    parser.add_argument('--main_dataset_path', type=str, default='StablePlacement/SP_Dataset/table_10_group0_dinning_table.h5')
    parser.add_argument('--model_save_folder', type=str, default='StablePlacement/SP_Result')
    parser.add_argument('--visualize_pc', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Save dataset or not')
    parser.add_argument('--checkpoint', type=str, default='StablePlacement_03-20_16:02_Deterministic_WeightedLoss_EntCoef0.0_weiDecay0.0')
    parser.add_argument('--index_episode', type=str, default="simbest")
    
    parser.add_argument('--use_simulator', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Save dataset or not')
    parser.add_argument('--rendering', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--realtime', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--vel_threshold', type=float, default=[0.005, np.pi/36], nargs='+')
    parser.add_argument('--acc_threshold', type=float, default=[1., np.pi*2], nargs='+') 
    parser.add_argument('--env_json_name', type=str, default='Union_03-12_23:40Sync_Beta_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab60_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_entropy0.01_seed123456_EVAL_best_objRange_10_10')
    parser.add_argument('--use_robot_sp', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Visualize critic')
    parser.add_argument('--num_trials', type=int, default=100)

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

    model = get_sp_model(env_args, device=device)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()
    print("Model loaded from", args.checkpoint_path)

    test_loss = 0.; success_sum = 0.; cluttered_record = 0; cluttered_success = 0; 
    pred_success_records = {
        "num_objs_on_qr_scene_success": {},
        "num_objs_on_qr_scene_counts": {},
        "num_times_obj_get_qr_success": {},
        "num_times_obj_get_qr_counts": {},
    }
    
    object_halfExtents = np.array(envs.tableHalfExtents)
    plane_scene_pc = np.random.uniform(-object_halfExtents, object_halfExtents, size=(10240, 3))
    plane_scene_pc[:, 2] = 0
    obj_names = list(envs.obj_name_data.keys())
    
    with torch.no_grad():
        for idx in range(args.num_trials):
            scene_pc = torch.tensor(plane_scene_pc).float().to(device).unsqueeze(0)
            sp_placed_obj_poses = {}
            for qr_obj_name in np.random.permutation(obj_names):
                qr_obj_pc = envs.obj_name_data[qr_obj_name]['pc']
                qr_obj_pc = torch.tensor(qr_obj_pc).float().to(device).unsqueeze(0)
                pred_qr_obj_pose, _ = model(scene_pc, qr_obj_pc)
                
                # Evaluate the model
                # Scene and obj feature tensor are keeping updated inplace]
                    ################ agent evaluation ################
                success, World_2_QRobjBase = envs.stable_placement_eval_step(qr_obj_name, pred_qr_obj_pose, sp_placed_obj_poses)
                
                pred_success_records["num_objs_on_qr_scene_success"][len(sp_placed_obj_poses)] = pred_success_records["num_objs_on_qr_scene_success"].get(len(sp_placed_obj_poses), 0) + success
                pred_success_records["num_objs_on_qr_scene_counts"][len(sp_placed_obj_poses)] = pred_success_records["num_objs_on_qr_scene_counts"].get(len(sp_placed_obj_poses), 0) + 1
                pred_success_records["num_times_obj_get_qr_success"][qr_obj_name] = pred_success_records["num_times_obj_get_qr_success"].get(qr_obj_name, 0) + success
                pred_success_records["num_times_obj_get_qr_counts"][qr_obj_name] = pred_success_records["num_times_obj_get_qr_counts"].get(qr_obj_name, 0) + 1
                
                sp_placed_obj_poses[qr_obj_name] = World_2_QRobjBase
                scene_pc_np = scene_pc.squeeze().cpu().numpy()
                qr_obj_pc_np = qr_obj_pc.squeeze().cpu().numpy()
                pred_qr_obj_pose_np = pred_qr_obj_pose.squeeze().cpu().numpy()
                transformed_pred_qr_obj_pc = se3_transform_pc(pred_qr_obj_pose_np[:3], pred_qr_obj_pose_np[3:], qr_obj_pc_np)
                scene_pc = torch.cat([scene_pc, torch.tensor(transformed_pred_qr_obj_pc).unsqueeze(0).to(device)], dim=1)

            if args.use_robot_sp:
                # Evaluate the model
                # Scene and obj feature tensor are keeping updated inplace]
                ################ agent evaluation ################
                envs.reset()
                if args.test_dataset:
                    pred_qr_obj_pose = qr_obj_pose
                success, _ = envs.stable_placement_eval_step(qr_obj_name, pred_qr_obj_pose, sp_placed_obj_poses)
                success_sum += success

            if args.visualize_pc:
                # Visualize the point cloud
                pu.visualize_pc_lst([scene_pc_np, transformed_pred_qr_obj_pc], color=[[0, 0, 1], [1, 0, 0]])

    pred_success_records["num_objs_on_qr_scene_success_rate"] = {k: v/pred_success_records["num_objs_on_qr_scene_counts"][k] for k, v in pred_success_records["num_objs_on_qr_scene_success"].items()}
    pred_success_records["num_times_obj_get_qr_success_rate"] = {k: v/pred_success_records["num_times_obj_get_qr_counts"][k] for k, v in pred_success_records["num_times_obj_get_qr_success"].items()}
    pred_success_records["Avg_success_rate"] = sum(pred_success_records["num_objs_on_qr_scene_success"].values())/(sum(pred_success_records["num_objs_on_qr_scene_counts"].values())+1e-8)
    pred_success_records["recorded_num_data_points"] = sum(pred_success_records["num_objs_on_qr_scene_counts"].values())
    pred_success_records = sorted_dict(pred_success_records)

    pred_qr_scene_success_misc = generate_table(pred_success_records, "num_objs_on_qr_scene_success_rate", "num_objs_on_qr_scene_counts", "Predicted QR Scene Success")
    pred_qr_obj_success_misc = generate_table(pred_success_records, "num_times_obj_get_qr_success_rate", "num_times_obj_get_qr_counts", "Predicted QR Objects Success")
    print(f"Prediction Avg Success Rate On Test Dataset: {pred_success_records['Avg_success_rate']} / Data Points: {pred_success_records['recorded_num_data_points']}")
    print(tabulate(pred_qr_scene_success_misc, headers="firstrow", tablefmt="grid"))
    print(tabulate(pred_qr_obj_success_misc, headers="firstrow", tablefmt="grid"))