import os
import datetime
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from SP_DataLoader import HDF5Dataset, custom_collate, create_subset_dataset
from sp_model import StablePlacementPolicy_Determ, StablePlacementPolicy_Beta, StablePlacementPolicy_Normal
from torch.optim import Adam
from torch.nn.functional import mse_loss

from tqdm import tqdm
import wandb
import argparse
from distutils.util import strtobool
from utils import save_json, read_json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='StablePlacement')
    parser.add_argument('--collect_data', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Save dataset or not')
    parser.add_argument('--main_dataset_path', type=str, default='StablePlacement/SP_Dataset/table_10_group0_dinning_table.h5')
    parser.add_argument('--save_folder', type=str, default='StablePlacement/SP_Result')
    
    # Training parameters
    parser.add_argument('--use_normal', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Save dataset or not')
    parser.add_argument('--use_beta', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Save dataset or not')
    parser.add_argument('--subset_ratio', type=float, default=None, help='The ratio of the subset of the dataset')
    parser.add_argument('--weighted_loss', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Save dataset or not')
    parser.add_argument('--epochs', type=int, default=1000, help='')
    parser.add_argument('--val_epochs', type=int, default=5, help='')
    parser.add_argument('--batch_size', type=int, default=40, help='')
    parser.add_argument('--ent_coef', type=float, default=0., help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay')

    # Evaluation parameters
    parser.add_argument('--use_simulator', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Save dataset or not')
    parser.add_argument('--rendering', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--realtime', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--vel_threshold', type=float, default=[0.005, np.pi/36], nargs='+')
    parser.add_argument('--acc_threshold', type=float, default=[1., np.pi*2], nargs='+') 

    parser.add_argument('--object_pool_name', type=str, default='Union', help="Object Pool. Ex: YCB, Partnet")
    parser.add_argument('--env_json_name', type=str, default='Union_03-12_23:40Sync_Beta_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab60_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_entropy0.01_seed123456_EVAL_best_objRange_10_10')
    parser.add_argument('--save_epoch', type=int, default=50, help='')
    parser.add_argument('--wandb', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Save dataset or not')

    args = parser.parse_args()

    # Specify the manual setting
    if args.use_simulator:
        args.val_sim_epochs = args.val_epochs * 10
    
    # Specify the final name of the model
    timer = '_' + '_'.join(str(datetime.datetime.now())[5:16].split())  # a time name file
    args.final_name = args.env_name + timer
    if args.use_beta:
        args.final_name += '_Beta'
    elif args.use_normal:
        args.final_name += '_Normal'
    else:
        args.final_name += '_Deterministic'

    if args.weighted_loss:
        args.final_name += '_WeightedLoss'
    if args.subset_ratio:
        args.final_name += f"_Subset{args.subset_ratio}"
    args.final_name += f"_EntCoef{args.ent_coef}"
    args.final_name += f"_weiDecay{args.weight_decay}"

    args.model_save_path = os.path.join(args.save_folder, args.final_name, "Checkpoint")
    args.json_save_path = os.path.join(args.save_folder, args.final_name, "Json")
    # Create folder to save the model
    if args.collect_data:
        os.makedirs(args.model_save_path, exist_ok=True)
        os.makedirs(args.json_save_path, exist_ok=True)
        save_json(vars(args), os.path.join(args.json_save_path, f'{args.final_name}.json'))

    return args

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset / Create the dataloader
main_dataset_path = args.main_dataset_path
train_dataset_path = main_dataset_path.replace('.h5', '_train.h5')
val_dataset_path = main_dataset_path.replace('.h5', '_val.h5')
if args.subset_ratio is not None:
    assert 0. < args.subset_ratio < 1., f"Subset ratio {args.subset_ratio} must be larger than 0 and less than 1"
    sp_train_dataloader = DataLoader(create_subset_dataset(HDF5Dataset(train_dataset_path), args.subset_ratio), batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate, num_workers=4)
else:
    sp_train_dataloader = DataLoader(HDF5Dataset(train_dataset_path), batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate, num_workers=4)

sp_val_dataloader = DataLoader(HDF5Dataset(val_dataset_path), batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate, num_workers=4)
if args.use_simulator:
    sp_sim_dataloader = DataLoader(HDF5Dataset(train_dataset_path), batch_size=1, shuffle=True, collate_fn=custom_collate)

# Create the model
if args.use_beta:
    model = StablePlacementPolicy_Beta(device=device).to(device)
elif args.use_normal:
    model = StablePlacementPolicy_Normal(device=device).to(device)
else:
    model = StablePlacementPolicy_Determ(device=device).to(device)
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Create the simulator
if args.use_simulator:
    from RoboSensai_bullet import RoboSensaiBullet
    from PPO.PPO_continuous_sg import Agent, update_tensor_buffer
    from utils import combine_envs_float_info2list
    import pybullet_utils as pu
    # If use the simulator, we need to specify the .json name
    args.env_json_path = os.path.join("eval_res", args.object_pool_name, "Json", args.env_json_name+'.json')
    assert os.path.exists(args.env_json_path), f"Json file for create envs\n {args.env_json_path}\n does not exist"
    env_args = argparse.Namespace()
    env_args.__dict__.update(read_json(args.env_json_path))
    env_args.rendering = args.rendering
    env_args.realtime = args.realtime
    env_args.vel_threshold = args.vel_threshold
    env_args.acc_threshold = args.acc_threshold
    envs = RoboSensaiBullet(env_args)
    # t_agent = Agent(envs).to(device)
    # t_agent.load_checkpoint(env_args.checkpoint_path, evaluate=True, map_location="cuda:0")
    # t_agent.pc_extractor.eval() # The PC extractor's BN layer was set to train so we keep it train first.

# Configure Wandb 
wb_config = dict(
    Name=args.env_name,
    Model='PointNet_MLP',
    Learning_rate=args.lr,
    Epochs=args.epochs,
    Batch_Size=args.batch_size,
    Entropy_Coefficient=args.ent_coef
)

if args.collect_data and args.wandb:
    wandb.init(project=args.env_name, entity='jiayinsen', config=wb_config, name=args.final_name)

best_val_loss = float('inf')
best_sim_success_rate = 0
for epoch in range(1, args.epochs+1):
    # Learning rate decay; Linear Schedular
    optimizer.param_groups[0]['lr'] = args.lr * (1 - epoch / args.epochs)

    pose_loss_record = 0; pos_loss_record = 0; quat_loss_record = 0; entropy_record = 0
    for batch in tqdm(sp_train_dataloader, desc=f'Epoch {epoch}/{args.epochs} Training'):
        scene_pc = batch['scene_pc'].to(device)
        qr_obj_pc = batch['qr_obj_pc'].to(device)
        qr_obj_pose = batch['qr_obj_pose'].to(device)
        # Forward pass
        pred_pose, entropy = model(scene_pc, qr_obj_pc)
        # Compute loss
        pos_loss = mse_loss(pred_pose[:, :3], qr_obj_pose[:, :3])
        quat_loss = mse_loss(pred_pose[:, 3:], qr_obj_pose[:, 3:])
        if entropy is None:
            entropy = torch.zeros_like(pos_loss, requires_grad=False)
        entropy_loss = -entropy.mean()
        
        if args.weighted_loss:
            # Weighted loss to balance the position and quaternion loss
            loss_sum = pos_loss.item() + quat_loss.item()
            weight_pos_loss = quat_loss.item() / (loss_sum + 1e-8)
            weight_quat_loss = pos_loss.item() / (loss_sum + 1e-8)
        else:
            weight_pos_loss = 1
            weight_quat_loss = 1

        loss = weight_pos_loss * pos_loss + weight_quat_loss * quat_loss + args.ent_coef * entropy_loss
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Record loss
        pose_loss_record += pos_loss.item() + quat_loss.item()
        pos_loss_record += pos_loss.item()
        quat_loss_record += quat_loss.item()
        entropy_record += -entropy_loss.item()
    pose_loss_record /= len(sp_train_dataloader)
    pos_loss_record /= len(sp_train_dataloader)
    quat_loss_record /= len(sp_train_dataloader)
    entropy_record /= len(sp_train_dataloader)
    
    if epoch % args.val_epochs == 0:
        val_loss = 0; val_pos_loss = 0; val_quat_loss = 0; sim_success_rate = 0
        with torch.no_grad():
            for batch in tqdm(sp_val_dataloader, desc=f'Epoch {epoch} Validation'):
                scene_pc = batch['scene_pc'].to(device)
                qr_obj_pc = batch['qr_obj_pc'].to(device)
                qr_obj_pose = batch['qr_obj_pose'].to(device)
                pred_pose, _ = model(scene_pc, qr_obj_pc)
                pos_loss = mse_loss(pred_pose[:, :3], qr_obj_pose[:, :3])
                quat_loss = mse_loss(pred_pose[:, 3:], qr_obj_pose[:, 3:])
                val_loss += pos_loss.item() + quat_loss.item()
                val_pos_loss += pos_loss.item()
                val_quat_loss += quat_loss.item()
            val_loss /= len(sp_val_dataloader)
            val_pos_loss /= len(sp_val_dataloader)
            val_quat_loss /= len(sp_val_dataloader)

            if args.use_simulator and epoch % args.val_sim_epochs == 0:
                # Evaluate the model
                # Scene and obj feature tensor are keeping updated inplace]
                success_sum = 0; eval_trials = 1000
                for eval_index, sim_batch in enumerate(tqdm(sp_sim_dataloader, desc=f'Epoch {epoch} Simulator Evaluation')):
                    if eval_index >= eval_trials:
                        break
                    ################ agent evaluation ################
                    envs.reset()
                    sp_placed_obj_poses = sim_batch['World2PlacedObj_poses'][0]
                    for obj_name, obj_pose in sp_placed_obj_poses.items():
                        sp_placed_obj_poses[obj_name] = pu.split_7d(obj_pose)
                    
                    scene_pc = sim_batch['scene_pc'].to(device)
                    qr_obj_pc = sim_batch['qr_obj_pc'].to(device)
                    qr_obj_pose = sim_batch['qr_obj_pose'].to(device)
                    pred_qr_obj_pose, _ = model(scene_pc, qr_obj_pc)

                    qr_obj_name = sim_batch['qr_obj_name'][0]
                    # Use qr_obj_pose to test the success rate is correct or not
                    success = envs.stable_placement_eval_step(qr_obj_name, pred_qr_obj_pose, sp_placed_obj_poses)
                    success_sum += success
                        
                sim_success_rate = success_sum / eval_index
                print(f"Epoch {epoch}: Real Simulator Success Rate={sim_success_rate}")
                if args.collect_data and args.wandb:
                    wandb.log({
                        'epoch': epoch,
                        'eval/sim_success_rate': sim_success_rate,
                    }, commit=False)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.collect_data:
                torch.save(model.state_dict(), os.path.join(args.model_save_path, 'epoch_best.pth'))
        
        if sim_success_rate > best_sim_success_rate:
            best_sim_success_rate = sim_success_rate
            if args.collect_data:
                torch.save(model.state_dict(), os.path.join(args.model_save_path, 'epoch_simbest.pth'))
        
        if epoch % args.save_epoch == 0 and args.collect_data:
            torch.save(model.state_dict(), os.path.join(args.model_save_path, f'epoch_{epoch}.pth'))
        
        if args.collect_data and args.wandb:
            wandb.log({
                'epoch': epoch,
                'eval/val_loss': val_loss,
                'eval/val_pos_loss': val_pos_loss,
                'eval/val_quat_loss': val_quat_loss,
                'eval/best_val_loss': best_val_loss,
            }, commit=False)
    
    print(f'Epoch {epoch}: Training Loss={pose_loss_record} | Best Validation Loss={best_val_loss}')

    if args.collect_data and args.wandb:
        wandb.log({
            'epoch': epoch,
            'train/learning_rate': optimizer.param_groups[0]["lr"], # 'param_group' is a list of dict, each dict is a group of parameters
            'train/pose_loss_record': pose_loss_record, 
            'train/pos_loss_record': pos_loss_record,
            'train/quat_loss_record': quat_loss_record,
            'train/entropy_record': entropy_record,
            })

if args.collect_data and args.wandb:
    wandb.finish()