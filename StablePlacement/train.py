import os
import datetime

import torch
from torch.utils.data import DataLoader
from DataLoader import HDF5Dataset
from sp_model import StablePlacementModel
from torch.optim import Adam
from torch.nn.functional import mse_loss

from tqdm import tqdm
import wandb
import argparse
from distutils.util import strtobool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='StablePlacement')
    parser.add_argument('--collect_data', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Save dataset or not')
    parser.add_argument('--main_dataset_path', type=str, default='SP_Dataset/table_10_group0_dinning_table.h5')
    parser.add_argument('--model_save_folder', type=str, default='SP_Model')
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--val_epochs', type=int, default=5, help='')
    parser.add_argument('--batch_size', type=int, default=10, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    args = parser.parse_args()
    timer = '_' + '_'.join(str(datetime.datetime.now())[5:16].split())  # a time name file
    args.final_name = args.env_name + timer
    args.model_save_path = os.path.join(args.model_save_folder, args.final_name)

    return args

args = parse_args()
# Create folder to save the model
if args.collect_data and not os.path.exists(args.model_save_path):
    os.makedirs(args.model_save_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_dataset_path = args.main_dataset_path
train_dataset_path = main_dataset_path.replace('.h5', '_train.h5')
val_dataset_path = main_dataset_path.replace('.h5', '_val.h5')
sp_train_dataloader = DataLoader(HDF5Dataset(train_dataset_path), batch_size=args.batch_size, shuffle=True)
sp_val_dataloader = DataLoader(HDF5Dataset(val_dataset_path), batch_size=args.batch_size, shuffle=False)

model = StablePlacementModel(device=device).to(device)
optimizer = Adam(model.parameters(), lr=args.lr)

# Configure Wandb 

wb_config = dict(
    Name=args.env_name,
    Model='PointNet_MLP',
    Learning_rate=args.lr,
)

if args.collect_data:
    wandb.init(project=args.env_name, entity='jiayinsen', config=wb_config, name=args.final_name)

best_val_loss = float('inf')
for epoch in range(1, args.epochs+1):
    # Learning rate decay; Linear Schedular
    optimizer.param_groups[0]['lr'] = args.lr * (1 - epoch / args.epochs)

    train_loss = 0
    for batch in tqdm(sp_train_dataloader):
        scene_pc = batch['scene_pc'].to(device)
        qr_obj_pc = batch['qr_obj_pc'].to(device)
        qr_obj_pose = batch['qr_obj_pose'].to(device)
        # Forward pass
        pred_pose = model(scene_pc, qr_obj_pc)
        # Compute loss
        loss = mse_loss(pred_pose, qr_obj_pose)
        train_loss += loss.item()
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(sp_train_dataloader)
    
    if epoch % args.val_epochs == 0:
        val_loss = 0
        with torch.no_grad():
            for batch in sp_val_dataloader:
                scene_pc = batch['scene_pc'].to(device)
                qr_obj_pc = batch['qr_obj_pc'].to(device)
                qr_obj_pose = batch['qr_obj_pose'].to(device)
                pred_pose = model(scene_pc, qr_obj_pc)
                val_loss += mse_loss(pred_pose, qr_obj_pose).item()
            val_loss /= len(sp_val_dataloader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.collect_data:
                torch.save(model.state_dict(), os.path.join(args.model_save_path, 'best.pth'))
        
        wandb.log({
            'epoch': epoch,
            'train/val_loss': val_loss,
            'train/best_val_loss': best_val_loss,
        }, commit=False)
    
    print(f'Epoch {epoch}: Training Loss={train_loss} | Best Validation Loss={best_val_loss}')
    wandb.log({
        'epoch': epoch,
        'train/learning_rate': optimizer.param_groups[0]["lr"], # 'param_group' is a list of dict, each dict is a group of parameters
        'train/train_loss': train_loss, 
        })
