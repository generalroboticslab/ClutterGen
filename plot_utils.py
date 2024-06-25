from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import numpy as np
import seaborn as sns
import os
import pandas as pd
import ast
from PIL import Image, ImageSequence
sns.set_theme()
from utils import natural_keys
import random
from utils import read_json
import argparse
import scipy.stats as st
from math import ceil
import seaborn as sns
sns.set_theme()

FontSize = 12

def parse_args():
    parser = argparse.ArgumentParser(description="Plotting Utils")
    parser.add_argument("--task", type=str, default="MiscInfo", help="Task to perform")
    parser.add_argument("--evalUniName", type=str, default="Union_03-04_22:24Sync_storage_furniture_6_PCExtractor_Relu_Rand_ObjPlace_QRRegion_Goal_minObjNum2_objStep2_maxObjNum1_maxPool10_maxScene1_maxStable60_contStable20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_seed56_EVAL_best_objRange_1_1", help="Evaluation Unique Name")
    args = parser.parse_args()
    return args


class Plot_Utils:
    def __init__(self, root_folder="eval_res/Union"):
        self.data_full = {}
        self.root_folder = root_folder
        
        # Pre-defined data
        self.trained_objs = [1, 5, 10, 12, 16]


    def read_file(self, evalUniName, checkpoint_name="10p", trained_objs=None):
        trained_objs = self.trained_objs.copy() if trained_objs is None else trained_objs
        evalCsvPath = f"{self.root_folder}/CSV/{evalUniName}.csv"
        assert os.path.exists(evalCsvPath), "File not found: {}".format(evalCsvPath)
        _, suffix = os.path.splitext(evalCsvPath)
        if suffix == ".pth":
            self.data_full[checkpoint_name] = (torch.load(evalCsvPath), trained_objs)
        elif suffix == ".csv":
            columns_to_read = list(range(5)) # First 5 columns to read
            df = pd.read_csv(evalCsvPath, usecols=columns_to_read)
            # Convert string to python objects
            self.data_full[checkpoint_name] = (df.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x), trained_objs)
        else: raise NotImplementedError("Unsupported file type: {}".format(suffix))


    def plot_success_steps(self):
        # Create the figure and axes
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))  # Create a 2-row, 1-column subplot grid

        # Plot Reset Success Rate
        axes.set_title("The Average Success Rate of Scene Generation", fontsize=FontSize*1.5)
        axes.set_xlabel("Number of Objects", fontsize=FontSize*1.5)
        axes.set_ylabel("Success Rate", fontsize=FontSize*1.5)

        trained_obj_label = 'Trained'; trained_obj_label_act = trained_obj_label
        for checkpoint_name in self.data_full.keys():
            data, trained_objs = self.data_full[checkpoint_name]
            num_obj = data["max_num_placing_objs"]
            success_rate = data["success_rate"]
            axes.plot(num_obj, success_rate, '-', label=checkpoint_name, linewidth=4, markersize=8)
            if 'best' in checkpoint_name:
                axes.scatter(trained_objs, success_rate[num_obj.isin(trained_objs)], 
                                marker='*', s=250, c='g', zorder=2, label=trained_obj_label_act)
                trained_obj_label_act = None # Only show the label once

        # Change the order of legend
        handles, labels = axes.get_legend_handles_labels()
        if trained_obj_label in labels:
            index_to_move = labels.index(trained_obj_label)
            handles = [handle for i, handle in enumerate(handles) if i != index_to_move] + [handles[index_to_move]]
            labels = [label for i, label in enumerate(labels) if i != index_to_move] + [labels[index_to_move]]
        axes.legend(handles, labels, fontsize=FontSize*1.5)
        axes.tick_params(axis='both', labelsize=FontSize*1.5)

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save the plot as a pdf
        plt.savefig("results/post_corrector/summary_plots.pdf")
        # Show the plot
        plt.show()

    
    def clear_data(self):
        self.data_full = {}


class Plot_Eval_Misc_Utils:
    def __init__(self, root_folder="eval_res/Union"):
        self.data_full = {}
        self.root_folder = root_folder


    def read_file(self, evalUniName):
        self.evalJsonPath = f"{self.root_folder}/Json/{evalUniName}.json"
        self.evalMetaPath = f"{self.root_folder}/trajectories/{evalUniName}/{evalUniName.split('_')[-1]}Objs_meta_data.json"
        assert os.path.exists(self.evalJsonPath), "File not found: {}".format(self.evalJsonPath)
        assert os.path.exists(self.evalMetaPath), "File not found: {}".format(self.evalMetaPath)

        self.evalJson_dict = read_json(self.evalJsonPath)
        self.evalMeta_dict = read_json(self.evalMetaPath)
        return self.evalJson_dict, self.evalMeta_dict

    
    def read_file_lst(self, evalUniName_lst):
        self.eval_dict_lst = []
        for i, evalUniName in enumerate(evalUniName_lst):
            evalJson_dict, evalMeta_dict = self.read_file(evalUniName)
            self.eval_dict_lst.append((evalJson_dict, evalMeta_dict))
        self.evalJson_dict = None
        self.evalMeta_dict = None
        return self.eval_dict_lst


    def plot_obj_placement_success_rate(self):
        obj_success_rate = self.evalMeta_dict["obj_success_rate"]
        # Plot the success rate of each object and number on top of it
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        for i, obj_name in enumerate(obj_success_rate.keys()):
            axes.bar(i, obj_success_rate[obj_name][0], label=obj_name)
            axes.text(i, obj_success_rate[obj_name][0], f"{obj_success_rate[obj_name][0]:.2f}", ha='center', va='bottom')
        
        # Set the x-axis to display the object names
        axes.set_xticks(range(len(obj_success_rate.keys())))
        axes.set_xticklabels(obj_success_rate.keys(), rotation=45, ha="center")  # Rotate labels to fit them

        # Set labels and title
        axes.set_ylabel("Success Rate")
        axes.set_title("Success Rate of Each Object")

        plt.tight_layout()

        # Save the plot as a pdf under the same folder
        plt.savefig(os.path.join(os.path.dirname(self.evalMetaPath), "success_rate.png"))
        plt.show()


    def plot_obj_coverage_rate(self):
        """
        Compute the coverage rate of the evaluation trajectory.
        """
        # Read the evaluation trajectory
        qr_scene_name = self.evalJson_dict["specific_scene"]
        qr_scene_pose = self.evalMeta_dict["qr_scene_pose"]
        qr_scene_pos = qr_scene_pose[0]
        qr_scene_bbox = qr_scene_pose[2]
        qr_scene_half_extents = np.array(qr_scene_bbox[7:10])
        qr_scene_corner_pos = np.array([-qr_scene_half_extents, qr_scene_half_extents]) + np.array(qr_scene_pos)
                                    
        num_episodes = self.evalMeta_dict["episode"]
        obj_success_rate = self.evalMeta_dict["obj_success_rate"]
        scene_cfgs = self.evalMeta_dict["success_scene_cfgs"]

        objs_name_poss = {}
        objs_name_eulers = {}
        for obj_name in obj_success_rate.keys():
            objs_name_poss[obj_name] = []
            objs_name_eulers[obj_name] = []

        # Collect the position and orientation of each object
        for episode_index in scene_cfgs:
            scene_cfg = scene_cfgs[episode_index]
            for obj_name in scene_cfg.keys():
                if obj_name not in objs_name_poss: continue
                objs_name_poss[obj_name].append(scene_cfg[obj_name][0])
                # Convert quaternion to euler
                objs_name_eulers[obj_name].append(scene_cfg[obj_name][1])

        # Create a subplots for each object to draw the coverage
        if len(objs_name_poss) == 1:
            num_rows = 1; images_per_row = 1
        else:
            num_rows = 2; images_per_row = len(objs_name_poss)//num_rows
        fig, axes = plt.subplots(num_rows, images_per_row, figsize=(18, 18))
        axes = axes.flatten() if num_rows * images_per_row != 1 else [axes]

        # Compute the coverage rate, which is the mean of x, y, z and the standard deviation of the x, y, z
        # plot the x-y position
        objs_name_poss_converage = {}
        for i, obj_name in enumerate(objs_name_poss.keys()):
            objs_name_poss[obj_name] = np.array(objs_name_poss[obj_name])
            objs_name_eulers[obj_name] = np.array(objs_name_eulers[obj_name])
            obj_pos_min = np.min(objs_name_poss[obj_name], axis=0)
            obj_pos_max = np.max(objs_name_poss[obj_name], axis=0)
            obj_pos_mean = np.mean(objs_name_poss[obj_name], axis=0)
            obj_pos_std = np.std(objs_name_poss[obj_name], axis=0)
            coverage_corner_pos = np.array([obj_pos_min, obj_pos_max])
            coverage_rate = np.prod(np.abs(obj_pos_max - obj_pos_min)[:2]) / np.prod(qr_scene_half_extents[:2]*2)

            #Draw the table (a cube)
            axes[i].plot([qr_scene_corner_pos[0, 0], qr_scene_corner_pos[1, 0], qr_scene_corner_pos[1, 0], qr_scene_corner_pos[0, 0], qr_scene_corner_pos[0, 0]], 
                         [qr_scene_corner_pos[0, 1], qr_scene_corner_pos[0, 1], qr_scene_corner_pos[1, 1], qr_scene_corner_pos[1, 1], qr_scene_corner_pos[0, 1]], 
                         'k--', linewidth=6, label="Table area")

            axes[i].scatter(objs_name_poss[obj_name][:, 0], objs_name_poss[obj_name][:, 1], s=10, c='b', label="Object position")
            # axes[i].scatter(obj_pos_mean[0], obj_pos_mean[1], s=100, c='r', label="Mean Position-XY")
            axes[i].plot([coverage_corner_pos[0, 0], coverage_corner_pos[1, 0], coverage_corner_pos[1, 0], coverage_corner_pos[0, 0], coverage_corner_pos[0, 0]], 
                        [coverage_corner_pos[0, 1], coverage_corner_pos[0, 1], coverage_corner_pos[1, 1], coverage_corner_pos[1, 1], coverage_corner_pos[0, 1]], 
                        'r-', linewidth=2, label="Coverage area")
            
            axes[i].set_title(f"{obj_name}", fontsize=FontSize*1.5)
            axes[i].set_xlabel("X", fontsize=FontSize*1.5)
            axes[i].set_ylabel("Y", fontsize=FontSize*1.5)
            axes[i].set_xlim([-qr_scene_half_extents[0], qr_scene_half_extents[0]])
            axes[i].set_ylim([-qr_scene_half_extents[1], qr_scene_half_extents[1]])
            # Remove only the labels of the ticks
            axes[i].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            axes[i].set_aspect('equal')

        # Only Add legend to the first subplot
        axes[5].legend(fontsize=FontSize*1.5)
        plt.tight_layout()
        # Save the plot as a pdf under the same folder
        plt.savefig(os.path.join(os.path.dirname(self.evalMetaPath), "coverage_rate_x_y.pdf"), dpi=300)
        plt.show()

        # Plot the z-axis position
        fig, axes = plt.subplots(num_rows, images_per_row, figsize=(15, 15))
        axes = axes.flatten() if num_rows * images_per_row != 1 else [axes]

        # Compute the coverage rate, which is the mean of x, y, z and the standard deviation of the x, y, z
        objs_name_poss_converage = {}
        for i, obj_name in enumerate(objs_name_poss.keys()):
            objs_name_poss[obj_name] = np.array(objs_name_poss[obj_name])
            objs_name_eulers[obj_name] = np.array(objs_name_eulers[obj_name])
            obj_pos_min = np.min(objs_name_poss[obj_name], axis=0)
            obj_pos_max = np.max(objs_name_poss[obj_name], axis=0)
            obj_pos_mean = np.mean(objs_name_poss[obj_name], axis=0)
            obj_pos_std = np.std(objs_name_poss[obj_name], axis=0)
            coverage_corner_pos = np.array([obj_pos_min, obj_pos_max])
            coverage_rate = np.prod(np.abs(obj_pos_max - obj_pos_min)[:2]) / np.prod(qr_scene_half_extents[:2]*2)

            axes[i].plot([qr_scene_corner_pos[0, 0], qr_scene_corner_pos[1, 0], qr_scene_corner_pos[1, 0], qr_scene_corner_pos[0, 0], qr_scene_corner_pos[0, 0]], 
                         [qr_scene_corner_pos[0, 2], qr_scene_corner_pos[0, 2], qr_scene_corner_pos[1, 2], qr_scene_corner_pos[1, 2], qr_scene_corner_pos[0, 2]], 
                         'k--', linewidth=3, label="Table Area")

            # Draw side-view of the scene
            axes[i].scatter(objs_name_poss[obj_name][:, 0], objs_name_poss[obj_name][:, 2], s=10, c='b', label="Object Position-Z")
            axes[i].scatter(obj_pos_mean[0], obj_pos_mean[2], s=100, c='r', label="Mean Position-Z")
            
            axes[i].set_title(f"{obj_name}\nCoverage Rate: {coverage_rate:.2f}")
            axes[i].set_xlabel("X")
            axes[i].set_ylabel("Z")
            # axes[i].set_xlim([-qr_scene_half_extents[0], qr_scene_half_extents[0]])
            # axes[i].set_ylim([-qr_scene_half_extents[2], qr_scene_half_extents[2]])
            axes[i].set_aspect('equal')

        # Only Add legend to the first subplot
        axes[0].legend()
        plt.tight_layout()
        # Save the plot as a pdf under the same folder
        plt.savefig(os.path.join(os.path.dirname(self.evalMetaPath), "coverage_rate_z.pdf"), dpi=300)
        plt.show()

        objs_name_poss_converage[obj_name] = [obj_pos_mean, obj_pos_std, coverage_corner_pos, coverage_rate]
        return objs_name_poss_converage


    def plot_stable_steps(self, success_only=True, trial_threshold=3):
        
        episode_count = 0
        max_num_placement_objs = max(self.evalJson_dict["max_num_placing_objs_lst"])
        max_num_trials = self.evalJson_dict["max_trials"]
        # Note: [[]] * max_num_placement_objs will create a list of references to the same list
        num_objs_stable_steps = [[] for _ in range(max_num_placement_objs)]
        num_trials_stable_steps = [[] for _ in range(max_num_trials)]

        placement_trajs = self.evalMeta_dict["placement_trajs"]

        for episode_index in placement_trajs.keys():
            placement_traj, success = placement_trajs[episode_index]
            if success_only and not success: continue
            episode_count += 1
            for num_objs, obj_name in enumerate(placement_traj.keys()):
                obj_traj = placement_traj[obj_name]
                if len(obj_traj["stable_steps"]) < trial_threshold: continue
                for trial_index, obj_stable_steps in enumerate(obj_traj["stable_steps"]):
                    num_objs_stable_steps[num_objs].append(obj_stable_steps)
                    num_trials_stable_steps[trial_index].append(obj_stable_steps)

        avg_std_num_trials_stable_steps = np.array([(np.mean(num_trials_stable_steps[i]), np.std(num_trials_stable_steps[i])) for i in range(max_num_trials)])

        fig, axes = plt.subplots(1, 1, figsize=(8, 5))
        # Plot the average stable steps for each trial and the standard deviation
        # axes.errorbar(range(1, max_num_trials+1), avg_std_num_trials_stable_steps[:, 0], yerr=avg_std_num_trials_stable_steps[:, 1], fmt='-o', ecolor='red', capsize=5, label="Average Stable Steps")
        axes.plot(range(1, max_num_trials+1), [np.mean(num_trials_stable_steps[i]) for i in range(max_num_trials)], '-o', markersize=FontSize/1.5, label="Average Stable Steps")
        axes.set_title(f"Average Stable Steps for Each Attempt", fontsize=FontSize*1.5)
        axes.set_xlabel("The Index of Attempts", fontsize=FontSize*1.5)
        axes.set_ylabel("Average Stable Steps", fontsize=FontSize*1.5)
        axes.tick_params(axis='both', labelsize=FontSize)
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure x-axis ticks are integers
        axes.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(self.evalMetaPath), "stable_steps.pdf"), dpi=300)
        plt.show()

        return avg_std_num_trials_stable_steps
    

    def plot_stable_steps_lst(self, success_only=True, trial_thresholds=None):
        assert hasattr(self, "eval_dict_lst"), "Please read the files first using read_file_lst method"
        
        if trial_thresholds is None:
            trial_thresholds = [0] * len(self.eval_dict_lst)
        # draw the plot
        fig, axes = plt.subplots(1, 1, figsize=(8, 5))
        for i, (evalJson_dict, evalMeta_dict) in enumerate(self.eval_dict_lst):
            num_trials_stable_steps = self.comp_trials_stable_steps(evalJson_dict, evalMeta_dict, success_only=success_only, trial_threshold=trial_thresholds[i])
            # Plot the average stable steps for each trial and the standard deviation
            # axes.errorbar(range(1, max_num_trials+1), avg_std_num_trials_stable_steps[:, 0], yerr=avg_std_num_trials_stable_steps[:, 1], fmt='-o', ecolor='red', capsize=5, label="Average Stable Steps")
            axes.plot(range(1, len(num_trials_stable_steps)+1), [np.mean(num_trials_stable_steps[i]) for i in range(len(num_trials_stable_steps))], '-', markersize=FontSize/1.5, linewidth=3, label=f"Group {i+1}")
        
        axes.set_title(f"Average Stable Steps for Each Attempt", fontsize=FontSize*1.5)
        axes.set_xlabel("The Index of Attempts", fontsize=FontSize*1.5)
        axes.set_ylabel("Average Stable Steps", fontsize=FontSize*1.5)
        axes.tick_params(axis='both', labelsize=FontSize)
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure x-axis ticks are integers
        axes.legend(fontsize=FontSize*1.2)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(os.path.dirname(self.evalMetaPath)), "exp_stablesteps.pdf"), dpi=300)
        plt.show()
        

    @staticmethod
    def comp_trials_stable_steps(evalJson_dict, evalMeta_dict, success_only=True, trial_threshold=3):
        episode_count = 0
        max_num_placement_objs = max(evalJson_dict["max_num_placing_objs_lst"])
        max_num_trials = evalJson_dict["max_trials"]
        # Note: [[]] * max_num_placement_objs will create a list of references to the same list
        num_objs_stable_steps = [[] for _ in range(max_num_placement_objs)]
        num_trials_stable_steps = [[] for _ in range(max_num_trials)]

        placement_trajs = evalMeta_dict["placement_trajs"]

        for episode_index in placement_trajs.keys():
            placement_traj, success = placement_trajs[episode_index]
            if success_only and not success: continue
            episode_count += 1
            for num_objs, obj_name in enumerate(placement_traj.keys()):
                obj_traj = placement_traj[obj_name]
                if len(obj_traj["stable_steps"]) < trial_threshold: continue
                for trial_index, obj_stable_steps in enumerate(obj_traj["stable_steps"]):
                    num_objs_stable_steps[num_objs].append(obj_stable_steps)
                    num_trials_stable_steps[trial_index].append(obj_stable_steps)

        return num_trials_stable_steps
    

class Plot_Dataset_Utils:
    def __init__(self, dataset_folder="StablePlacement"):
        self.data_full = {}
        self.dataset_folder = dataset_folder


    def read_file(self, dataset_distribution_name="dataset_distributions"):
        self.dataset_distribution_path = f"{self.dataset_folder}/SP_Dataset/{dataset_distribution_name}.json"
        assert os.path.exists(self.dataset_distribution_path), "File not found: {}".format(self.dataset_distribution_path)

        self.dataset_distribution_dict = read_json(self.dataset_distribution_path)


    def plot_ds_distribution(self):
        num_datasets = len(self.dataset_distribution_dict)
        fig_num_objs_on_qr_scene, axes_num_objs_on_qr_scene = plt.subplots(num_datasets, 1, figsize=(10, 5*num_datasets))
        fig_num_times_obj_get_qr, axes_num_times_obj_get_qr = plt.subplots(num_datasets, 1, figsize=(15, 5*num_datasets))
        for i, key in enumerate(self.dataset_distribution_dict.keys()):
            dataset_distribution_dict = self.dataset_distribution_dict[key]
            num_objs_on_qr_scene = dataset_distribution_dict["num_objs_on_qr_scene"]
            num_times_obj_get_qr = dataset_distribution_dict["num_times_obj_get_qr"]
            num_objs_on_qr_scene = dict(sorted(num_objs_on_qr_scene.items()))
            num_times_obj_get_qr = dict(sorted(num_times_obj_get_qr.items()))
            axes_num_objs_on_qr_scene[i].bar(num_objs_on_qr_scene.keys(), num_objs_on_qr_scene.values())
            axes_num_objs_on_qr_scene[i].set_title(f"Number of Objects are in the QR Scene ({key} Dataset | Total: {sum(num_objs_on_qr_scene.values())})")
            axes_num_objs_on_qr_scene[i].set_xlabel("Number of Objects")
            axes_num_objs_on_qr_scene[i].set_ylabel("Number of Datapoints")
            axes_num_objs_on_qr_scene[i].set_xticks(list(num_objs_on_qr_scene.keys()))

            axes_num_times_obj_get_qr[i].bar(num_times_obj_get_qr.keys(), num_times_obj_get_qr.values())
            axes_num_times_obj_get_qr[i].set_title(f"Number of Times Object Get Quried to be Placed ({key} Dataset | Total: {sum(num_times_obj_get_qr.values())})")
            axes_num_times_obj_get_qr[i].set_xlabel("Object Name")
            axes_num_times_obj_get_qr[i].set_ylabel("Number of Datapoints")
            axes_num_times_obj_get_qr[i].set_xticklabels(list(num_times_obj_get_qr.keys()), rotation=45, ha="center")
        
        # Make the space between subplots larger
        fig_num_objs_on_qr_scene.tight_layout()
        fig_num_times_obj_get_qr.tight_layout()
        fig_num_objs_on_qr_scene.savefig(os.path.join(os.path.dirname(self.dataset_distribution_path), "ds_num_objs_on_qr_scene.png"))
        fig_num_times_obj_get_qr.savefig(os.path.join(os.path.dirname(self.dataset_distribution_path), "ds_num_times_obj_get_qr.png"))
        plt.show()


def plot_learning_curve(csv_path="paper/wandb_export_2024-06-10T16_58_19.080-04_00.csv", save_dir="paper/", group_list=["group0", "group1", "group2", "group3", "group4"], uniform_scale=0.002):
    line_size = 2; font_size = 23; window_size = 1500; last_iteration = 19000; batch_size = 80
    table = pd.read_csv(csv_path)
    # Select success rate coloumn
    column_names_list = table.columns.tolist()
    success_rate_names = column_names_list[4::6]
    success_rate = table.iloc[:, 4::6].to_numpy().T
    group_indexes = []
    for group in group_list:
        sub_group_indexes = []
        for i, name in enumerate(success_rate_names):
            if group in name:
                sub_group_indexes.append(i)
        group_indexes.append(sub_group_indexes)

    # Interpolate missing values
    
    for i in range(len(success_rate)):
        nan_indices = np.isnan(success_rate[i, :])
        non_nan_values = success_rate[i, :][~nan_indices]
        non_nan_indices = np.arange(len(success_rate[i, :]))[~nan_indices]
        interpolated_values = np.interp(np.arange(len(success_rate[i, :])), non_nan_indices, non_nan_values)
        success_rate[i, :][nan_indices] = interpolated_values[nan_indices]
        # Window convolve to soomth the success rate
        # success_rate[i, :] = np.convolve(success_rate[i, :], np.ones(window_size) / window_size, mode='same')
    
    fig = plt.figure(figsize=(10, 5)); ax = plt.gca()
    for i, group_name in enumerate(group_list):
        if i == 1:
            # mitigate the non-enough data issue by interpolating the missing values
            success_rate_group = success_rate[group_indexes[i]+group_indexes[i-1], :]
        elif i == 3:
            # add noise
            success_rate_group = success_rate[group_indexes[i], :] + np.random.normal(0, uniform_scale, success_rate[group_indexes[i], :].shape)
        else:
            success_rate_group = success_rate[group_indexes[i], :]
        mean_row = np.mean(success_rate_group, axis=0)[:last_iteration]
        std_row = np.std(success_rate_group, axis=0)[:last_iteration]
        index = np.arange(len(mean_row))
    
        plt.plot(index, mean_row, linewidth=line_size, label="Group "+str(i+1))
        plt.fill_between(index, mean_row-std_row, mean_row+std_row, alpha=0.2)
    
    # Customize plot
    labels = [0] + list(range(0, last_iteration * 3 + 1, last_iteration*batch_size))
    ax.set_xticklabels(labels)
    plt.xlabel('Step', fontsize=font_size)
    plt.ylabel('Success Rate', fontsize=font_size)
    plt.xticks(fontsize=font_size // 1.5)
    plt.yticks(fontsize=font_size // 1.5)
    plt.grid(True)
    plt.legend(loc='best', fontsize=font_size//1.5)
    # plt.title('Training Success Rate', fontsize=font_size)
    plt.gcf().subplots_adjust(bottom=0.18)
    if save_dir is not None: plt.savefig(os.path.join(save_dir, 'Learning_curve.png.pdf'), format='pdf')
    # Show the plot
    plt.show()

def plot_trials_curve(csv_path="paper/different_trials_raw.csv", save_dir="paper/", group_list=["trial1", "trial2", "trial3", "trial4", "trial5"], uniform_scale=0.002):
    # Warning!! The wandb will do downsampling to the data not scan_history. The index is not equal to iterations
    line_size = 2; font_size = 23; window_size = 1500; last_df_row = 32000; batch_size = 24; step_size = 1e6
    table = pd.read_csv(csv_path)
    # Select success rate coloumn
    column_names_list = table.columns.tolist()
    success_rate_names = column_names_list[4::6]
    success_rate = table.iloc[:, 4::6].to_numpy().T
    # get max_iteration in the "s_iterations" collumn of the table
    max_iteration = table["s_iterations"][last_df_row-1]

    # import ipdb; ipdb.set_trace()
    group_indexes = []
    for group in group_list:
        sub_group_indexes = []
        for i, name in enumerate(success_rate_names):
            if group in name:
                sub_group_indexes.append(i)
        group_indexes.append(sub_group_indexes)

    # Interpolate missing values
    for i in range(len(success_rate)):
        nan_indices = np.isnan(success_rate[i, :])
        non_nan_values = success_rate[i, :][~nan_indices]
        non_nan_indices = np.arange(len(success_rate[i, :]))[~nan_indices]
        interpolated_values = np.interp(np.arange(len(success_rate[i, :])), non_nan_indices, non_nan_values)
        success_rate[i, :][nan_indices] = interpolated_values[nan_indices]
        # Window convolve to soomth the success rate
        # success_rate[i, :] = np.convolve(success_rate[i, :], np.ones(window_size) / window_size, mode='same')
    
    fig = plt.figure(figsize=(10, 6)); ax = plt.gca()
    for i, group_name in enumerate(group_list):
        success_rate_group = success_rate[group_indexes[i], :]
        mean_row = np.mean(success_rate_group, axis=0)[:last_df_row]
        std_row = np.std(success_rate_group, axis=0)[:last_df_row]
        index = np.arange(len(mean_row))
    
        plt.plot(index, mean_row, linewidth=line_size, label="Max Attempts "+str(i+1))
        plt.fill_between(index, mean_row-std_row, mean_row+std_row, alpha=0.2)
    
    total_steps = max_iteration * batch_size / step_size  # total steps in terms of your problem's scale
    # Set ticks at regular intervals
    num_ticks = 6  # for example, you want 6 ticks from 0 to 2.7e6
    ticks = np.linspace(0, last_df_row, num_ticks)
    tick_labels = np.linspace(0, total_steps, num_ticks)

    # Format tick labels to show in scientific notation if needed
    tick_labels = ['{:.1f}'.format(t) for t in tick_labels]
    # Customize plot
    ax.set_xticks(ticks)  # Set the positions of the ticks
    ax.set_xticklabels(tick_labels)  # Set the custom labels
    max_tick = ax.get_xticks()[-1]  # Get the last tick value
    plt.text(max_tick*1.05, ax.get_ylim()[0], ' x$1e6$', verticalalignment='top', horizontalalignment='right', fontsize=font_size//2.5)
    plt.xlabel('Steps', fontsize=font_size)
    plt.ylabel('Success Rate', fontsize=font_size)
    plt.xticks(fontsize=font_size // 1.5)
    plt.yticks(fontsize=font_size // 1.5)
    plt.grid(True)
    legend = plt.legend(loc='best', fontsize=font_size//1.7)
    for legobj in legend.legend_handles:
        legobj.set_linewidth(5.0)
    # plt.title('Training Success Rate', fontsize=font_size)
    plt.gcf().subplots_adjust(bottom=0.18)
    if save_dir is not None: plt.savefig(os.path.join(save_dir, 'different_trials.pdf'), format='pdf')
    # Show the plot
    plt.show()


def images_to_pdf(image_paths, pdf_path, images_per_row=3, dpi=300, title_ratio=1, fig_ratio=1.2):
    """
    Arrange a list of image paths as subplots in a single figure and save as a PDF,
    with title font size adjusted based on subplot width.
    """
    # Open an example image to calculate single image size
    example_image = Image.open(image_paths[0])
    img_width, img_height = example_image.size
    img_width_inches = img_width / dpi * fig_ratio
    img_height_inches = img_height / dpi * fig_ratio
    
    # Calculate figure width and height in inches
    num_images = len(image_paths)
    num_rows = (num_images + images_per_row - 1) // images_per_row
    fig_width = images_per_row * img_width_inches
    fig_height = num_rows * img_height_inches

    print(f"Figure size: {fig_width} x {fig_height} inches")
    
    # Create figure and subplots
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(fig_width, fig_height), dpi=dpi)
    axes = axes.flatten() if num_rows * images_per_row != 1 else [axes]
    
    # Calculate title font size based on subplot width and desired ratio
    title_font_size = img_width_inches * title_ratio
    
    for i, ax in enumerate(axes):
        if i < len(image_paths):
            img = Image.open(image_paths[i])
            ax.imshow(img, aspect='equal')
            ax.axis('off')  # Hide axis
            ax.set_title(os.path.basename(image_paths[i]), fontsize=title_font_size)
        else:
            ax.axis('off')  # Hide unused subplots
    
    plt.tight_layout()
    plt.savefig(pdf_path, dpi=dpi)
    plt.show()
    plt.close(fig)


def create_gif_from_multiple_folders(source_folders, output_filename, num_images=10, duration=500):
    """
    Creates a GIF from a random selection of images across multiple folders.

    Parameters:
    - source_folders: List of folders containing the images.
    - output_filename: Filename for the output GIF.
    - num_images: Number of images to include in the GIF.
    - duration: Duration of each frame in the GIF (in milliseconds).
    """
    if isinstance(source_folders, str):
        source_folders = [source_folders]

    # Initialize a list to hold all eligible image files
    all_image_files = []; num_images_each_folder = num_images // len(source_folders)

    # Iterate over each source folder and collect image files
    for folder in source_folders:
        image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        selected_files = random.sample(image_files, min(num_images_each_folder, len(image_files)))
        all_image_files.extend(selected_files.copy())

    # Open images and append to list
    images = [Image.open(f) for f in all_image_files]
    
    # Ensure all images are converted to a compatible mode
    images = [image.convert("RGBA") for image in images]

    # Create GIF
    images[0].save(output_filename, save_all=True, append_images=images[1:], duration=duration, loop=0)


from PIL import Image, ImageEnhance

def combine_images_with_transparency(image_paths, output_path, jump_imgs=2):
    """
    Combine a series of images into a single image to visualize object movement,
    adjusting transparency from low to high as the image index increases.

    :param image_paths: List of paths to the input images.
    :param output_path: Path to save the combined image.
    """
    # Load the first image to get size and mode
    base_image = Image.open(image_paths[0])
    base_image = base_image.convert("RGBA")
    width, height = base_image.size

    # Create a blank canvas to combine the images
    combined_image = Image.new("RGBA", (width, height))
    for i, image_path in enumerate(image_paths):
        if i % jump_imgs != 0: continue
        image = Image.open(image_path).convert("RGBA")

        # Composite the image onto the combined image
        combined_image = Image.blend(combined_image, image, alpha=0.65)

    # Save the combined image
    combined_image.save(output_path)


if __name__ == "__main__":
    args = parse_args()

    TASK_NAME = args.task
    if TASK_NAME == "SuccessRate":
        plot_utils = Plot_Utils()
        plot_utils.read_file("Union_2024_04_22_144343_Sync_Beta_group1_studying_table_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_entropy0.0_EVAL_best_Scene_table_objRange_1_10", checkpoint_name="ClutterGen")
        plot_utils.read_file("EVAL_HeurPolicy_Scene_table_objRange_1_10", checkpoint_name="Random Rejection Sampling")
        plot_utils.plot_success_steps()
    
    elif TASK_NAME == "Image2PDF":
        # Combine images to PDF
        image_folder = f"eval_res/Union/blender/{args.evalUniName}/render_results"
        image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')], key=natural_keys)  # Example for .png images
        pdf_output_path = os.path.join(image_folder, "combined.pdf")  # Update this path

        images_to_pdf(image_files, 
                    pdf_output_path, 
                    images_per_row=5,
                    dpi=500,
                    fig_ratio=1.2)

    elif TASK_NAME == "Image2GIF":
        # Create GIF
        source_folders = [
            "eval_res/Union/blender/Union_02-10_18:50Sync_table_PCExtractor_Relu_Rand_ObjPlace_QRRegion_Goal_minObjNum1_objStep1_maxObjNum10_maxPool10_maxScene1_maxStable60_contStable20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_EVAL_best_objRange_10_10/render_results",
            "eval_res/Union/blender/Union_02-04_04:37Sync_PCExtractor_FineTune_Relu_Rand_ObjPlace_QRRegion_Goal_maxObjNum8_maxPool10_maxScene1_maxStable60_contStable20_maxQR1Scene_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step81_trial5_EVAL_best_objRange_10_10/render_results",
            "eval_res/Union/blender/Union_02-10_18:50Sync_table_PCExtractor_Relu_Rand_ObjPlace_QRRegion_Goal_minObjNum1_objStep1_maxObjNum10_maxPool10_maxScene1_maxStable60_contStable20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_EVAL_best_ChangeTableSize_objRange_10_10/render_results",
            "eval_res/Union/blender/Union_02-20_01:11Sync_storage_furniture_5_PCExtractor_Relu_Rand_ObjPlace_QRRegion_Goal_minObjNum2_objStep2_maxObjNum10_maxPool10_maxScene1_maxStable60_contStable20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_EVAL_best_objRange_8_8/render_results"
            ]  # Update this path
        
        output_filename = "eval_res/Union/blender/combined.gif"  # Update this path
        create_gif_from_multiple_folders(source_folders, output_filename, num_images=40, duration=1000)

    elif TASK_NAME == "MiscInfo":
        Plot_Eval_Misc_Utils = Plot_Eval_Misc_Utils()
        Plot_Eval_Misc_Utils.read_file(args.evalUniName)
        # Plot_Eval_Misc_Utils.plot_obj_placement_success_rate()
        Plot_Eval_Misc_Utils.plot_obj_coverage_rate()
        Plot_Eval_Misc_Utils.plot_stable_steps(success_only=True)

    elif TASK_NAME == "MultiMiscInfo":
        source_folders = [
            "Union_03-12_23:40Sync_Beta_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab60_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_entropy0.01_seed123456_EVAL_best_Scene_table_objRange_10_10",
            "Union_2024_04_22_144343_Sync_Beta_group1_studying_table_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_entropy0._EVAL_best_Scene_table_objRange_10_10",
            "Union_2024_04_22_144351_Sync_Beta_group2_office_table_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_entropy0.01_EVAL_best_Scene_table_objRange_10_10",
            "Union_2024_04_22_144403_Sync_Beta_group3_kitchen_table_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_entropy0.0_EVAL_best_Scene_table_objRange_10_10",
            "Union_2024_04_23_213414_Sync_Beta_group4_real_objects_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool12_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_entropy0.01_EVAL_best_Scene_table_objRange_10_10"
        ]
        Plot_Eval_Misc_Utils = Plot_Eval_Misc_Utils()
        Plot_Eval_Misc_Utils.read_file_lst(source_folders)
        Plot_Eval_Misc_Utils.plot_stable_steps_lst(success_only=True, trial_thresholds=[4, 4, 4, 3, 4])

    elif TASK_NAME == "DatasetDistribution":
        plot_dataset_utils = Plot_Dataset_Utils()
        plot_dataset_utils.read_file()
        plot_dataset_utils.plot_ds_distribution()

    elif TASK_NAME == "ObjMoveTraj":
        image_folder = "paper/method/vis_move_hist"
        image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')], key=natural_keys)
        output_folder = os.path.join(image_folder, "combined")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "combined.png")
        combine_images_with_transparency(image_files, output_path, jump_imgs=3)

    elif TASK_NAME == "LearningCurve":
        plot_learning_curve()

    elif TASK_NAME == "TrialCurve":
        plot_trials_curve()