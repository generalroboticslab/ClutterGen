from matplotlib import pyplot as plt
import torch
import numpy as np
import seaborn as sns
import os
import pandas as pd
import ast
sns.set_theme()


class Plot_Utils:
    def __init__(self):
        self.data_full = {}
        
        # Pre-defined data
        self.trained_objs = [1, 5, 10, 12, 16]


    def read_file(self, file_path, checkpoint_name="10p", trained_objs=None):
        trained_objs = self.trained_objs.copy() if trained_objs is None else trained_objs
        assert os.path.exists(file_path), "File not found: {}".format(file_path)
        _, suffix = os.path.splitext(file_path)
        if suffix == ".pth":
            self.data_full[checkpoint_name] = (torch.load(file_path), trained_objs)
        elif suffix == ".csv":
            columns_to_read = list(range(5)) # First 5 columns to read
            df = pd.read_csv(file_path, usecols=columns_to_read)
            # Convert string to python objects
            self.data_full[checkpoint_name] = (df.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x), trained_objs)
        else: raise NotImplementedError("Unsupported file type: {}".format(suffix))


    def plot_success_steps(self):
        # Create the figure and axes
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))  # Create a 2-row, 1-column subplot grid

        # Plot Reset Success Rate
        axes[0].set_title("Reset Success Rate Summary", fontsize=16)
        axes[0].set_xlabel("Number of Objects", fontsize=14)
        axes[0].set_ylabel("Reset Success Rate", fontsize=14)

        trained_obj_label = 'Trained'; trained_obj_label_act = trained_obj_label
        for checkpoint_name in self.data_full.keys():
            data, trained_objs = self.data_full[checkpoint_name]
            num_obj = data["num_placing_objs"]
            success_rate = data["success_rate"]
            axes[0].plot(num_obj, success_rate, '-o', label=checkpoint_name, linewidth=2, markersize=8)
            if 'best' in checkpoint_name:
                axes[0].scatter(trained_objs, success_rate[num_obj.isin(trained_objs)], 
                                marker='*', s=250, c='g', zorder=2, label=trained_obj_label_act)
                trained_obj_label_act = None # Only show the label once

        # Change the order of legend
        handles, labels = axes[0].get_legend_handles_labels()
        if trained_obj_label in labels:
            index_to_move = labels.index(trained_obj_label)
            handles = [handle for i, handle in enumerate(handles) if i != index_to_move] + [handles[index_to_move]]
            labels = [label for i, label in enumerate(labels) if i != index_to_move] + [labels[index_to_move]]
        axes[0].legend(handles, labels, fontsize=12)
        axes[0].tick_params(axis='both', labelsize=12)

        # Plot Episode Steps per Number of Placing Objects
        axes[1].set_title("Episode Steps per Number of Objects", fontsize=16)
        axes[1].set_xlabel("Number of Objects", fontsize=14)
        axes[1].set_ylabel("Episode Steps", fontsize=14)

        trained_obj_label_act = trained_obj_label
        for checkpoint_name in self.data_full.keys():
            data, trained_objs = self.data_full[checkpoint_name]
            num_obj = data["num_placing_objs"]
            unstable_steps = data["unstable_steps"]
            axes[1].plot(num_obj, unstable_steps, '-o', label=checkpoint_name, linewidth=2, markersize=8)

            if 'best' in checkpoint_name:
                axes[1].scatter(trained_objs, unstable_steps[num_obj.isin(trained_objs)], 
                                marker='*', s=250, c='g', zorder=2, label=trained_obj_label_act)
                trained_obj_label_act = None # Only show the label once

        # Change the order of legend
        handles, labels = axes[0].get_legend_handles_labels()
        if trained_obj_label in labels:
            index_to_move = labels.index(trained_obj_label)
            handles = [handle for i, handle in enumerate(handles) if i != index_to_move] + [handles[index_to_move]]
            labels = [label for i, label in enumerate(labels) if i != index_to_move] + [labels[index_to_move]]
        axes[1].legend(handles, labels, fontsize=12)
        axes[1].tick_params(axis='both', labelsize=12)

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save the plot as a pdf
        plt.savefig("results/post_corrector/summary_plots.pdf")
        # Show the plot
        plt.show()

    
    def clear_data(self):
        self.data_full = {}



if __name__ == "__main__":
    plot_utils = Plot_Utils()
    plot_utils.read_file("eval_res/YCB/CSV/YCB_11-22_02:32_FC_FT_Rand_placing_Goal_10_maxstable50_Weight_rewardPobj100.0_EVALbest_Setup.csv", checkpoint_name="10p_best", trained_objs=[1, 5, 10])
    plot_utils.read_file("eval_res/YCB/CSV/YCB_11-28_01:10_FC_FT_Rand_placing_Goal_12_maxstable50_Weight_rewardPobj100.0_EVALbest_Setup.csv", checkpoint_name="12p_best", trained_objs=[1, 5, 10, 12])
    plot_utils.read_file("eval_res/YCB/CSV/YCB_11-30_21:39_FC_FT_Rand_placing_Goal_16_maxstable50_Weight_rewardPobj100.0_EVALbest_Setup.csv", checkpoint_name="16p_best", trained_objs=[1, 5, 10, 12, 16])
    plot_utils.read_file("eval_res/YCB/CSV/EVAL_RandomPolicy_Setup.csv", checkpoint_name="RejectionSampling")
    plot_utils.plot_success_steps()