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
        self.data = {}
        
        # Pre-defined data
        self.trained_objs = [1, 5, 10, 12]


    def read_file(self, file_path, checkpoint_name="10p"):
        assert os.path.exists(file_path), "File not found: {}".format(file_path)
        _, suffix = os.path.splitext(file_path)
        if suffix == ".pth":
            self.data[checkpoint_name] = torch.load(file_path)
        elif suffix == ".csv":
            columns_to_read = list(range(5)) # First 5 columns to read
            df = pd.read_csv(file_path, usecols=columns_to_read)
            # Convert string to python objects
            self.data[checkpoint_name] = df.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        else: raise NotImplementedError("Unsupported file type: {}".format(suffix))


    def plot_success_steps(self):
        # Create the figure and axes
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))  # Create a 2-row, 1-column subplot grid

        # Plot Reset Success Rate
        axes[0].set_title("Reset Success Rate Summary", fontsize=16)
        axes[0].set_xlabel("Number of Objects", fontsize=14)
        axes[0].set_ylabel("Reset Success Rate", fontsize=14)

        for checkpoint_name in self.data.keys():
            num_obj = self.data[checkpoint_name]["num_placing_objs"]
            success_rate = self.data[checkpoint_name]["success_rate"]
            axes[0].plot(num_obj, success_rate, '-o', label=checkpoint_name, linewidth=2, markersize=8)
            if 'p' in checkpoint_name:
                axes[0].scatter(self.trained_objs, success_rate[num_obj.isin(self.trained_objs)], 
                                marker='*', s=250, c='g', zorder=2, label='Trained Objects')

        axes[0].legend(fontsize=12)
        axes[0].tick_params(axis='both', labelsize=12)

        # Plot Episode Steps per Number of Placing Objects
        axes[1].set_title("Episode Steps per Number of Objects", fontsize=16)
        axes[1].set_xlabel("Number of Objects", fontsize=14)
        axes[1].set_ylabel("Episode Steps", fontsize=14)

        for checkpoint_name in self.data.keys():
            num_obj = self.data[checkpoint_name]["num_placing_objs"]
            unstable_steps = self.data[checkpoint_name]["unstable_steps"]
            axes[1].plot(num_obj, unstable_steps, '-o', label=checkpoint_name, linewidth=2, markersize=8)

            if 'p' in checkpoint_name:
                axes[1].scatter(self.trained_objs, unstable_steps[num_obj.isin(self.trained_objs)], 
                                marker='*', s=250, c='g', zorder=2, label='Trained Objects')

        axes[1].legend(fontsize=12)
        axes[1].tick_params(axis='both', labelsize=12)

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save the plot as a pdf
        plt.savefig("results/post_corrector/summary_plots.pdf")
        # Show the plot
        plt.show()

    
    def clear_data(self):
        self.data = {}



if __name__ == "__main__":
    plot_utils = Plot_Utils()
    plot_utils.read_file("eval_res/YCB/CSV/YCB_11-22_02:32_FC_FT_Rand_placing_Goal_10_maxstable50_Weight_rewardPobj100.0_EVALbest_Setup.csv", checkpoint_name="10p_best")
    plot_utils.read_file("eval_res/YCB/CSV/EVAL_RandomPolicy_Setup.csv", checkpoint_name="Random")
    plot_utils.plot_success_steps()