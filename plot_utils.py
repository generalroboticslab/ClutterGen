from matplotlib import pyplot as plt
import torch
import numpy as np
import seaborn as sns
sns.set_theme()


dict_path = "results/post_corrector/failure_rate_record.pth"
failure_dicts_raw = torch.load(dict_path)

failure_rate_summary = {}
for num_obj, failure_rate_dict in failure_dicts_raw.items():
    for checker_name, failure_rate in failure_rate_dict.items():
        if checker_name not in failure_rate_summary:
            failure_rate_summary[checker_name] = []
        failure_rate_summary[checker_name].append((num_obj, failure_rate))


# Create the figure and axes
fig, axes = plt.subplots(1, 1, figsize=(8, 6))
# Set the title with a larger font size
axes.set_title("Reset Success Rate Summary", fontsize=16)
# Set the labels with larger font sizes
axes.set_xlabel("Number of Objects", fontsize=14)
axes.set_ylabel("Reset Success Rate", fontsize=14)

# Loop through your data and adjust the marker size and line width
for checker_name, failure_rate_corr in failure_rate_summary.items():
    failure_rate_corr = np.array(failure_rate_corr)
    
    # Adjust the marker size and line width
    axes.plot(
        failure_rate_corr[:, 0], 
        1 - failure_rate_corr[:, 1], 
        '-o', 
        label=checker_name, 
        linewidth=2,  # Adjust line width
        markersize=8,  # Adjust marker size
    )

# Set the legend with a larger font size
axes.legend(fontsize=12)
# Set the tick label font size for both x and y axes
axes.tick_params(axis='both', labelsize=12)
# Save the plot as a pdf
plt.savefig("results/post_corrector/failure_rate_summary.pdf")
# Show the plot
plt.show()