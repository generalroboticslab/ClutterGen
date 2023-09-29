from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import mpl_toolkits.axisartist.floating_axes as floating_axes
import torch
import os

from isaacgym.torch_utils import tf_apply, quat_apply, normalize
import matplotlib.transforms as transforms

Finger_dim = [0.0375, 0.0375, 0.06875]
Table_dim = [0.6, 1.2, 0.6]
Table_pos = [-0.9, -0.25, Table_dim[2] / 2]
Direction_scale = 20
Shift_value = -0.2

def callibration_vis(sim_finger_pos_box, rob_finger_pos_box, 
                     sim_finger_contact_box, rob_finger_contact_box=None, 
                     sim_rob_finger_pose_diff_box=None, save_path=None):
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.delaxes(axes[0, 1]); axes[0, 1] = fig.add_subplot(222, projection="3d")

    axes[0, 0].set_title("Top-down View Trajectory; Direction <---")
    axes[0, 1].set_title("3D Contact Info")
    axes[1, 0].set_title("Position Difference")
    axes[1, 1].set_title("Contact force value")

    axes[0, 1].set_xlabel('x_axis'); axes[0, 1].set_ylabel('y_axis'); axes[0, 1].set_zlabel('z_axis')
    axes[0, 1].view_init(azim=-180, elev=87)

    sim_finger_pos_world, sim_finger_ori_world = sim_finger_pos_box[:, :3], sim_finger_pos_box[:, 3:]
    sim_contact_pos_world, sim_contact_force_world = sim_finger_contact_box[:, :3], sim_finger_contact_box[:, 3:6]
    if rob_finger_contact_box is not None:
        rob_finger_pos_world, rob_finger_ori_world = rob_finger_pos_box[:, :3], rob_finger_pos_box[:, 3:]
        rob_contact_pos_world, rob_contact_force_world = rob_finger_contact_box[:, :3], rob_finger_contact_box[:, 3:6]
        rob_finger_pos_world[:, 1] += Shift_value; rob_contact_pos_world[:, 1] += Shift_value

    # Draw Position Difference
    if sim_rob_finger_pose_diff_box is not None:
        axes[1, 0].plot(sim_rob_finger_pose_diff_box, '-o', color='b')
    # Draw 3D conctact Position
    labels = ['Sim', 'Real']
    for i in range(len(sim_contact_pos_world)):
        if not torch.allclose(sim_contact_pos_world[i], sim_finger_pos_world[i]):
            axes[0, 1].scatter(sim_contact_pos_world[i, 0], sim_contact_pos_world[i, 1], sim_contact_pos_world[i, 2], c='r', label=labels[0])
            labels[0] = None
        if rob_finger_contact_box is not None and (not torch.allclose(rob_contact_pos_world[i], rob_finger_pos_world[i])):
            axes[0, 1].scatter(rob_contact_pos_world[i, 0], rob_contact_pos_world[i, 1], rob_contact_pos_world[i, 2], c='g', label=labels[1])
            labels[1] = None
    # Draw Contact Force
    axes[1, 1].plot(torch.norm(sim_contact_force_world, dim=1), '-o', color='r', label='Sim')
    if rob_finger_contact_box is not None:
        axes[1, 1].plot(torch.norm(rob_contact_force_world, dim=1), '-*', color='g', label='Real')
    # Draw Top Down View Trajectory
    labels = ['Sim', 'Real']
    for i in range(0, len(sim_finger_pos_world), 1):
        sim_contact_pos_w, sim_contact_norm_w = sim_contact_pos_world[i], normalize(sim_contact_force_world[i])
        sim_norm_end_point = (sim_contact_pos_w+sim_contact_norm_w/Direction_scale)[:2]
        axes[0, 0].add_patch(Circle(sim_finger_pos_world[i, :2], radius=Finger_dim[0]/2, fill=False, color='g'))
        axes[0, 0].scatter(sim_contact_pos_w[0], sim_contact_pos_w[1], s=12, c='r')
        axes[0, 0].plot((sim_contact_pos_w[0], sim_norm_end_point[0]), (sim_contact_pos_w[1], sim_norm_end_point[1]), color='r', label=labels[0])
        labels[0] = None

        if rob_finger_contact_box is not None:
            rob_contact_pos_w, rob_contact_norm_w = rob_contact_pos_world[i], normalize(rob_contact_force_world[i])
            rob_norm_end_point = (rob_contact_pos_w+rob_contact_norm_w/Direction_scale)[:2]
            axes[0, 0].add_patch(Circle((rob_finger_pos_world[i, 0], rob_finger_pos_world[i, 1]), radius=Finger_dim[0]/2, fill=False, color='r'))
            axes[0, 0].scatter(rob_contact_pos_w[0], rob_contact_pos_w[1],  s=12, c='g')
            axes[0, 0].plot((rob_contact_pos_w[0], rob_norm_end_point[0]), (rob_contact_pos_w[1], rob_norm_end_point[1]), color='g', label=labels[1])
            labels[1] = None
    
    axes[0, 0].relim(); axes[0, 0].autoscale_view()
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[1, 1].legend()

    if save_path is not None: plt.savefig(os.path.join(save_path, "picture.pdf"), format='pdf')
    plt.show()


