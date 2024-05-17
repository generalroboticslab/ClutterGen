from utils import se3_transform_pc
import pybullet_utils_cust as pu
import torch


# Update the evaluation success records
def update_success_records(success_records, qr_obj_name, sp_placed_obj_poses, gt_success):
    success_records["num_objs_on_qr_scene_counts"][len(sp_placed_obj_poses)] = success_records["num_objs_on_qr_scene_counts"].get(len(sp_placed_obj_poses), 0) + 1
    success_records["num_times_obj_get_qr_counts"][qr_obj_name] = success_records["num_times_obj_get_qr_counts"].get(qr_obj_name, 0) + 1
    success_records["num_objs_on_qr_scene_success"][len(sp_placed_obj_poses)] = success_records["num_objs_on_qr_scene_success"].get(len(sp_placed_obj_poses), 0) + gt_success
    success_records["num_times_obj_get_qr_success"][qr_obj_name] = success_records["num_times_obj_get_qr_success"].get(qr_obj_name, 0) + gt_success
                

def compute_success_records_summary(success_records):
    success_records["num_objs_on_qr_scene_success_rate"] = {k: v/success_records["num_objs_on_qr_scene_counts"][k] for k, v in success_records["num_objs_on_qr_scene_success"].items()}
    success_records["num_times_obj_get_qr_success_rate"] = {k: v/success_records["num_times_obj_get_qr_counts"][k] for k, v in success_records["num_times_obj_get_qr_success"].items()}
    success_records["Avg_success_rate"] = sum(success_records["num_objs_on_qr_scene_success"].values())/(sum(success_records["num_objs_on_qr_scene_counts"].values())+1e-8)
    success_records["recorded_num_data_points"] = sum(success_records["num_objs_on_qr_scene_counts"].values())


def visualize_pred_pose(scene_pc, qr_obj_pc, pred_qr_obj_pose, qr_obj_pose=None):
    scene_pc_np = scene_pc.cpu().numpy() if isinstance(scene_pc, torch.Tensor) else scene_pc
    qr_obj_pc_np = qr_obj_pc.cpu().numpy() if isinstance(qr_obj_pc, torch.Tensor) else qr_obj_pc
    pred_qr_obj_pose_np = pred_qr_obj_pose.cpu().numpy() if isinstance(pred_qr_obj_pose, torch.Tensor) else pred_qr_obj_pose
    qr_obj_pose_np = qr_obj_pose.cpu().numpy() if qr_obj_pose is not None and isinstance(qr_obj_pose, torch.Tensor) else qr_obj_pose
    # Visualize the point cloud
    for i in range(len(scene_pc_np)):
        transformed_pred_qr_obj_pc = se3_transform_pc(*pu.split_7d(pred_qr_obj_pose_np[i]), qr_obj_pc_np[i])
        if qr_obj_pose is not None:
            transformed_ground_truth_qr_obj_pc = se3_transform_pc(*pu.split_7d(qr_obj_pose[i]), qr_obj_pc_np[i])
            pu.visualize_pc(
                [scene_pc_np[i], transformed_pred_qr_obj_pc, transformed_ground_truth_qr_obj_pc], 
                    color=[[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        else:
            # Pink color for the predicted object
            pu.visualize_pc([scene_pc_np[i], transformed_pred_qr_obj_pc], color=[None, [1, 0, 0]])