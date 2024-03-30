


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