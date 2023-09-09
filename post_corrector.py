from isaacgym_utils import *
from isaacgym.torch_utils import tf_apply, quat_from_euler_xyz


class PostCorrector:
    def __init__(self, env, device="cuda") -> None:
        self.env = env
        self.device = device

        self.force_threshold = 20
        self.vel_threshold = [0.1, torch.pi/5] # Translation, rotation speed
        # The goal is to input cur_scene_dict which contains objects' mesh description, position, orientation, scaling
        # Check whether they are realiable and feasible
        # Use this to train randomizer to generate more realiable and feasible scene
        # keep increasing the ability to distinguish realiablity and feasibility


    def handed_check_realiablity(self, env_info, cur_scene_dict):
        env_ids, body_ids, body_names, positions, orientations, linvels, angvels = env_info
        # Check object force
        object_force = get_force(self.env.force_tensor, body_ids)
        obj_force_fail_index = (object_force > self.force_threshold).any(dim=1)
        force_fail_force = object_force[obj_force_fail_index]
        force_fail_env_ids = env_ids[obj_force_fail_index]
        force_fail_obj_ids = body_ids[obj_force_fail_index]
        force_fail_obj_names = body_names[obj_force_fail_index]
        # Check object velocity (To see if everything is stable)
        object_pose, object_vel = get_pose(self.env.rb_states, body_ids)
        obj_vel_fail_index = ((object_vel[:, :3]>self.vel_threshold[0]).any(dim=1) + (object_vel[:, 3:]>self.vel_threshold[1]).any(dim=1))
        vel_fail_vel = object_vel[obj_vel_fail_index]
        vel_fail_env_ids = env_ids[obj_vel_fail_index]
        vel_fail_obj_ids = body_ids[obj_vel_fail_index]
        vel_fail_obj_names = body_names[obj_vel_fail_index]

        for i in range(len(force_fail_env_ids)):
            print(f"FORCE FAIL: {force_fail_env_ids[i]} | Name: {force_fail_obj_names[i]}, ID: {force_fail_obj_ids[i]}, Force: {force_fail_force[i]}")

        for i in range(len(force_fail_env_ids)):
            print(f"FORCE FAIL: {vel_fail_env_ids[i]} | Name: {vel_fail_obj_names[i]}, ID: {vel_fail_obj_ids[i]}, Vel: {vel_fail_vel[i]}")

        return torch.cat([force_fail_env_ids, vel_fail_env_ids]).unique()


    def handed_check_feasibility(self):
        pass
        # Check object can be reached by the robot arm

        # Check objects should be reasonable based on GPT description


                
    def to_torch(self, x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=self.device, requires_grad=False)


