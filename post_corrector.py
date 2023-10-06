from isaacgym_utils import *
from isaacgym.torch_utils import tf_apply, quat_from_euler_xyz

from tabulate import tabulate


class PostCorrector:
    def __init__(self, handem_env, device="cuda") -> None:
        self.handem_env = handem_env
        self.device = device

        self.force_threshold = 20
        self.vel_threshold = [0.02, torch.pi/5] # Translation, rotation speed
        # The goal is to input cur_scene_dict which contains objects' mesh description, position, orientation, scaling
        # Check whether they are realiable and feasible
        # Use this to train randomizer to generate more realiable and feasible scene
        # keep increasing the ability to distinguish realiablity and feasibility


    def handed_check_realiablity(self, env_info, cur_scene_dict, force_check=True, vel_check=True, gen_readable_table=False, verbose=False):
        env_ids, body_ids, body_names, positions, orientations, linvels, angvels = env_info
        # Check object force
        object_force = get_force(self.handem_env.force_tensor, body_ids)
        obj_force_fail_index = (object_force.abs() > self.force_threshold).any(dim=1)
        force_fail_force = object_force[obj_force_fail_index]
        force_fail_env_ids = env_ids[obj_force_fail_index]
        force_fail_obj_ids = body_ids[obj_force_fail_index]
        force_fail_obj_names = body_names[obj_force_fail_index.nonzero().squeeze().cpu()]
        # Check object velocity (To see if everything is stable)
        object_pose, object_vel = get_pose(self.handem_env.rb_states, body_ids)
        obj_vel_fail_index = ((object_vel[:, :3].abs()>self.vel_threshold[0]).any(dim=1) + (object_vel[:, 3:].abs()>self.vel_threshold[1]).any(dim=1))
        vel_fail_vel = object_vel[obj_vel_fail_index]
        vel_fail_env_ids = env_ids[obj_vel_fail_index]
        vel_fail_obj_ids = body_ids[obj_vel_fail_index]
        vel_fail_obj_names = body_names[obj_vel_fail_index.nonzero().squeeze().cpu()]

        # Define your data as a list of lists
        check_table = None
        if gen_readable_table:
            data = []

            if force_check:
                for i in range(len(force_fail_env_ids)):
                    data.append(["FORCE FAIL", force_fail_env_ids[i], force_fail_obj_names[i], force_fail_obj_ids[i], force_fail_force[i]])
            if vel_check:
                for k in range(len(vel_fail_env_ids)):
                    data.append(["VEL FAIL", vel_fail_env_ids[k], vel_fail_obj_names[k], vel_fail_obj_ids[k], vel_fail_vel[k]])

            # Define headers for the table
            headers = ["Type", "Env ID", "Name", "ID", "Value"]

            # Generate the table and print it
            check_table = tabulate(data, headers, tablefmt="pretty")

            if verbose: print(check_table)

        if force_check and not vel_check: return force_fail_env_ids.unique(), check_table
        elif vel_check and not force_check: return vel_fail_env_ids.unique(), check_table
        else: return torch.cat([force_fail_env_ids, vel_fail_env_ids]).unique(), check_table


    def handed_check_feasibility(self):
        pass
        # Check object can be reached by the robot arm

        # Check objects should be reasonable based on GPT description

                
    def to_torch(self, x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=self.device, requires_grad=False)


