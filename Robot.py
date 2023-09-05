from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

from isaacgym_utils import *

class Franka:
    def __init__(self, sim, asset_root, robot_name="franka", device='cuda') -> None:
        self.device = device
        self.sim = sim
        self.asset_root = asset_root

        franka_asset_file = "robots/franka_description/robots/franka_panda.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True # ??? What is this for?
        self.franka_asset = gym.load_asset(self.sim, self.asset_root, franka_asset_file, asset_options)
        self.franka_pose = gymapi.Transform()
        self.franka_pose.p = gymapi.Vec3(0, 0, 0)
        # configure franka dofs
        self.franka_dof_props = gym.get_asset_dof_properties(self.franka_asset)
        self.franka_lower_limits = self.franka_dof_props["lower"]
        self.franka_upper_limits = self.franka_dof_props["upper"]
        franka_ranges = self.franka_upper_limits - self.franka_lower_limits
        franka_mids = 0.3 * (self.franka_upper_limits + self.franka_lower_limits)
        # use position drive for all dofs
        self.franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
        self.franka_dof_props["stiffness"][:7].fill(400.0)
        self.franka_dof_props["damping"][:7].fill(40.0)
        # grippers
        self.franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        self.franka_dof_props["stiffness"][7:].fill(800.0)
        self.franka_dof_props["damping"][7:].fill(40.0)

        # default dof states and position targets
        self.franka_num_dofs = gym.get_asset_dof_count(self.franka_asset)
        
        default_dof_pos = np.zeros(self.franka_num_dofs, dtype=np.float32)
        default_dof_pos[:7] = franka_mids[:7]
        # grippers open
        default_dof_pos[7:] = self.franka_upper_limits[7:]
        self.default_dof_state = np.zeros(self.franka_num_dofs, gymapi.DofState.dtype)
        self.default_dof_state["pos"] = default_dof_pos

        # send to torch
        self.default_dof_pos_tensor = self.to_torch(default_dof_pos)

        # get link index of panda hand, which we will use as end effector
        self.franka_link_dict = gym.get_asset_rigid_body_dict(self.franka_asset)
        self.franka_hand_index = self.franka_link_dict["panda_hand"]


    def to_torch(self, x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=self.device, requires_grad=False)

