from isaacgym_utils import *
from isaacgym.torch_utils import tf_apply, quat_from_euler_xyz


class DomainRandomizer:
    def __init__(self, device="cuda") -> None:
        self.device = device


    def fill_obj_poses(self, scene_dicts):
        for scene_dict in scene_dicts:
            prefixed_objs = scene_dict["prefixed"]
            movable_objs = scene_dict["movable"]

            table_bbox, table_miscs = prefixed_objs["table"]
            table_position, table_orientation, table_scaling = table_miscs
            # Table height needs to be adjust later, RandomAreaScale needs to be adjust
            table_height, RandomAreaScale = table_position[2] + table_bbox[2]/2, 0.8
            obj_pos_min = tf_apply(table_orientation, table_position, -self.to_torch(table_bbox)/2 * RandomAreaScale)
            obj_pos_max = tf_apply(table_orientation, table_position, self.to_torch(table_bbox)/2 * RandomAreaScale)
            obj_ori_min, obj_ori_max = self.to_torch([0., 0., -torch.pi]), self.to_torch([0., 0., torch.pi])

            # Random 1; Using contact to sample object position one by one
            """
            for obj in movable_objs.keys():
                obj_bbox, obj_pose = movable_objs[obj]
                obj_pos = torch_rand_float(obj_pos_min, obj_pos_max, (3, ))
                obj_ori = quat_from_euler(torch_rand_float(obj_ori_min, obj_ori_max, (3, )))
                self.env.set_obj(obj_id, obj_pos, obj_ori)
                contacts = gym.get_env_rigid_contacts(env)
                # Check contacts if obj contact with other object or not, if contact needs to go back
            """
            
            # Random 2; Purely random without any check; we now only randomize x, y which might be easier. If we go to more complicated env, it will be impossible!
            num_movable_objs = len(movable_objs)
            obj_pos = torch_rand_float(obj_pos_min, obj_pos_max, (num_movable_objs, 3))
            obj_ori = quat_from_euler(torch_rand_float(obj_ori_min, obj_ori_max, (num_movable_objs, 3)))
            scaling = 1
            for i, name in enumerate(movable_objs):
                # We need to fix the obj height on the table!
                obj_bbox = scene_dict["movable"][name][0]
                obj_pos[i, 2] = table_height + obj_bbox[2]/2 + 1e-3
                scene_dict["movable"][name][1] = [obj_pos[i], obj_ori[i], scaling]

        # print(scene_dicts)
        return scene_dicts

                
    def to_torch(self, x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=self.device, requires_grad=False)


