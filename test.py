import pybullet as p
import time
import pybullet_data
from utils import read_json
import time
import numpy as np

import itertools


# physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
# p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
# p.setGravity(0,0,-9.8)
# planeId = p.loadURDF("plane.urdf")
# startPos = [0,0,1]
# startOrientation = p.getQuaternionFromEuler([0,0,0])
# #set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)

# asset_root = "assets"
# object_folder = "objects"

# num_objects = 50
# obj_categories = read_json(f'{asset_root}/{object_folder}/obj_categories.json')
# object_name = np.random.choice(list(obj_categories.keys()), num_objects, replace=False)
# object_name = list(obj_categories.keys())
# for obj_name in object_name:
#     try:
#         object_id = p.loadURDF(f"assets/objects/{obj_name}/0/mobility.urdf", basePosition=np.random.rand(3), globalScaling=0.1)  # Load an object at position [0, 0, 1]
#     except:
#         print(f"Failed to load object {obj_name}")

# print("Start Simulation")
# for i in range (10000):
#     p.stepSimulation()
#     time.sleep(1./240.)
# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)
# p.disconnect()




# print("Simulation Done")

tippe_top = """
<mujoco model="tippe top">
  <option integrator="RK4"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>

  <worldbody>
    <geom size=".2 .2 .01" type="plane" material="grid"/>
    <light pos="0 0 .6"/>
    <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
    <body name="top" pos="0 0 .02">
      <freejoint/>
      <geom name="ball" type="sphere" size=".02" />
      <geom name="stem" type="cylinder" pos="0 0 .02" size="0.004 .008"/>
      <geom name="ballast" type="box" size=".023 .023 0.005"  pos="0 0 -.015"
       contype="0" conaffinity="0" group="3"/>
    </body>
  </worldbody>

  <keyframe>
    <key name="spinning" qpos="0 0 0.02 1 0 0 0" qvel="0 0 0 0 1 200" />
  </keyframe>
</mujoco>
"""

# import mujoco
# from mujoco import viewer

# XML=r"""
# <mujoco>
#   <asset>
#     <mesh file="gizmo.stl"/>
#   </asset>
#   <worldbody>
#     <body>
#       <freejoint/>
#       <geom type="mesh" name="gizmo" mesh="gizmo"/>
#     </body>
#   </worldbody>
# </mujoco>
# """

# ASSETS=dict()
# with open('/path/to/gizmo.stl', 'rb') as f:
#   ASSETS['gizmo.stl'] = f.read()

# model = mujoco.MjModel.from_xml_string(tippe_top)
# model.opt.gravity = (0, 0, -9.8)
# data = mujoco.MjData(model)
# mujoco.mj_resetData(model, data)
# mj_viewer = mujoco.viewer.launch_passive(model, data)
# renderer = mujoco.Renderer(model)
# while mj_viewer.is_running():
#     ...
#     # Step the physics.
#     mujoco.mj_step(model, data)
#     renderer.update_scene(data, camera="closeup")
#     # Add a 3x3x3 grid of variously colored spheres to the middle of the scene.
#     mj_viewer.user_scn.ngeom = 0
#     i = 0
#     for x, y, z in itertools.product(*((range(-1, 2),) * 3)):
#         mujoco.mjv_initGeom(
#             mj_viewer.user_scn.geoms[i],
#             type=mujoco.mjtGeom.mjGEOM_SPHERE,
#             size=[0.02, 0, 0],
#             pos=0.1*np.array([x, y, z]),
#             mat=np.eye(3).flatten(),
#             rgba=0.5*np.array([x + 1, y + 1, z + 1, 2])
#         )
#         i += 1
    
#     mj_viewer.user_scn.ngeom = i
#     mj_viewer.sync()


# from dm_control import mjcf

# # aa = mjcf.RootElement()
# # robot = mjcf.from_path(f"assets/objects/{object_name[0]}/0/mobility.urdf")
# # aa.attach(robot)

# model = mujoco.MjModel.from_xml_path(f"assets/scenes/test_scene/humanoid.xml")

# data = mujoco.MjData(model)
# renderer = mujoco.Renderer(model)

# # enable joint visualization option:
# scene_option = mujoco.MjvOption()
# scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

# duration = 3.8  # (seconds)
# framerate = 60  # (Hz)

# frames = []
# mujoco.mj_resetData(model, data)
# mj_viewer = mujoco.viewer.launch_passive(model, data)

# while mj_viewer.is_running():
#     ...
#     # Step the physics.
#     mujoco.mj_step(model, data)

#     # Add a 3x3x3 grid of variously colored spheres to the middle of the scene.
#     mj_viewer.sync()
#     time.sleep(1./240.)


# import torch
# from pytorch3d.io import load_obj
# from pytorch3d.ops import sample_points_from_mesh
# import matplotlib.pyplot as plt

# # Load the mesh from an OBJ file
# mesh_path = 'path/to/your/mesh.obj'
# mesh = load_obj(mesh_path)

# # Set the number of points to sample
# num_points = 10000

# # Sample points from the mesh
# points = sample_points_from_mesh(mesh, num_points)

# # Convert the points to a NumPy array for visualization
# points_np = points.detach().cpu().numpy()

# # Visualize the generated point cloud (optional)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c='b', marker='.')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()


# URDF to PointsCloud
# import pybullet_utils as pu
# import open3d as o3d
# client_id = p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.loadURDF("plane.urdf")
# p.setGravity(0, 0, -9.8)
# obj_id = p.loadURDF("assets/union_objects_test/Eyeglasses/0/mobility.urdf", basePosition=[0, 0, 10.], useFixedBase=True, globalScaling=0.2)

# world2baselink = pu.get_link_pose(obj_id, -1)
# num_links = pu.get_num_links(obj_id)
# whole_pc = []

# for link_id in range(-1, num_links):
#     link_pc_world = pu.get_link_pc_from_id(obj_id, link_id, min_num_points=1024, use_worldpos=False, client_id=client_id)
#     whole_pc.extend(link_pc_world)

# whole_pc = np.array(whole_pc)
# print(whole_pc.shape)
# whole_pc = pu.get_obj_pc_from_id(obj_id, num_points=50000, use_worldpos=True, client_id=client_id)
# print(whole_pc.shape)
# o3d_pc = o3d.geometry.PointCloud()
# o3d_pc.points = o3d.utility.Vector3dVector(whole_pc)
# bbox = o3d_pc.get_axis_aligned_bounding_box()
# bbox.color = np.array([0, 0, 0.])
# print(bbox.get_center(), bbox.get_half_extent())
# coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
# # coord_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=world2link[0])
# o3d.visualization.draw_geometries([o3d_pc, coord_frame, bbox])



# import pybullet as p
# import pybullet_data
# import pybullet_utils as pu
# import os


# def load_urdf_and_get_bbox(urdf_file, label_file):
#     """
#     Load a URDF file and return its bounding box dimensions.
#     """
#     globalScaling = 1.
#     if os.path.exists(label_file):
#       obj_label = read_json(label_file)
#       globalScaling = obj_label["globalScaling"]
#     obj_id = p.loadURDF(urdf_file, useFixedBase=False, globalScaling=globalScaling)
#     obj_pc = pu.get_obj_pc_from_id(obj_id)
#     obj_bbox = pu.get_obj_axes_aligned_bbox_from_pc(obj_pc)

#     obj_joints_num = pu.get_num_joints(obj_id)
#     if obj_joints_num > 0: # If the object is not movable, we need to control its joints to make it movable below each reload urdf!
#         joints_limits = np.array([pu.get_joint_limits(obj_id, joint_i) for joint_i in range(obj_joints_num)])
#         pu.set_joint_positions(obj_id, list(range(obj_joints_num)), joints_limits[:, 0])
#         pu.control_joints(obj_id, list(range(obj_joints_num)), joints_limits[:, 0])
#     return obj_id, obj_bbox


# def main(urdf_files):
#     # Start PyBullet
#     p.connect(p.GUI)
#     p.setAdditionalSearchPath(pybullet_data.getDataPath())
#     p.setGravity(0, 0, -10)
#     plane_id = p.loadURDF("plane.urdf")
#     pu.change_obj_color(plane_id, rgba_color=[1., 1., 1., 0.2])
    
#     # Load table URDF (adjust the path to your table URDF)
#     tableHalfExtents = [1.1, 1.5, 0.2]
#     tableId = pu.create_box_body(position=[0., 0., tableHalfExtents[2]], orientation=p.getQuaternionFromEuler([0., 0., 0.]),
#                                           halfExtents=tableHalfExtents, rgba_color=[1, 1, 1, 0.5], mass=0.)

#     # Initialize variables for object placement
#     current_x, current_y = -tableHalfExtents[0], -tableHalfExtents[1]
#     max_row_height = 0

#     for i, urdf_file in enumerate(urdf_files):
#         # Load URDF and get its bounding box size
#         label_file = urdf_file.replace("mobility.urdf", "label.json")
#         obj_id, bbox = load_urdf_and_get_bbox(urdf_file, label_file)
#         obj_size = [v*2 for v in bbox[-3:]]
        
#         # Check if the object fits in the current row, else move to next column
#         if current_x + obj_size[0] > tableHalfExtents[0]:
#             current_x = -tableHalfExtents[0]
#             current_y += max_row_height
#             max_row_height = 0

#         # Update the maximum height of the current row
#         max_row_height = max(max_row_height, obj_size[1])

#         # Check if the object fits in the current column, else skip or handle appropriately
#         if current_y + obj_size[1] > tableHalfExtents[1]:
#             print(f"Not enough space for object {urdf_file}")
#             continue

#         # Place the object
#         pu.set_pose(obj_id, ([current_x + obj_size[0]/2, current_y + obj_size[1]/2, obj_size[2]/2+tableHalfExtents[2]*2], [0, 0, 0, 1]))

#         # Update the x-coordinate for the next object
#         current_x += obj_size[0]

#     # Keep the simulation running
#     p.setRealTimeSimulation(1)
#     input("Press Enter to exit...")  # Wait for user input to exit
#     p.disconnect()


# urdf_files = []
# # List of URDF files to load
# obj_source_folder = "assets/union_objects"
# obj_categories = os.listdir(obj_source_folder)
# for obj_category in obj_categories:
#     obj_category_folder = os.path.join(obj_source_folder, obj_category)
#     obj_instances = os.listdir(obj_category_folder)
#     for obj_instance in obj_instances:
#         obj_instance_folder = os.path.join(obj_category_folder, obj_instance)
#         obj_urdf = os.path.join(obj_instance_folder, "mobility.urdf")
#         urdf_files.append(obj_urdf)

# urdf_files = np.random.choice(urdf_files, 50, replace=False)
# print(f"Number of objects to load: {len(urdf_files)}")
# # Run the main function
# main(urdf_files)

# from utils import read_json
# aa = read_json("assets/union_objects/Scissors/7/label.json")


# import psutil
# import os

# process = psutil.Process(477850)
# print(f"Memory Usage: {process.memory_info().rss / 1024 ** 2} MB")  # RSS memory in MB


# import pybullet_utils as pu
# import open3d as o3d

# client_id = p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.loadURDF("plane.urdf")
# p.setGravity(0, 0, -9.8)

# dataset_folder = "assets/union_objects_test"
# cate = "AlarmClock"
# idx = "1"
# obj_label_path = f"{dataset_folder}/{cate}/{idx}/label.json"
# obj_label = read_json(obj_label_path)
# obj_id = p.loadURDF(f"{dataset_folder}/{cate}/{idx}/mobility.urdf", basePosition=[0, 0, 0.], useFixedBase=False, globalScaling=obj_label["globalScaling"])


# world2baselink = pu.get_link_pose(obj_id, -1)
# num_links = pu.get_num_links(obj_id)
# whole_pc = []


# from Blender_script.PybulletRecorder import PyBulletRecorder
# pb_recorder = PyBulletRecorder(client_id=client_id)
# pb_recorder.register_object(obj_id)

# for i in range(1):
#     p.stepSimulation()
#     pb_recorder.add_keyframe()
#     time.sleep(1./240.)

# pb_recorder.save("test.pkl")


# Point Cloud Extraction Speed Test
from PointNet_Model.pointnet2_cls_ssg import get_model
import torch
import time

device = "cuda:0"
pc_extractor = get_model(num_class=40, normal_channel=False).to(device) # num_classes is used for loading checkpoint to make sure the model is the same
pc_extractor.load_checkpoint(ckpt_path="PointNet_Model/checkpoints/best_model.pth", evaluate=True, map_location=device)
test_pc = torch.rand((1, 3, 10240), device=device)
with torch.no_grad():
    print(f"Doing 1 batch PC")
    avg_time = 0
    for i in range(100):
      start_time = time.time()
      feature = pc_extractor(test_pc)
      avg_time += time.time() - start_time
      print(f"Feature:", feature[0])
      print(feature.shape)
    print(f"1 Batch PC 100 times Average time: {avg_time/100.}")


# test_pc = torch.rand((20, 3, 10240), device=device)
# with torch.no_grad():
#     print(f"Doing 20 batches PC")
#     avg_time = 0
#     for i in range(100):
#       start_time = time.time()
#       pc_extractor(test_pc)
#       avg_time += time.time() - start_time
#     print(f"20 Batches PC 100 times Average time: {avg_time/100.}")

