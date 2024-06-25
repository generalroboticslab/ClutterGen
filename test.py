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
# pc_extractor = get_model(num_class=40, normal_channel=False).to(device) # num_classes is used for loading checkpoint to make sure the model is the same
# pc_extractor.load_checkpoint(ckpt_path="PointNet_Model/checkpoints/best_model.pth", evaluate=True, map_location=device)
# test_pc = torch.rand((1, 3, 10240), device=device)
# with torch.no_grad():
#     print(f"Doing 1 batch PC")
#     avg_time = 0
#     for i in range(100):
#       start_time = time.time()
#       feature = pc_extractor(test_pc)
#       avg_time += time.time() - start_time
#       print(f"Feature:", feature[0])
#       print(feature.shape)
#     print(f"1 Batch PC 100 times Average time: {avg_time/100.}")


# test_pc = torch.rand((20, 3, 10240), device=device)
# with torch.no_grad():
#     print(f"Doing 20 batches PC")
#     avg_time = 0
#     for i in range(100):
#       start_time = time.time()
#       pc_extractor(test_pc)
#       avg_time += time.time() - start_time
#     print(f"20 Batches PC 100 times Average time: {avg_time/100.}")


# def torch_gpu_memory_usage_during_forward_pass(model, input_size, dtype=torch.float32):
#     """
#     Estimates the GPU memory usage of a PyTorch model during a forward pass.
#     """
#     # Ensure model is in eval mode to disable dropout, batchnorm, etc.
#     model.eval()

#     # Create a dummy input tensor
#     dummy_input = torch.zeros(*input_size, dtype=dtype)

#     # Move model and dummy input to GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     dummy_input = dummy_input.to(device)

#     # Measure GPU memory before the forward pass
#     # Reset peak memory stats
#     torch.cuda.reset_peak_memory_stats()
#     initial_memory = torch.cuda.memory_allocated(device)

#     # Forward pass
#     with torch.no_grad():
#         model(dummy_input)

#     # Measure GPU memory after the forward pass
#     after_memory = torch.cuda.memory_allocated(device)
#     peak_memory_during_forward = torch.cuda.max_memory_allocated(device)

#     # Calculate the memory used by the model during the forward pass
#     memory_used_during_forward = after_memory - initial_memory
#     peak_memory_usage = peak_memory_during_forward - initial_memory

#     # Clean up
#     del dummy_input
#     model.to('cpu')
#     torch.cuda.empty_cache()

#     return memory_used_during_forward, peak_memory_usage

# Usage Example
# init_gpu_memory = torch.cuda.memory_allocated(device)
# bunch_models = []
# for i in range(20):
#   pc_extractor = get_model(num_class=40, normal_channel=False).to(device) # num_classes is used for loading checkpoint to make sure the model is the same
#   pc_extractor.load_checkpoint(ckpt_path="PointNet_Model/checkpoints/best_model.pth", evaluate=True, map_location=device)
#   bunch_models.append(pc_extractor)
# model_gpu_memory = torch.cuda.memory_allocated(device) - init_gpu_memory
# print(f"Model GPU memory usage: {model_gpu_memory / (1024 ** 2)} MB")

# input_size = (20, 3, 10240)
# memory_used, peak_memory = torch_gpu_memory_usage_during_forward_pass(pc_extractor, input_size)
# print(f"Memory used during forward pass: {memory_used / (1024 ** 2)} MB")
# print(f"Input size: {input_size}; Peak memory usage during forward pass: {peak_memory / (1024 ** 2)} MB")


# Pybullet Step time test with different number of objects
# import pybullet_utils as pu
# import time
# import pybullet as p
# import pybullet_data
# import os
# import numpy as np
# import random

# client_id = p.connect(p.DIRECT)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -9.8)
# number_of_objects = 10

# for i in range(number_of_objects):
#     obj_id = p.loadURDF("assets/union_objects_test/AlarmClock/1/mobility.urdf", basePosition=[0, 0, 10.], useFixedBase=False, globalScaling=0.2)
#     pu.change_obj_color(obj_id, rgba_color=[random.random(), random.random(), random.random(), 1.])
#     # pu.set_mass(obj_id, 0.)
#     # p.setCollisionFilterGroupMask(obj_id, -1, 0, 0)
  
# start_time = time.time()
# for i in range(240):
#     p.stepSimulation()
# time_elapsed = time.time() - start_time

# print(f"Time elapsed: {time_elapsed} s for {number_of_objects} objects")


# from moviepy.editor import VideoFileClip

# def convert_mp4_to_gif(video_path, gif_path, start_time=0, end_time=None, resize=None):
#     """
#     Converts a segment of a MP4 video to a GIF image.

#     Parameters:
#     - video_path: Path to the source MP4 video file.
#     - gif_path: Path where the GIF should be saved.
#     - start_time: Start time in seconds for the GIF segment.
#     - end_time: End time in seconds for the GIF segment. If None, goes to the end of the video.
#     - resize: Tuple of new size (width, height), or None to keep original size.
#     """
    
#     # Load the video file
#     clip = VideoFileClip(video_path)
    
#     # If an end time is not specified, use the full duration of the clip
#     if end_time is None:
#         end_time = clip.duration
    
#     # Trim the clip to the desired segment
#     gif_clip = clip.subclip(start_time, end_time)
    
#     # Resize the clip if a new size is specified
#     if resize is not None:
#         gif_clip = gif_clip.resize(newsize=resize)
    
#     # Write the GIF
#     gif_clip.write_gif(gif_path, fps=10)  # Adjust fps for different quality/speed

# # Example usage
# video_path = 'eval_res/Union/blender/Research_presentation_recording/10Objs_2eps_success_blender.mp4'  # Update this to your video file path
# gif_path = 'eval_res/Union/blender/Research_presentation_recording/10Objs_2eps_success_blender.gif'  # Desired output GIF path
# convert_mp4_to_gif(video_path, gif_path, start_time=0, end_time=5)  # Example conversion

# from moviepy.editor import VideoFileClip, vfx
# def slow_down_video(input_video_path, output_video_path, slowdown_factor=4):
#     """
#     Slows down a video by the specified slowdown factor and saves it as a new video file.

#     Parameters:
#     - input_video_path: Path to the source MP4 video file.
#     - output_video_path: Path where the slowed down video should be saved.
#     - slowdown_factor: Factor by which the video speed should be reduced. Default is 4.
#     """
    
#     # Load the source video
#     video_clip = VideoFileClip(input_video_path)
    
#     # Apply the slowdown effect
#     slowed_clip = video_clip.fx(vfx.speedx, 1 / slowdown_factor)
    
#     # Write the output video file
#     slowed_clip.write_videofile(output_video_path, audio_codec='aac')  # Ensure compatibility with MP4 format

# # Example usage
# input_video_path = 'eval_res/Union/blender/Union_02-19_15:44Sync_table_PCExtractor_Relu_Rand_ObjPlace_QRRegion_Goal_minObjNum2_objStep2_maxObjNum10_maxPool10_maxScene1_maxStable60_contStable20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_EVAL_best_objRange_10_10/render_results/10Objs_3eps_success_blender.mp4'  # Update this to your video file path
# output_video_path = 'eval_res/Union/blender/Union_02-19_15:44Sync_table_PCExtractor_Relu_Rand_ObjPlace_QRRegion_Goal_minObjNum2_objStep2_maxObjNum10_maxPool10_maxScene1_maxStable60_contStable20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_EVAL_best_objRange_10_10/render_results/10Objs_3eps_success_blender_slow_down.mp4'  # Desired output video file path
# slow_down_video(input_video_path, output_video_path)


# client_id = p.connect(p.DIRECT)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -9.8)
# number_of_objects = 10

# for i in range(number_of_objects):
#     obj_id = p.loadURDF("assets/union_objects_test/AlarmClock/1/mobility.urdf", basePosition=[0, 0, 10.], useFixedBase=False, globalScaling=0.2)
#     pu.change_obj_color(obj_id, rgba_color=[random.random(), random.random(), random.random(), 1.])
#     # pu.set_mass(obj_id, 0.)
#     # p.setCollisionFilterGroupMask(obj_id, -1, 0, 0)
  
# start_time = time.time()
# for i in range(240):
#     p.stepSimulation()
# time_elapsed = time.time() - start_time

# print(f"Time elapsed: {time_elapsed} s for {number_of_objects} objects")


# Camera Calibration
# import pybullet_utils_cust as pu
# import open3d as o3d
# import numpy as np

# client_id = p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -9.8)
# p.loadURDF("plane.urdf")
# tableHalfExtents = [0.2, 0.3, 0.35]
# tableId = pu.create_box_body(position=[0., 0., tableHalfExtents[2]], orientation=p.getQuaternionFromEuler([0., 0., 0.]),
#                              halfExtents=tableHalfExtents, rgba_color=[1, 1, 1, 1], mass=0., client_id=client_id)
# obj_id = p.loadURDF("assets/group_objects/group0_dinning_table/005_tomato_soup_can/0/mobility.urdf", basePosition=[0, 0, 0.72], useFixedBase=True)

# def compute_camera_matrix(cameraEyePosition, cameraTargetPosition, cameraUpVector=[0, 0, 1]):
#     return p.computeViewMatrix(cameraEyePosition, cameraTargetPosition, cameraUpVector)

# def get_camera_image(viewMatrix, projMatrix=None, width=224, height=224, client_id=0):
#     width, height, rgbImg, depthImg, segImg = p.getCameraImage(
#         width=width, height=height, 
#         viewMatrix=viewMatrix, 
#         projectionMatrix=projMatrix,
#         renderer=p.ER_BULLET_HARDWARE_OPENGL, 
#         flags=p.ER_NO_SEGMENTATION_MASK,
#         physicsClientId=client_id)
#     return rgbImg, depthImg, segImg

# tableheight = tableHalfExtents[2] * 2
# camera_height = 0.3 + tableheight
# camera1_view_matrix_1 = compute_camera_matrix([tableHalfExtents[0]*2, 0., camera_height], [0, 0, tableheight])
# camera1_view_matrix_2 = compute_camera_matrix([0, tableHalfExtents[1]*2, camera_height], [0, 0, tableheight])
# camera1_view_matrix_3 = compute_camera_matrix([-tableHalfExtents[0]*2, 0., camera_height], [0, 0, tableheight])
# camera1_view_matrix_4 = compute_camera_matrix([0, -tableHalfExtents[1]*2, camera_height], [0, 0, tableheight])


# def depth_image_to_point_cloud(depth_image, camera_intrinsics, view_matrix, projection_matrix):
#     """
#     Convert a depth image to a point cloud in the world coordinate system.
#     """
#     # Get the intrinsic parameters of the camera
#     fx, _, cx, _, fy, cy, _, _, _ = camera_intrinsics.flatten()

#     # Create a mesh grid of pixel coordinates
#     height, width = depth_image.shape
#     u, v = np.meshgrid(np.arange(width), np.arange(height))

#     # Convert depth image to z values in camera coordinates
#     z = depth_image

#     # Convert pixel coordinates to camera coordinates
#     x = (u - cx) * z / fx
#     y = (v - cy) * z / fy

#     # Combine x, y, z to camera coordinates
#     points_camera = np.stack((x, y, z), axis=-1)

#     # Convert camera coordinates to world coordinates
#     view_matrix = np.array(view_matrix).reshape(4, 4).T
#     projection_matrix = np.array(projection_matrix).reshape(4, 4).T
#     camera_matrix = np.linalg.inv(projection_matrix @ view_matrix)

#     # Add a dimension for matrix multiplication
#     points_camera_hom = np.concatenate([points_camera, np.ones((*points_camera.shape[:-1], 1))], axis=-1)
#     points_world_hom = points_camera_hom @ camera_matrix.T

#     # Convert homogenous coordinates to 3D
#     points_world = points_world_hom[..., :3] / points_world_hom[..., 3:]

#     return points_world.reshape(-1, 3)


# def get_point_cloud(width, height, view_matrix, proj_matrix):
#     # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer

#     # get a depth image
#     # "infinite" depths will have a value close to 1
#     image_arr = p.getCameraImage(width=width, 
#                                  height=height, 
#                                  viewMatrix=view_matrix, 
#                                  projectionMatrix=proj_matrix)
#     depth = image_arr[3]

#     # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
#     proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
#     view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
#     world2pixels = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

#     # create a grid with pixel coordinates and depth values
#     y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
#     y *= -1.
#     x, y, z = x.reshape(-1), y.reshape(-1), np.array(depth).reshape(-1)
#     h = np.ones_like(z)

#     pixels = np.stack([x, y, z, h], axis=1)
#     # filter out "infinite" depths
#     pixels = pixels[z < 0.99]
#     pixels[:, 2] = 2 * pixels[:, 2] - 1

#     # turn pixels to world coordinates
#     points = np.matmul(world2pixels, pixels.T).T
#     points /= points[:, 3: 4]
#     points = points[:, :3]

#     return points


# def compute_intrinsic_matrix(fov, width, height):
#     # Assuming fov is in degrees and width, height are the dimensions of the camera image
#     fov_rad = np.deg2rad(fov)
#     f = width / (2 * np.tan(fov_rad / 2))  # Focal length calculation

#     cx = width / 2
#     cy = height / 2

#     intrinsic_matrix = np.array([
#         [f, 0, cx],
#         [0, f, cy],
#         [0, 0, 1]
#     ])

#     return intrinsic_matrix

# # Usage example
# width = 224  # Width of the depth image
# height = 224  # Height of the depth image
# fov = 60  # Field of view in degrees
# aspect = width / height  # Aspect ratio
# near = 0.02  # Near clipping plane
# far = 5.0  # Far clipping plane

# projection_matrix = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=near, farVal=far)
# camera_intrinsics = compute_intrinsic_matrix(fov, width, height)

# points_world_1 = get_point_cloud(width, height, camera1_view_matrix_1, projection_matrix)
# points_world_2 = get_point_cloud(width, height, camera1_view_matrix_2, projection_matrix)
# points_world_3 = get_point_cloud(width, height, camera1_view_matrix_3, projection_matrix)
# points_world_4 = get_point_cloud(width, height, camera1_view_matrix_4, projection_matrix)
# points_world = np.concatenate([points_world_1, points_world_2, points_world_3, points_world_4], axis=0)
# points_world = points_world[points_world[:, 2] >= 0.7]
# pu.visualize_pc(points_world)


# from ur5_robotiq_controller import UR5RobotiqPybulletController, load_ur_robotiq_robot
# import pybullet_utils_cust as pu
# from pynput import keyboard
# client_id = p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -9.81, physicsClientId=client_id)
# plane_id = p.loadURDF("plane.urdf", physicsClientId=client_id)
# pu.change_obj_color(plane_id, rgba_color=[1., 1., 1., 0.3])

# tableHalfExtents = [0.4, 0.5, 0.35]
# tablepos = [0.7, 0., tableHalfExtents[2]]
# tableId = pu.create_box_body(position=tablepos, orientation=p.getQuaternionFromEuler([0., 0., 0.]),
#                                       halfExtents=tableHalfExtents, rgba_color=[1, 1, 1, 1], mass=0., client_id=client_id)
# test_object = p.loadURDF("assets/group_objects/group0_dinning_table/005_tomato_soup_can/0/mobility.urdf", 
#                           basePosition=[tablepos[0], 0, 0.8], useFixedBase=False)
# pu.set_mass(test_object, 1., client_id=client_id)

# vis_gripper_id = p.loadURDF("robot_related/ur5_robotiq_description/urdf/robotiq_2f_85_gripper_visualization/urdf/robotiq_arg2f_85_model.urdf", 
#                             basePosition=[0, 0, 0.], 
#                             useFixedBase=True)

# pu.change_obj_color(vis_gripper_id, rgba_color=[0., 0., 0., .2])
# gripper_links_num = pu.get_num_links(vis_gripper_id, client_id=client_id)
# for link_id in [-1]+list(range(gripper_links_num)):
#     p.setCollisionFilterGroupMask(vis_gripper_id, link_id, 0, 0, physicsClientId=client_id)
# gripper_face_direction = [0., 0., 1.]

# robot_id, urdf_path = load_ur_robotiq_robot(robot_initial_pose=[[0., 0., 0.5], [0., 0., 0., 1.]], client_id=client_id)
# robot = UR5RobotiqPybulletController(robot_id, rng=None, client_id=client_id)
# robot.update_collision_check()
# robot.reset()

# with keyboard.Events() as events:
#     while True:
#         key = None
#         event = events.get(0.0001)
#         cur_gripper_pose = robot.get_gripper_base_pose()
#         if event is not None:
#           if isinstance(event, events.Press) and hasattr(event.key, 'char'):
#               key = event.key.char
#           if key is not None:
#               if key == 's':
#                   print(f"Before: {cur_gripper_pose}")
#                   cur_gripper_pose[0][2] -= 0.02
#                   print(f"Before: {cur_gripper_pose}")
#                   joint_v = robot.get_arm_ik(cur_gripper_pose)
#                   if joint_v is not None:
#                     robot.control_arm_joints(joint_v)
#               elif key == 'w':
#                   cur_gripper_pose[0][2] += 0.02
#                   joint_v = robot.get_arm_ik(cur_gripper_pose)
#                   if joint_v is not None:
#                     robot.control_arm_joints(joint_v)
#               elif key == 'a':
#                   cur_gripper_pose[0][0] -= 0.02
#                   joint_v = robot.get_arm_ik(cur_gripper_pose)
#                   if joint_v is not None:
#                     robot.control_arm_joints(joint_v)
#               elif key == 'd':
#                   cur_gripper_pose[0][0] += 0.02
#                   joint_v = robot.get_arm_ik(cur_gripper_pose)
#                   if joint_v is not None:
#                     robot.control_arm_joints(joint_v)
#               elif key == 'e':
#                  print(robot.get_arm_joint_values())

#         pu.step(client_id=client_id)


# import socket

# HEADER = 64
# PORT = 5050
# FORMAT = 'utf-8'
# DISCONNECT_MESSAGE = "!DISCONNECT"

# SERVER = "10.194.140.188"  # Change this to the server's IP address if it's not running locally
# ADDR = (SERVER, PORT)

# client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client.connect(ADDR)

# def send_message(message):
#     message_length = len(message)
#     send_length = str(message_length).encode(FORMAT)
#     send_length += b' ' * (HEADER - len(send_length))
#     client.send(send_length)
#     client.send(message.encode(FORMAT))
#     print(client.recv(2048).decode(FORMAT))

# send_message("Hello from the client!")
# send_message("Hello from the client!")  # Assuming this triggers a specific action on the server

# Disconnect from the server
# send_message(DISCONNECT_MESSAGE)


# Visualize Mesh using trimesh
# # Create a scene from the mesh
# scene = trimesh.Scene(mesh)
# # Add coordinate axes to the scene
# # The size of the axes can be adjusted with the scale parameter
# axes = trimesh.creation.axis(origin_color=(0, 0, 0), origin_size=0.01)
# scene.add_geometry(axes)
# # Show the scene
# scene.show()
# transform_mesh = mesh.apply_transform(to_origin)
# transform_scene = trimesh.Scene(transform_mesh)
# # Add coordinate axes to the scene
# # The size of the axes can be adjusted with the scale parameter
# # Create the oriented bounding box using the obtained transformation and extents
# obb = trimesh.primitives.Box(transform=to_origin, extents=extents)
# # The OBB is a mesh itself, so we can set its visual style to make it semi-transparent and colored for visibility
# obb.visual.face_colors = [200, 200, 250, 100]  # RGBA, A for alpha (transparency)
# # Add the oriented bounding box to the scene
# transform_scene.add_geometry(obb)
# axes = trimesh.creation.axis(origin_color=(0, 0, 0), origin_size=0.01)
# transform_scene.add_geometry(axes)
# transform_scene.show()

"""
How to use logging info
"""
# import logging
# import importlib

# # Configure logging
# def set_logging_format(level=logging.INFO, simple=True):
#   importlib.reload(logging)
#   FORMAT = '[%(funcName)s] %(message)s' if simple else '%(asctime)s - %(levelname)s - %(message)s'
#   logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

# set_logging_format(level=logging.INFO)
# # Log some messages
# logging.debug("This is a debug message")
# logging.info("This is an info message")
# logging.warning("This is a warning message")
# logging.error("This is an error message")
# logging.critical("This is a critical message")

# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt
# from realsense_camera import RealSenseCamera
# import cv2

# camera = RealSenseCamera()
# color_intrin_mat = camera.color_intrin_mat
# fx, fy, cx, cy = color_intrin_mat[0, 0], color_intrin_mat[1, 1], color_intrin_mat[0, 2], color_intrin_mat[1, 2]
# width, height = camera.img_w, camera.img_h

# # Load your mesh
# mesh = o3d.io.read_triangle_mesh("assets/group_objects/group4_real_objects/23_dawn/0/textured_objs/textured.obj")
# mesh.compute_vertex_normals()  # Optional: compute normals if needed for rendering

# # Define camera intrinsic parameters
# intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, color_intrin_mat)

# # Define camera pose (6DOF: rotation + translation)
# # Example: No rotation and a slight translation
# # camera points to the positive z-axis
# World2Camera_pose = np.eye(4)

# # Create a transformation matrix for the mesh
# # Example: Rotate 45 degrees around the z-axis and translate by (0.5, 0.5, 0)
# rotation = o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi / 2, 0, 0.])
# translation = [0., 0., 0.5]
# transformation_matrix = np.eye(4)
# transformation_matrix[:3, :3] = rotation
# transformation_matrix[:3, 3] = translation

# # Apply the transformation to the mesh
# mesh.transform(transformation_matrix)

# # Render the mesh
# renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
# renderer.setup_camera(intrinsic, np.linalg.inv(World2Camera_pose)) # open3d uese camera-to-world pose
# renderer.scene.set_background([1.0, 1.0, 1.0, 0.])

# # Add mesh to scene for rendering
# renderer.scene.add_geometry("mesh", mesh, o3d.visualization.rendering.MaterialRecord())  # Adding default material

# # Add lighting
# # light_dir = np.array([0, -1, -1.])  # Example direction
# # light_color = np.array([1, 1, 1.])  # White light
# # renderer.scene.scene.add_directional_light("dir_light", light_color, light_dir, 50, False)

# # Create an image using the renderer
# img = renderer.render_to_image()
# # Convert Open3D Image to numpy array
# img_np = np.asarray(img)
# img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGRA)  # Ensure image has alpha

# # Load another image to blend with

# camera_img, _ = camera.get_rgbd_frame()
# camera_img = cv2.resize(camera_img, (width, height))
# camera_img = cv2.cvtColor(camera_img, cv2.COLOR_RGB2BGRA)
# # Create an alpha channel filled with 255 (fully opaque)
# alpha_channel = np.full((camera_img.shape[0], camera_img.shape[1]), 255, dtype=np.uint8)
# # Add alpha channel to the RGB image to make it RGBA
# camera_img_bgra = cv2.merge((camera_img[..., 0], camera_img[..., 1], camera_img[..., 2], alpha_channel))
# print(camera_img_bgra.shape, img_np.shape)
# # Combine the images
# combined_img = cv2.addWeighted(camera_img_bgra, 1.0, img_np, 0.3, 0)

# # Save and display the result
# # cv2.imwrite("combined_image.png", combined_img)
# plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGRA2RGBA))
# plt.axis('off')
# plt.show()


# Better way to render mesh
# import open3d as o3d

# # Load your point cloud
# pcd = o3d.io.read_point_cloud("your_point_cloud.ply")

# # Create a Visualizer
# vis = o3d.visualization.Visualizer()
# vis.create_window()

# # Add the point cloud to the visualizer
# vis.add_geometry(pcd)

# # Add a coordinate frame to the visualizer
# coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
# vis.add_geometry(coordinate_frame)

# # Get the view control and set the camera parameters
# view_ctl = vis.get_view_control()
# view_ctl.set_lookat([0.0, 0.0, 0.0])
# view_ctl.set_front([-1.0, 0.0, 0.5])
# view_ctl.set_up([0.0, 0.0, 1.0])
# view_ctl.set_zoom(0.8)  # Adjust zoom as needed

# # Run the visualizer
# vis.run()
# vis.destroy_window()

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.stats import beta

# import numpy as np
# import matplotlib.pyplot as plt

# # Define data points
# x = np.array([0, 1, 2])
# y = np.array([0, 1, 2])
# z = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9]])

# # Flatten input data
# x_flat = x.flatten()
# y_flat = y.flatten()
# z_flat = z.flatten()

# # Create a grid
# resolution = 100
# x_grid = np.linspace(0, 2, resolution)
# y_grid = np.linspace(0, 2, resolution)
# x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

# # Perform bilinear interpolation
# z_interp = np.interp(x_mesh.flatten(), x_flat, np.interp(y_mesh.flatten(), y_flat, z_flat)).reshape(x_mesh.shape)

# # Plot the gradient surface
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x_mesh, y_mesh, z_interp, cmap='viridis')

# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Smooth Gradient Surface')

# # Display the plot
# plt.show()

# Load mesh, reduce UV index, and save
# import trimesh
# import numpy as np

# mesh_path = "assets/group_objects/group2_office_table/Book/5/textured_objs/textured.obj"
# mesh = trimesh.load(mesh_path)
# mesh.export("assets/group_objects/group2_office_table/Book/5/textured.obj")

# import pandas as pd
# import wandb

# api = wandb.Api(timeout=10000)
# entity, project = "jiayinsen", "RoboSensai_SG"
# runs_id = [""]
# runs = api.run("jiayinsen/RoboSensai_SG/tsar4q3w")
# history = runs.history(samples=100000, keys=None, x_axis="s_iterations", pandas=(True), stream="default")