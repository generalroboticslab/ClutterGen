
import numpy as np
import json
from PIL import Image
import cv2
import os
import open3d as o3d
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import torch


def save_pointcloud_o3d(depth_path, rgb_path, cam_intrinsic_matrix, cam_extrinsic_matrix, output_path):
    # Load depth and RGB images
    depth = np.load(depth_path).astype(np.float32)
    color = np.array(Image.open(rgb_path))
    
    depth_img = o3d.geometry.Image(depth)
    color_img = o3d.geometry.Image(color)
    
    # Create an RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, 
                                                              depth_img, 
                                                              depth_scale=1.0, 
                                                              depth_trunc=1.0, 
                                                              convert_rgb_to_intensity=False)
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.intrinsic_matrix = cam_intrinsic_matrix
    
    intrinsic.height = 256
    intrinsic.width = 256
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    # print(cam_extrinsic_matrix)
    pcd.transform(cam_extrinsic_matrix)

    o3d.io.write_point_cloud(output_path, pcd)

    return pcd


def get_pointcloud_o3d(color, depth, cam_intrinsic_matrix, cam_extrinsic_matrix, output_path):
    # Load depth and RGB images
    
    depth_img = o3d.geometry.Image(depth)
    color_img = o3d.geometry.Image(color)
    
    # Create an RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, 
                                                              depth_img, 
                                                              depth_scale=1.0, 
                                                              depth_trunc=30., 
                                                              convert_rgb_to_intensity=False)
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.intrinsic_matrix = cam_intrinsic_matrix
    intrinsic.height = 256
    intrinsic.width = 256
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    # print(cam_extrinsic_matrix)
    pcd.transform(cam_extrinsic_matrix)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20., origin=[0, 0, 0])

    # o3d.io.write_point_cloud(output_path, pcd)
    print(pcd.points)
    o3d.visualization.draw_geometries([pcd, coord_frame])

    return pcd


def rgbd_to_point_cloud(rgb_image, depth_image, intrinsic_matrix):
    # Convert input NumPy arrays to PyTorch tensors
    depth_image_tensor = torch.tensor(depth_image, dtype=torch.float32).to("cuda")
    rgb_image_tensor = torch.tensor(rgb_image, dtype=torch.uint8).to("cuda")
    intrinsic_matrix_tensor = torch.tensor(intrinsic_matrix, dtype=torch.float32).to("cuda")
    
    # Extract height and width from the RGB image
    height, width, _ = rgb_image.shape
    
    # Create 2D grids representing pixel coordinates; v is height, u is width
    v, u = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    u, v = u.to("cuda"), v.to("cuda")
    
    # Compute 3D coordinates
    print(u.shape, v.shape, depth_image_tensor.shape, intrinsic_matrix_tensor.shape)
    x = ((u - intrinsic_matrix_tensor[0, 2]) * depth_image_tensor / intrinsic_matrix_tensor[0, 0])
    y = ((v - intrinsic_matrix_tensor[1, 2]) * depth_image_tensor / intrinsic_matrix_tensor[1, 1])
    z = depth_image_tensor
    
    # Stack the 3D coordinates and RGB values
    points_3d = torch.stack((x, y, z), dim=2)
    colors = rgb_image_tensor
    
    # Reshape tensors to flatten them
    points_3d = points_3d.view(-1, 3)
    colors = colors.view(-1, 3)
    
    return points_3d.cpu().numpy(), colors.cpu().numpy()


if __name__=="__main__":
    # Define intrinsic and extrinsic matrices
    org_image = cv2.imread("assets/image_dataset/scratch/testenv.jpg")
    depth_image = np.load("assets/image_dataset/scratch/testenv.npy")
    # depth_img = depth_img.resize((256, 256))
    # org_image = org_image.resize((256, 256))

    # Intrinsic matrix parameters
    # fx = 166.81  # Focal length in x-direction
    # fy = 166.81  # Focal length in y-direction
    # cx = 128.0   # X-coordinate of the principal point
    # cy = 128.0   # Y-coordinate of the principal point

    # fx = 1  # Focal length in x-direction
    # fy = 1  # Focal length in y-direction
    # cx = 0.   # X-coordinate of the principal point
    # cy = 0.   # Y-coordinate of the principal point

    fx = 422.364  # Focal length in x-direction
    fy = 422.364  # Focal length in y-direction
    cx = 128.0   # X-coordinate of the principal point
    cy = 128.0   # Y-coordinate of the principal point

    # # Create the intrinsic matrix
    # intrinsic_matrix = np.array([[fx, 0, cx],
    #               [0, fy, cy],
    #               [0, 0, 1]])

    # print("Intrinsic Matrix:")
    # print(intrinsic_matrix)

    # extrinsic_translation = np.array([0, 0, 0])  # Translation vector [x, y, z]
    # extrinsic_rotation = np.eye(3)  # Identity matrix for rotation (camera looking along -z axis)
    # extrinsic_matrix = np.hstack([extrinsic_rotation, extrinsic_translation.reshape(-1, 1)])
    # extrinsic_matrix = np.vstack([extrinsic_matrix, [0, 0, 0, 1]])  # Homogeneous coordinates

    # print("Extrinsic Matrix:")
    # print(extrinsic_matrix)

    # points_cloud = get_pointcloud_o3d(org_image, depth_img, intrinsic_matrix, extrinsic_matrix, "assets/image_dataset/scratch/testenv.pcd")

    # Example usage
    # Load RGB and depth images (replace 'rgb_image.png' and 'depth_image.png' with your file paths)

    # Define the intrinsic matrix (replace with your camera's intrinsic parameters)
    intrinsic_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]])

    # Convert RGBD image to point cloud and colors
    points_3d, colors = rgbd_to_point_cloud(org_image, depth_image, intrinsic_matrix)

    # # Create an Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)
    point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # Visualize the point cloud using Open3D
    o3d.visualization.draw_geometries([point_cloud])
