import pyrealsense2 as rs
import cv2
import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt


class RealSenseCamera:
    def __init__(self, img_w=640, img_h=480, fps=30, color_align=True, serial_number=None) -> None:
        # Configure depth and color streams
        # 640x480 both 30 fps; 1280x720 depth only 6 fps and color only 15 fps
        # "Couldn't resolve requests - 'self.pipeline.start(config)'"
        self.pipeline = rs.pipeline()
        self.cfg = rs.config()
        self.serial_number = serial_number
        if serial_number is not None:
            self.cfg.enable_device(serial_number)
        self.img_w = img_w
        self.img_h = img_h
        self.cfg.enable_stream(rs.stream.color, img_w, img_h, rs.format.rgb8, fps)
        self.cfg.enable_stream(rs.stream.depth, img_w, img_h, rs.format.z16, fps)
        
        # Start streaming
        self.start_stream()
        self.aligner = rs.align(rs.stream.color) if color_align else None
        self.filter_group = self.init_filters()

        # Test reading and also refresh intrinsics
        self.get_rgbd_frame()

    
    def start_stream(self):
        self.pp_profile = self.pipeline.start(self.cfg)
        self.depth_scale = self.pp_profile.get_device().first_depth_sensor().get_depth_scale()

    
    def stop_stream(self):
        self.pipeline.stop()


    def init_filters(self):
        decimation = rs.decimation_filter()
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        return [decimation, spatial, temporal]
    

    def refresh_intrinsics(self, color_frame, depth_frame):
        # The camera intrinsics are different with color and depth frames; It is safe to refresh them
        self.depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
        self.depth_intrin_mat = np.array([[self.depth_intrinsics.fx, 0, self.depth_intrinsics.ppx],
                                          [0, self.depth_intrinsics.fy, self.depth_intrinsics.ppy],
                                          [0, 0, 1]])
        self.color_intrin_mat = np.array([[self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
                                          [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
                                          [0, 0, 1]])
    

    def get_rgbd_frame(self, convert2m=True):
        frames = self.pipeline.wait_for_frames()
        if self.aligner is not None:
            frames = self.aligner.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Post processing depth frame
        for f in self.filter_group:
            depth_frame = f.process(depth_frame)
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        # Why the depth image is (320, 240) while the color image is (640, 480)?
        # After resize, we can not use the original depth intrinsics to calculate the 3D points, need to adjust intrinsics using adjust_intrinsics
        depth_img = cv2.resize(depth_img, (color_img.shape[1], color_img.shape[0])) # Resize depth image to match color image; Why they are mismatch?

        # Convert depth to m
        if convert2m:
            depth_img = depth_img * self.depth_scale
        
        self.refresh_intrinsics(color_frame, depth_frame)
        return color_img, depth_img
    

    def get_pc_from_rgbd(self, color_img, depth_img, visual=False):
        assert self.aligner is not None, \
             "Color and depth images are not aligned! We currently do not support point cloud generation without alignment. Please specify color_align=True"
        color_intrin = self.color_intrinsics
        depth_intrin = color_intrin # We align the depth image to the color image so the intrinsics are the same

        # Correctly assign width and height
        height, width = depth_img.shape

        # Generate meshgrid for matrix operations
        xx, yy = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')

        # Calculate the real world coordinates
        x = (xx - depth_intrin.ppx) * depth_img / depth_intrin.fx
        y = (yy - depth_intrin.ppy) * depth_img / depth_intrin.fy
        z = depth_img

        # Ensure the arrays are of the same shape
        assert x.shape == y.shape == z.shape, "Shapes of x, y, and z do not match"

        # Reshape and stack to get 3D points in space
        xyz = np.dstack((x, y, z)).reshape(-1, 3)
        rgb = color_img.reshape(-1, 3)

        # Filter out invalid points
        valid_points = xyz[z.ravel() != 0]
        valid_rgb = rgb[z.ravel() != 0]

        if visual:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(valid_points)
            pcd.colors = o3d.utility.Vector3dVector(valid_rgb / 255.0)
            o3d.visualization.draw_geometries([pcd])

        return valid_points, valid_rgb
    

    def __del__(self):
        self.stop_stream()
    


def adjust_intrinsics(intrinsics, original_size, new_size):
    # Extract original intrinsic parameters
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
    
    # Calculate scale factors
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]
    
    # Adjust intrinsics
    new_fx = fx * scale_x
    new_fy = fy * scale_y
    new_cx = cx * scale_x
    new_cy = cy * scale_y
    
    # Return adjusted intrinsics
    adjusted_intrinsics = {
        'fx': new_fx, 'fy': new_fy,
        'cx': new_cx, 'cy': new_cy
    }
    return adjusted_intrinsics


if __name__ == "__main__":

    # Configure depth and color streams
    context = rs.context()
    device_list = context.query_devices()

    camera = RealSenseCamera(color_align=True)

    try:
        frame_count = 0
        while True:  # Infinite loop
            color_image, depth_image = camera.get_rgbd_frame()

            # Apply colormap on depth image
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255/depth_image.max()), cv2.COLORMAP_JET)
            valid_points, valid_rgb = camera.get_pc_from_rgbd(color_image, depth_image)

            # Show images
            BGR_color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('RealSenseCamera', np.hstack((BGR_color_image, depth_colormap)))

            # hot keys: press w to capture, press q to quit
            key = cv2.waitKey(1)
            if key == ord('w'):
                cv2.imwrite(f'dataset/color_images/color_image_{frame_count}.png', color_image)
                cv2.imwrite(f'dataset/depth_images/depth_colormap_{frame_count}.png', depth_colormap)
                o3d.io.write_point_cloud(f'dataset/point_clouds/point_cloud_{frame_count}.ply', filtered_pcd)
                frame_count += 1
            elif key == ord('q'):
                break


    finally:
        camera.stop_stream()