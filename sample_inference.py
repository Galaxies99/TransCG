import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from inference import Inferencer


def draw_point_cloud(color, depth, camera_intrinsics, use_mask = False, use_inpainting = True, scale = 1000.0, inpainting_radius = 5, fault_depth_limit = 0.2, epsilon = 0.01):
    """
    Given the depth image, return the point cloud in open3d format.
    The code is adapted from [graspnet.py] in the [graspnetAPI] repository.
    """
    d = depth.copy()
    c = color.copy() / 255.0
    
    if use_inpainting:
        fault_mask = (d < fault_depth_limit * scale)
        d[fault_mask] = 0
        inpainting_mask = (np.abs(d) < epsilon * scale).astype(np.uint8)  
        d = cv2.inpaint(d, inpainting_mask, inpainting_radius, cv2.INPAINT_NS)

    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    xmap, ymap = np.arange(d.shape[1]), np.arange(d.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = d / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    points = np.stack([points_x, points_y, points_z], axis = -1)

    if use_mask:
        mask = (points_z > 0)
        points = points[mask]
        c = c[mask]
    else:
        points = points.reshape((-1, 3))
        c = c.reshape((-1, 3))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(c)
    return cloud


inferencer = Inferencer()

rgb = np.array(Image.open('data/scene1/1/rgb1.png'), dtype = np.float32)
depth = np.array(Image.open('data/scene1/1/depth1.png'), dtype = np.float32)

depth = depth / 1000

res = inferencer.inference(rgb, depth)

cam_intrinsics = np.load('data/camera_intrinsics/camIntrinsics-D435.npy')

res = np.clip(res, 0.1, 1.5)
depth = np.clip(depth, 0.1, 1.5)

cloud = draw_point_cloud(rgb, res, cam_intrinsics, scale = 1.0)

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
sphere = o3d.geometry.TriangleMesh.create_sphere(0.002,20).translate([0,0,0.490])
o3d.visualization.draw_geometries([cloud, frame, sphere])

