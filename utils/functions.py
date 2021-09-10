"""
Useful Functions.

Authors: Hongjie Fang.
"""
import torch
import einops
import numpy as np
import torch.nn.functional as F


def display_results(metrics_dict, logger):
    """
    Given a metrics dict, display the results using the logger.

    Parameters
    ----------
        
    metrics_dict: dict, required, the given metrics dict;

    logger: logging.Logger object, the logger.
    """
    try:
        display_list = []
        for key in metrics_dict.keys():
            if key == 'samples':
                num_samples = metrics_dict[key]
            else:
                display_list.append([key, float(metrics_dict[key])])
        logger.info("Metrics on {} samples:".format(num_samples))
        for display_line in display_list:
            metric_name, metric_value = display_line
            logger.info("  {}: {:.6f}".format(metric_name, metric_value))    
    except Exception:
        logger.warning("Unable to display the results, the operation is ignored.")
        pass


def gradient(x):
    """
    Get gradient of xyz image.

    This is adapted from implicit-depth repository, ref: https://github.com/NVlabs/implicit_depth/blob/main/src/utils/point_utils.py.

    Parameters
    ----------
    
    x: the xyz map to get gradient.

    Returns
    -------

    the x-axis-in-image gradient and y-axis-in-image gradient of the xyz map.
    """
    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]
    dx, dy = right - left, bottom - top 
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0
    return dx, dy


def get_surface_normal_from_xyz(x, epsilon = 1e-8):
    """
    Get the surface normal of xyz image.

    This is adapted from implicit-depth repository, ref: https://github.com/NVlabs/implicit_depth/blob/main/src/utils/point_utils.py.

    Parameters
    ----------
    
    x: the xyz map to get surface normal;
    
    epsilon: float, optional, default: 1e-8, the epsilon to avoid nan.

    Returns
    -------

    The surface normals.
    """
    dx, dy = gradient(x)
    surface_normal = torch.cross(dx, dy, dim = 1)
    surface_normal = surface_normal / (torch.norm(surface_normal, dim = 1, keepdim=True) + epsilon)
    return surface_normal


def get_xyz(depth, fx, fy, cx, cy, original_size = (1280, 720)):
    """
    Get XYZ from depth image and camera intrinsics.

    Parameters
    ----------
    
    depth: tensor, required, the depth image;

    fx, fy, cx, cy: tensor, required, the camera intrinsics;

    original_size: tuple of (int, int), optional, default: (1280, 720), the original size of image.

    Returns
    -------
    
    The XYZ value of each pixel.
    """
    bs, h, w = depth.shape
    indices = np.indices((h, w), dtype=np.float32)
    indices = torch.FloatTensor(np.array([indices] * bs)).to(depth.device)
    x_scale = w / original_size[0]
    y_scale = h / original_size[1]
    fx *= x_scale
    fy *= y_scale
    cx *= x_scale
    cy *= y_scale
    z = depth
    x = (indices[:, 1, :, :] - einops.repeat(cx, 'bs -> bs h w', h = h, w = w)) * z / einops.repeat(fx, 'bs -> bs h w', h = h, w = w)
    y = (indices[:, 0, :, :] - einops.repeat(cy, 'bs -> bs h w', h = h, w = w)) * z / einops.repeat(fy, 'bs -> bs h w', h = h, w = w)
    return torch.stack([x, y, z], axis = 1)


def get_surface_normal_from_depth(depth, fx, fy, cx, cy, original_size = (1280, 720), epsilon = 1e-8):
    """
    Get surface normal from depth and camera intrinsics.

    Parameters
    ----------
    
    depth: tensor, required, the depth image;

    fx, fy, cx, cy: tensor, required, the camera intrinsics;
  
    original_size: tuple of (int, int), optional, default: (1280, 720), the original size of image;

    epsilon: float, optional, default: 1e-8, the epsilon to avoid nan.

    Returns
    -------
    
    The surface normals.
    """
    xyz = get_xyz(depth, fx, fy, cx, cy, original_size = original_size)
    return get_surface_normal_from_xyz(xyz, epsilon = epsilon)


def to_device(data_dict, device):
    """
    Put the data in the data_dict to the specified device.
    
    Parameters
    ----------
    
    data_dict: dict, required, dict that contains tensors;
    
    device: torch.device object, required, the device.

    Returns
    -------

    The final data_dict.
    """
    for key in data_dict.keys():
        data_dict[key] = data_dict[key].to(device)
    return data_dict