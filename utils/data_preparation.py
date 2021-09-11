"""
Data preparation, including scaling, augmentation and tensorize.

Authors: Authors from [implicit-depth] repository, Hongjie Fang.
Ref: 
    1. [implicit-depth] repository: https://github.com/NVlabs/implicit_depth
"""
import cv2
import torch
import Imath
import random
import OpenEXR
import numpy as np
from utils.functions import get_surface_normal_from_depth
from utils.constants import DILATION_KERNEL


def chromatic_transform(image):
    """
    Add the hue, saturation and luminosity to the image.

    This is adapted from implicit-depth repository, ref: https://github.com/NVlabs/implicit_depth/blob/main/src/utils/data_augmentation.py

    Parameters
    ----------

    image: array, required, the given image.

    Returns
    -------

    The new image after augmentation in HLS space.
    """
    # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1
    d_h = (np.random.rand(1) - 0.5) * 0.1 * 180
    d_l = (np.random.rand(1) - 0.5) * 0.2 * 256
    d_s = (np.random.rand(1) - 0.5) * 0.2 * 256
    # Convert the BGR to HLS
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    # Add the values to the image H, L, S
    new_h = (h + d_h) % 180
    new_l = np.clip(l + d_l, 0, 255)
    new_s = np.clip(s + d_s, 0, 255)
    # Convert the HLS to BGR
    new_hls = cv2.merge((new_h, new_l, new_s)).astype('uint8')
    new_image = cv2.cvtColor(new_hls, cv2.COLOR_HLS2BGR)
    return new_image


def add_noise(image, level = 0.1):
    """
    Add noise to the image.

    This is adapted from implicit-depth repository, ref: https://github.com/NVlabs/implicit_depth/blob/main/src/utils/data_augmentation.py

    Parameters
    ----------

    image: array, required, the given image;

    level: float, optional, default: 0.1, the maximum noise level.

    Returns
    -------

    The new image after augmentation of adding noises.
    """
    # random number
    r = np.random.rand(1)

    # gaussian noise
    if r < 0.9:
        row,col,ch= image.shape
        mean = 0
        noise_level = random.uniform(0, level)
        sigma = np.random.rand(1) * noise_level * 256
        gauss = sigma * np.random.randn(row,col) + mean
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
    else:
        # motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = np.zeros((size, size))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        else:
            kernel_motion_blur[:, int((size-1)/2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        noisy = cv2.filter2D(image, -1, kernel_motion_blur)

    return noisy.astype('uint8')


def exr_loader(exr_path, ndim = 3, ndim_representation = ['R', 'G', 'B']):
    """
    Loads a .exr file as a numpy array.

    This is adapted from implicit-depth repository, ref: https://github.com/NVlabs/implicit_depth/blob/main/src/utils/data_augmentation.py.

    Parameters
    ----------

    exr_path: path to the exr file
    
    ndim: number of channels that should be in returned array. Valid values are 1 and 3.
        - if ndim=1, only the 'R' channel is taken from exr file;
        - if ndim=3, the 'R', 'G' and 'B' channels are taken from exr file. The exr file must have 3 channels in this case.
    
    depth_representation: list of str, the representation of channels, default = ['R', 'G', 'B'].
    
    Returns
    -------

    numpy.ndarray (dtype=np.float32).
        - If ndim=1, shape is (height x width);
        - If ndim=3, shape is (3 x height x width)
    """

    exr_file = OpenEXR.InputFile(exr_path)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    assert ndim == len(ndim_representation), "ndim should match ndim_representation."

    if ndim == 3:
        # read channels indivudally
        allchannels = []
        for c in ndim_representation:
            # transform data to numpy
            channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)

        # create array and transpose dimensions to match tensor style
        exr_arr = np.array(allchannels).transpose((0, 1, 2))
        return exr_arr

    if ndim == 1:
        # transform data to numpy
        channel = np.frombuffer(exr_file.channel(ndim_representation[0], pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        exr_arr = np.array(channel)
        return exr_arr


def process_depth(depth, camera_type = 0, depth_min = 0.3, depth_max = 1.5, depth_norm = 1.0):
    """
    Process the depth information, including scaling, normalization and clear NaN values.
    
    Paramters
    ---------

    depth: array, required, the depth image;

    camera_type: int in [0, 1, 2], optional, default: 0, the camera type;
        - 0: no scale is applied;
        - 1: scale 1000 (RealSense D415, RealSense D435, etc.);
        - 2: scale 4000 (RealSense L515).
    
    depth_min, depth_max: int, optional, default: 0.3, 1.5, the min depth and the max depth;

    depth_norm: float, optional, default: 1.0, the depth normalization coefficient.

    Returns
    -------

    The depth image after scaling.
    """
    scale_coeff = 1
    if camera_type == 1:
        scale_coeff = 1000
    if camera_type == 2:
        scale_coeff = 4000
    depth = depth / scale_coeff
    depth[np.isnan(depth)] = 0.0
    depth = np.where(depth < depth_min, 0, depth)
    depth = np.where(depth > depth_max, 0, depth)
    depth = depth / depth_norm
    return depth


def process_data(
    rgb, 
    depth, 
    depth_gt, 
    depth_gt_mask, 
    camera_intrinsics, 
    scene_type = "cluttered", 
    camera_type = 0, 
    split = 'train', 
    image_size = (720, 1280), 
    depth_min = 0.3, 
    depth_max = 1.5,
    depth_norm = 10,
    use_aug = True, 
    rgb_aug_prob = 0.8, 
    **kwargs):
    """
    Process images and perform data augmentation.

    Parameters
    ----------

    rgb: array, required, the rgb image;
    
    depth: array, required, the original depth image;

    depth_gt: array, required, the ground-truth depth image;
    
    depth_gt_mask: array, required, the ground-truth depth image mask;

    camera_intrinsics: array, required, the camera intrinsics of the image;
    
    scene_type: str in ['cluttered', 'isolated'], optional, default: 'cluttered', the scene type;
    
    camera_type: int in [0, 1, 2], optional, default: 0, the camera type;
        - 0: no scale is applied;
        - 1: scale 1000 (RealSense D415, RealSense D435, etc.);
        - 2: scale 4000 (RealSense L515).
    
    split: str in ['train', 'test'], optional, default: 'train', the split of the dataset;
    
    image_size: tuple of (int, int), optional, default: (720, 1280), the size of the image;
    
    depth_min, depth_max: float, optional, default: 0.1, 1.5, the min depth and the max depth;

    depth_norm: float, optional, default: 1.0, the depth normalization coefficient;

    use_aug: bool, optional, default: True, whether use data augmentation;
    
    rgb_aug_prob: float, optional, default: 0.8, the rgb augmentation probability (only applies when use_aug is set to True).

    Returns
    -------
    
    data_dict for training and testing.
    """

    depth_original = process_depth(depth.copy(), camera_type = camera_type, depth_min = depth_min, depth_max = depth_max, depth_norm = depth_norm)
    depth_gt_original = process_depth(depth_gt.copy(), camera_type = camera_type,  depth_min = depth_min, depth_max = depth_max, depth_norm = depth_norm)
    depth_gt_mask_original = depth_gt_mask.copy()
    zero_mask_original = np.where(depth_gt_original < 0.01, False, True).astype(np.bool)

    rgb = cv2.resize(rgb, image_size, interpolation = cv2.INTER_NEAREST)
    depth = cv2.resize(depth, image_size, interpolation = cv2.INTER_NEAREST)
    depth_gt = cv2.resize(depth_gt, image_size, interpolation = cv2.INTER_NEAREST)
    depth_gt_mask = cv2.resize(depth_gt_mask, image_size, interpolation = cv2.INTER_NEAREST)
    depth_gt_mask = depth_gt_mask.astype(np.bool)

    # depth processing
    depth = process_depth(depth, camera_type = camera_type, depth_min = depth_min, depth_max = depth_max, depth_norm = depth_norm)
    depth_gt = process_depth(depth_gt, camera_type = camera_type, depth_min = depth_min, depth_max = depth_max, depth_norm = depth_norm)

    # RGB augmentation.
    if split == 'train' and use_aug and np.random.rand(1) > 1 - rgb_aug_prob:
        rgb = chromatic_transform(rgb)
        rgb = add_noise(rgb)
    
    # Geometric augmentation
    if split == 'train' and use_aug:
        has_aug = False
        if np.random.rand(1) > 0.5:
            has_aug = True
            rgb = np.flip(rgb, axis = 0)
            depth = np.flip(depth, axis = 0)
            depth_gt = np.flip(depth_gt, axis = 0)
            depth_gt_mask = np.flip(depth_gt_mask, axis = 0)
        if np.random.rand(1) > 0.5:
            has_aug = True
            rgb = np.flip(rgb, axis = 1)
            depth = np.flip(depth, axis = 1)
            depth_gt = np.flip(depth_gt, axis = 1)
            depth_gt_mask = np.flip(depth_gt_mask, axis = 1)
        if has_aug:
            rgb = rgb.copy()
            depth = depth.copy()
            depth_gt = depth_gt.copy()
            depth_gt_mask = depth_gt_mask.copy()

    # RGB normalization
    rgb = rgb / 255.0
    rgb = rgb.transpose(2, 0, 1)

    # process scene mask
    scene_mask = (scene_type == 'cluttered')

    # zero mask
    neg_zero_mask = np.where(depth_gt < 0.01, 255, 0).astype(np.uint8)
    neg_zero_mask_dilated = cv2.dilate(neg_zero_mask, kernel = DILATION_KERNEL)
    neg_zero_mask[neg_zero_mask != 0] = 1
    neg_zero_mask_dilated[neg_zero_mask_dilated != 0] = 1
    zero_mask = np.logical_not(neg_zero_mask)
    zero_mask_dilated = np.logical_not(neg_zero_mask_dilated)

    # loss mask
    initial_loss_mask = np.logical_and(depth_gt_mask, zero_mask)
    initial_loss_mask_dilated = np.logical_and(depth_gt_mask, zero_mask_dilated)
    if scene_mask:
        loss_mask = initial_loss_mask
        loss_mask_dilated = initial_loss_mask_dilated
    else:
        loss_mask = zero_mask
        loss_mask_dilated = zero_mask_dilated

    data_dict = {
        'rgb': torch.FloatTensor(rgb),
        'depth': torch.FloatTensor(depth),
        'depth_gt': torch.FloatTensor(depth_gt),
        'depth_gt_mask': torch.BoolTensor(depth_gt_mask),
        'scene_mask': torch.tensor(scene_mask),
        'zero_mask': torch.BoolTensor(zero_mask),
        'zero_mask_dilated': torch.BoolTensor(zero_mask_dilated),
        'initial_loss_mask': torch.BoolTensor(initial_loss_mask),
        'initial_loss_mask_dilated': torch.BoolTensor(initial_loss_mask_dilated),
        'loss_mask': torch.BoolTensor(loss_mask),
        'loss_mask_dilated': torch.BoolTensor(loss_mask_dilated),
        'depth_original': torch.FloatTensor(depth_original),
        'depth_gt_original': torch.FloatTensor(depth_gt_original),
        'depth_gt_mask_original': torch.BoolTensor(depth_gt_mask_original),
        'zero_mask_original': torch.BoolTensor(zero_mask_original),
        'fx': torch.tensor(camera_intrinsics[0, 0]),
        'fy': torch.tensor(camera_intrinsics[1, 1]),
        'cx': torch.tensor(camera_intrinsics[0, 2]),
        'cy': torch.tensor(camera_intrinsics[1, 2])
    }

    data_dict['depth_gt_sn'] = get_surface_normal_from_depth(data_dict['depth_gt'].unsqueeze(0), data_dict['fx'].unsqueeze(0), data_dict['fy'].unsqueeze(0), data_dict['cx'].unsqueeze(0), data_dict['cy'].unsqueeze(0)).squeeze(0)

    return data_dict
