"""
Data preparation, including scaling, augmentation and tensorize.

Authors: Authors from [implicit-depth] repository, Hongjie Fang.
Ref: 
    1. [implicit-depth] repository: https://github.com/NVlabs/implicit_depth
"""
import cv2
import torch
import random
import numpy as np


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


def process_data(rgb, depth, depth_gt, depth_gt_mask, scene_type = "cluttered", camera_type = 1, split = 'train', use_aug = True, aug_prob = 0.8):
    """
    Process images and perform data augmentation.

    Parameters
    ----------
    rgb: array, required, the rgb image;
    depth: array, required, the original depth image;
    depth_gt: array, required, the ground-truth depth image;
    depth_gt_mask: array, required, the ground-truth depth image mask;
    scene_type: str in ['cluttered', 'isolated'], optional, default: 'cluttered', the scene type;
    camera_type: int in [1, 2], optional, default: 1, the camera type;
    split: str in ['train', 'test'], optional, default: 'train', the split of the dataset;
    use_aug: bool, optional, default: True, whether use data augmentation;
    aug_prob: float, optional, default: 0.8, the data augmentation probability (only applies when use_aug is set to True).

    Returns
    -------
    rgb, depth, depth_gt, depth_gt_mask, scene_mask tensors for training and testing.
    """
    # depth scaling
    depth = depth / (1000 if camera_type == 1 else 4000) # depth sensor scaling
    depth_gt = depth_gt / (1000 if camera_type == 1 else 4000) # depth sensor scaling

    # Depth Normalization
    depth = np.where(depth > 10, 1, depth / 10)
    depth_gt = np.where(depth_gt > 10, 1, depth_gt / 10)

    # RGB augmentation.
    if split == 'train' and use_aug and np.random.rand(1) > 1 - aug_prob:
        rgb = chromatic_transform(rgb)
        rgb = add_noise(rgb)

    # RGB normalization
    rgb = rgb / 255.0
    rgb = rgb.transpose(2, 0, 1)

    # process scene mask
    scene_mask = np.array([1 if scene_type == 'cluttered' else 0], dtype = np.bool)
    return torch.FloatTensor(rgb), torch.FloatTensor(depth), torch.FloatTensor(depth_gt), torch.BoolTensor(depth_gt_mask), torch.BoolTensor(scene_mask)
