"""
Omniverse Object Dataset.

Author: Hongjie Fang.

Ref:
    [1] Implicit-depth official website: https://research.nvidia.com/publication/2021-03_RGB-D-Local-Implicit
    [2] Implicit-depth official repository: https://github.com/NVlabs/implicit_depth/
"""
import os
import cv2
import h5py
import torch
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from utils.data_preparation import process_data, add_noise_to_depth


class OmniverseObject(Dataset):
    """
    Omniverse Object dataset.
    """
    def __init__(self, data_dir, split = 'test', **kwargs):
        """
        Initialization.

        Parameters
        ----------

        data_dir: str, required, the data path;
        
        split: str in ['train', 'test'], optional, default: 'test', the dataset split option.
        """
        super(OmniverseObject, self).__init__()
        if split not in ['train', 'test']:
            raise AttributeError('Invalid split option.')
        self.data_dir = data_dir
        self.split = split
        self.split_ratio = kwargs.get('split_ratio', 0.9)
        full_h5_paths = sorted(glob(os.path.join(self.data_dir, '*/*.h5')))
        split_idx = int(len(full_h5_paths) * self.split_ratio)
        if split == 'train':
            self.h5_paths = full_h5_paths[:split_idx]
        else:
            self.h5_paths = full_h5_paths[split_idx:]
        self.use_aug = kwargs.get('use_augmentation', True)
        self.rgb_aug_prob = kwargs.get('rgb_augmentation_probability', 0.8)
        self.image_size = kwargs.get('image_size', (1280, 720))
           
    def get_transparent_mask(self, instance_mask, semantic_mask, instance_num, corrupt_all=False, ratio_low=0.4, ratio_high=0.8):
        """
        Get transparent mask from Omniverse datasets.
        
        This is adapted from implicit-depth repository, ref: https://github.com/NVlabs/implicit_depth/blob/main/src/datasets/omniverse_dataset.py
        """
        rng = np.random.default_rng()
        corrupt_mask = np.zeros((instance_mask.shape[0], instance_mask.shape[1]))
        if self.exp_type == 'train':
            if corrupt_all:
                corrupt_obj_num = instance_num
                corrupt_obj_ids = np.arange(instance_num)
            else:
                # randomly select corrupted objects number
                corrupt_obj_num = rng.choice(np.arange(1,instance_num+1), 1, replace=False)[0]
                # randomly select corrupted objects ids
                corrupt_obj_ids = rng.choice(instance_num, corrupt_obj_num, replace=False)
            for cur_obj_id in corrupt_obj_ids:
                cur_obj_id = cur_obj_id + 1
                nonzero_idx = np.transpose(np.nonzero(instance_mask==cur_obj_id))
                if nonzero_idx.shape[0] == 0:
                    continue
                # transparent objects: select all pixels
                if semantic_mask[nonzero_idx[0,0],nonzero_idx[0,1]] == 2:
                    sampled_nonzero_idx = nonzero_idx
                # opaque objects: select partial pixels.
                else:
                    ratio = np.random.random() * (ratio_high - ratio_low) + ratio_low
                    sample_num = int(nonzero_idx.shape[0] * ratio)
                    sample_start_idx = rng.choice(nonzero_idx.shape[0]-sample_num, 1, replace=False)[0]
                    sampled_nonzero_idx = nonzero_idx[sample_start_idx:sample_start_idx+sample_num]
                corrupt_mask[sampled_nonzero_idx[:,0],sampled_nonzero_idx[:,1]] = 1
        else:
            for cur_obj_id in range(instance_num):
                cur_obj_id += 1
                nonzero_idx = np.transpose(np.nonzero(instance_mask==cur_obj_id))
                if nonzero_idx.shape[0] == 0:
                    continue
                # transparent objects: select all pixels
                if semantic_mask[nonzero_idx[0,0],nonzero_idx[0,1]] == 2:
                    sampled_nonzero_idx = nonzero_idx
                # opaque objects: skip
                else:
                    continue
                corrupt_mask[sampled_nonzero_idx[:,0],sampled_nonzero_idx[:,1]] = 1
        return corrupt_mask
    
        
    def __getitem__(self, id):
        f = h5py.File(self.h5_paths[id], 'r')
        # rgb
        rgb = cv2.cvtColor(f['rgb_glass'][:], cv2.COLOR_RGB2BGR)
        # depth-gt-mask
        instance_seg = f['instance_seg'][:]
        instance_id = np.arange(1, instance_seg.shape[0]+1).reshape(-1, 1, 1)
        instance_mask = np.sum(instance_seg * instance_id,0).astype(np.uint8)
        semantic_seg = f['semantic_seg'][:]
        semantic_id = np.arange(1, semantic_seg.shape[0]+1).reshape(-1, 1, 1)
        semantic_mask = np.sum(semantic_seg * semantic_id,0).astype(np.uint8)
        depth_gt_mask = self.get_transparent_mask(instance_mask, semantic_mask, instance_seg.shape[0], ratio_low = 0.3, ratio_high = 0.7)
        # depth, depth-gt
        disparity = f['depth'][:]
        depth_gt = 1. / (disparity + self.epsilon) * 0.01
        depth_gt = np.clip(depth_gt, 0, 4)
        depth = depth_gt.copy() * (1 - depth_gt_mask.float())
        return process_data(rgb, depth, depth_gt, depth_gt_mask, scene_type = "cluttered", camera_type = 0, split = self.split, image_size = self.image_size, use_aug = self.use_aug, rgb_aug_prob = self.rgb_aug_prob)
    
    def __len__(self):
        return len(self.h5_paths)
