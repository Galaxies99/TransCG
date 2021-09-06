"""
Transparent Object Dataset.

Author: Hongjie Fang.

Ref:
    [1] KeyPose official website: https://sites.google.com/view/keypose
    [2] KeyPose official repository: https://github.com/google-research/google-research/tree/master/keypose
"""
import os
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from utils.data_preparation import process_data, exr_loader


class TransparentObject(Dataset):
    """
    Transparent Object dataset.
    """
    def __init__(self, data_dir, split = 'test', **kwargs):
        """
        Initialization.

        Parameters
        ----------

        data_dir: str, required, the data path;
        
        split: str in ['train', 'test'], optional, default: 'test', the dataset split option.
        """
        super(TransparentObject, self).__init__()
        if split not in ['train', 'test']:
            raise AttributeError('Invalid split option.')
        self.data_dir = data_dir
        self.split = split
        if split == 'train':
            raise AttributeError('Transparent object dataset is not supported for training.')
        self.image_paths = []
        self.depth_paths = []
        self.depth_gt_paths = []
        for second_dir in os.listdir(self.data_dir):
            second_dir_full = os.path.join(self.data_dir, second_dir)
            for third_dir in os.listdir(second_dir_full):
                cur_path = os.path.join(second_dir_full, third_dir)
                cur_image_paths = sorted(glob(os.path.join(cur_path, '*_L.png')))
                cur_depth_paths = [p.replace('_L.png', '_Dt.exr') for p in cur_image_paths]
                cur_depth_gt_paths = [p.replace('_L.png', '_Do.exr') for p in cur_image_paths]
                self.image_paths += cur_image_paths
                self.depth_paths += cur_depth_paths
                self.depth_gt_paths += cur_depth_gt_paths
        # Integrity double-check
        assert len(self.image_paths) == len(self.depth_paths) and len(self.depth_paths) == len(self.depth_gt_paths)
        self.image_size = kwargs.get('image_size', (1280, 720))
    
    def __getitem__(self, id):
        rgba = np.array(Image.open(self.image_paths[id]).convert('RGBA'), dtype = np.float32)
        rgb = rgba[..., :3].copy()
        depth = exr_loader(self.depth_paths[id], ndim = 1, ndim_representation = ['D'])
        depth_gt = exr_loader(self.depth_gt_paths[id], ndim = 1, ndim_representation = ['D'])
        depth_gt_mask = rgba[..., 3].copy() # Not always available.
        depth_gt_mask[depth_gt_mask != 0] = 1
        return process_data(rgb, depth, depth_gt, depth_gt_mask, scene_type = "isolated", camera_type = 0, split = self.split, image_size = self.image_size, use_aug = False)
    
    def __len__(self):
        return len(self.image_paths)
        