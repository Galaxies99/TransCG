"""
Transparent Grasp Dataset.

Author: Hongjie Fang.
"""
import os
import json
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
from utils.data_preparation import process_data


class TransparentGrasp(Dataset):
    """
    Transparent Grasp dataset.
    """
    def __init__(self, data_dir, split = 'train', **kwargs):
        """
        Initialization.

        Parameters
        ----------

        data_dir: str, required, the data path;
        
        split: str in ['train', 'test'], optional, default: 'train', the dataset split option.
        """
        super(TransparentGrasp, self).__init__()
        if split not in ['train', 'test']:
            raise AttributeError('Invalid split option.')
        self.data_dir = data_dir
        self.split = split
        self.high_resolution = kwargs.get('high_resolution', False)
        if self.high_resolution and split == 'train':
            raise AttributeError('Does not support returning high resolution images during training. If you want to train on high resolution samples, please set image_size arguments in high resolution.')
        with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as fp:
            self.dataset_metadata = json.load(fp)
        self.scene_num = self.dataset_metadata['total_scenes']
        self.perspective_num = self.dataset_metadata['perspective_num']
        self.scene_metadata = [None]
        for scene_id in range(1, self.scene_num + 1):
            with open(os.path.join(self.data_dir, 'scene{}'.format(scene_id), 'metadata.json'), 'r') as fp:
                self.scene_metadata.append(json.load(fp))
        self.total_samples = self.dataset_metadata['{}_samples'.format(split)]
        self.sample_info = []
        for scene_id in self.dataset_metadata[split]:
            scene_type = self.scene_metadata[scene_id]['type']
            scene_split = self.scene_metadata[scene_id]['split']
            assert scene_split == split, "Error in scene {}, expect split property: {}, found split property: {}.".format(scene_id, split, scene_split)
            for perspective_id in self.scene_metadata[scene_id]['D435_valid_perspective_list']:
                self.sample_info.append([
                    os.path.join(self.data_dir, 'scene{}'.format(scene_id), '{}'.format(perspective_id)),
                    1, # (for D435)
                    scene_type
                ])
            for perspective_id in self.scene_metadata[scene_id]['L515_valid_perspective_list']:
                self.sample_info.append([
                    os.path.join(self.data_dir, 'scene{}'.format(scene_id), '{}'.format(perspective_id)),
                    2, # (for L515)
                    scene_type
                ])
        # Integrity double-check
        assert len(self.sample_info) == self.total_samples, "Error in total samples, expect {} samples, found {} samples.".format(self.total_samples, len(self.sample_info))
        self.use_aug = kwargs.get('use_augmentation', True)
        self.rgb_aug_prob = kwargs.get('rgb_augmentation_probability', 0.8)
        self.image_size = kwargs.get('image_size', (1280, 720))
        self.depth_min = kwargs.get('depth_min', 0.0)
        self.depth_max = kwargs.get('depth_max', 10.0)

    def __getitem__(self, id):
        img_path, camera_type, scene_type = self.sample_info[id]
        rgb = np.array(Image.open(os.path.join(img_path, 'rgb{}.png'.format(camera_type))), dtype = np.float32)
        depth = np.array(Image.open(os.path.join(img_path, 'depth{}.png'.format(camera_type))), dtype = np.float32)
        depth_gt = np.array(Image.open(os.path.join(img_path, 'depth{}-gt.png'.format(camera_type))), dtype = np.float32)
        depth_gt_mask = np.array(Image.open(os.path.join(img_path, 'depth{}-gt-mask.png'.format(camera_type))), dtype = np.uint8)
        return process_data(rgb, depth, depth_gt, depth_gt_mask, scene_type, camera_type, split = self.split, image_size = self.image_size, depth_min = self.depth_min, depth_max = self.depth_max, use_aug = self.use_aug, rgb_aug_prob = self.rgb_aug_prob, retain_original = self.high_resolution)
    
    def __len__(self):
        return self.total_samples
