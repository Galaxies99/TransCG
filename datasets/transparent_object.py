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
        self.camera_intrinsics = {}
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
                with open(os.path.join(cur_path, 'data_params.pbtxt')) as f:
                    cx, cy, fx, fy = None, None, None, None
                    for line in f.readlines():
                        if 'cx' in line:
                            cx = float(str.split(line, ':')[1])
                        if 'cy' in line:
                            cy = float(str.split(line, ':')[1])
                        if 'fx' in line:
                            fx = float(str.split(line, ':')[1])
                        if 'fy' in line:
                            fy = float(str.split(line, ':')[1])
                    assert cx is not None and cy is not None and fx is not None and fy is not None
                self.camera_intrinsics[cur_path] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)
        # Integrity double-check
        self.remove_damaged_samples()
        assert len(self.image_paths) == len(self.depth_paths) and len(self.depth_paths) == len(self.depth_gt_paths)
        self.image_size = kwargs.get('image_size', (1280, 720))
        self.depth_min = kwargs.get('depth_min', 0.3)
        self.depth_max = kwargs.get('depth_max', 1.5)
        self.depth_norm = kwargs.get('depth_norm', 1.0)
    
    def remove_damaged_samples(self):
        damaged_list = [
            os.path.join('mug_3', 'texture_8_pose_3', '000010'),
            os.path.join('mug_4', 'texture_4_pose_1', '000013')
        ]
        for item in self.image_paths:
            rm = False
            for d in damaged_list:
                if d in item:
                    rm = True
                    break
            if rm:
                self.image_paths.remove(item)
                self.depth_paths.remove(item.replace('_L.png', '_Dt.exr'))
                self.depth_gt_paths.remove(item.replace('_L.png', '_Do.exr'))
        
    def get_camera_intrinsics(self, id):
        camera_intrinsics = None
        for key in self.camera_intrinsics.keys():
            if key in self.image_paths[id]:
                camera_intrinsics = self.camera_intrinsics[key]
        assert camera_intrinsics is not None
        return camera_intrinsics
    
    def __getitem__(self, id):
        rgba = np.array(Image.open(self.image_paths[id]).convert('RGBA'), dtype = np.float32)
        rgb = rgba[..., :3].copy()
        depth = exr_loader(self.depth_paths[id], ndim = 1, ndim_representation = ['D'])
        depth_gt = exr_loader(self.depth_gt_paths[id], ndim = 1, ndim_representation = ['D'])
        depth_gt_mask = rgba[..., 3].copy() # Not always available.
        depth_gt_mask[depth_gt_mask != 0] = 1
        camera_intrinsics = self.get_camera_intrinsics(id)
        return process_data(rgb, depth, depth_gt, depth_gt_mask, camera_intrinsics, scene_type = "isolated", camera_type = 0, split = self.split, image_size = self.image_size, depth_min = self.depth_min, depth_max = self.depth_max, depth_norm = self.depth_norm, use_aug = False)
    
    def __len__(self):
        return len(self.image_paths)
        