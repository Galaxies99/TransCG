"""
ClearGrasp Dataset.

Author: Hongjie Fang.

Ref:
    [1] ClearGrasp official website: https://sites.google.com/view/cleargrasp
    [2] ClearGrasp official repository: https://github.com/Shreeyak/cleargrasp
"""
import os
import yaml
import json
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from utils.data_preparation import process_data, exr_loader


class ClearGraspRealWorld(Dataset):
    """
    ClearGrasp real-world dataset.
    """
    def __init__(self, data_dir, split = 'test', **kwargs):
        """
        Initialization.

        Parameters
        ----------

        data_dir: str, required, the data path;
        
        split: str in ['train', 'test'], optional, default: 'test', the dataset split option.
        """
        super(ClearGraspRealWorld, self).__init__()
        if split not in ['train', 'test']:
            raise AttributeError('Invalid split option.')
        self.data_dir = data_dir
        self.split = split
        if split == 'train':
            raise AttributeError('Cleargrasp real-world dataset is not supported for training.')
        self.image_paths = []
        self.depth_paths = []
        self.mask_paths = []
        self.depth_gt_paths = []
        self.camera_intrinsics = {}
        for data_type in ['real-test', 'real-val']:
            for camera_type in ['d415', 'd435']:
                cur_path = os.path.join(data_dir, 'cleargrasp-dataset-test-val', data_type, camera_type)
                if not os.path.exists(cur_path):
                    continue
                cur_image_paths = sorted(glob(os.path.join(cur_path, '*-transparent-rgb-img.jpg')))
                cur_mask_paths = [p.replace('-transparent-rgb-img.jpg', '-mask.png') for p in cur_image_paths]
                cur_depth_paths = [p.replace('-transparent-rgb-img.jpg', '-transparent-depth-img.exr') for p in cur_image_paths]
                cur_depth_gt_paths = [p.replace('-transparent-rgb-img.jpg', '-opaque-depth-img.exr') for p in cur_image_paths]
                self.image_paths += cur_image_paths
                self.mask_paths += cur_mask_paths
                self.depth_paths += cur_depth_paths
                self.depth_gt_paths += cur_depth_gt_paths
                with open(os.path.join(cur_path, 'camera_intrinsics.yaml')) as cam_intrinsics_file:
                    cam_intrinsics = yaml.load(cam_intrinsics_file, Loader = yaml.FullLoader)
                    fx = cam_intrinsics['fx']
                    fy = cam_intrinsics['fy']
                    cx = cam_intrinsics['cx']
                    cy = cam_intrinsics['cy']
                    self.camera_intrinsics[cur_path] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)
        # Integrity double-check
        assert len(self.image_paths) == len(self.mask_paths) and len(self.mask_paths) == len(self.depth_paths) and len(self.depth_paths) == len(self.depth_gt_paths)
        self.image_size = kwargs.get('image_size', (1280, 720))
        self.depth_min = kwargs.get('depth_min', 0.3)
        self.depth_max = kwargs.get('depth_max', 1.5)
        self.depth_norm = kwargs.get('depth_norm', 1.0)
        self.with_original = kwargs.get('with_original', False)
    
    def get_camera_intrinsics(self, id):
        camera_intrinsics = None
        for key in self.camera_intrinsics.keys():
            if key in self.image_paths[id]:
                camera_intrinsics = self.camera_intrinsics[key]
        assert camera_intrinsics is not None
        return camera_intrinsics
    
    def __getitem__(self, id):
        rgb = np.array(Image.open(self.image_paths[id]), dtype = np.float32)
        depth = exr_loader(self.depth_paths[id], ndim = 1, ndim_representation = ['R'])
        depth_gt = exr_loader(self.depth_gt_paths[id], ndim = 1, ndim_representation = ['R'])
        depth_gt_mask = np.array(Image.open(self.mask_paths[id]), dtype = np.uint8)
        depth_gt_mask[depth_gt_mask != 0] = 1
        camera_intrinsics = self.get_camera_intrinsics(id)
        return process_data(rgb, depth, depth_gt, depth_gt_mask, camera_intrinsics, scene_type = "isolated", camera_type = 0, split = self.split, image_size = self.image_size, depth_min = self.depth_min, depth_max = self.depth_max, depth_norm = self.depth_norm, use_aug = False, with_original = self.with_original)

    def __len__(self):
        return len(self.image_paths)


class ClearGraspSynthetic(Dataset):
    """
    ClearGrasp synthetic dataset.
    """
    def __init__(self, data_dir, split = 'train', **kwargs):
        super(ClearGraspSynthetic, self).__init__()
        if split not in ['train', 'test']:
            raise AttributeError('Invalid split option.')
        self.data_dir = data_dir
        self.split = split
        self.image_paths = []
        self.mask_paths = []
        self.depth_gt_paths = []
        self.json_paths = []
        if split == 'test':
            secondary_dir = 'cleargrasp-dataset-test-val'
            data_type_list = ['synthetic-test', 'synthetic-val']
        else:
            secondary_dir = 'cleargrasp-dataset-train'
            data_type_list = ['.']
        for data_type in data_type_list:
            cur_path_before_scene = os.path.join(self.data_dir, secondary_dir, data_type)
            for scene in os.listdir(cur_path_before_scene):
                cur_path = os.path.join(cur_path_before_scene, scene)
                cur_image_paths = sorted(glob(os.path.join(cur_path, 'rgb-imgs', '*-rgb.jpg')))
                cur_mask_paths = [p.replace('rgb-imgs', 'segmentation-masks').replace('-rgb.jpg', '-segmentation-mask.png') for p in cur_image_paths]
                cur_depth_gt_paths = [p.replace('rgb-imgs', 'depth-imgs-rectified').replace('-rgb.jpg', '-depth-rectified.exr') for p in cur_image_paths]
                cur_json_paths = [p.replace('rgb-imgs', 'json-files').replace('-rgb.jpg', '-masks.json') for p in cur_image_paths]
                self.image_paths += cur_image_paths
                self.mask_paths += cur_mask_paths
                self.depth_gt_paths += cur_depth_gt_paths
                self.json_paths += cur_json_paths
        # Integrity double-check
        assert len(self.image_paths) == len(self.mask_paths) and len(self.mask_paths) == len(self.depth_gt_paths) and len(self.json_paths) == len(self.depth_gt_paths)
        self.image_size = kwargs.get('image_size', (1280, 720))
        self.use_aug = kwargs.get('use_augmentation', True)
        self.rgb_aug_prob = kwargs.get('rgb_augmentation_probability', 0.8)
        self.depth_min = kwargs.get('depth_min', 0.3)
        self.depth_max = kwargs.get('depth_max', 1.5)
        self.depth_norm = kwargs.get('depth_norm', 1.0)
        self.with_original = kwargs.get('with_original', False)

    def get_camera_intrinsics(self, id, img_size):
        with open(self.json_paths[id], 'r') as fp:
            meta_data = json.load(fp)
        if 'camera' not in meta_data.keys() or 'field_of_view' not in meta_data['camera'].keys():
            fov_x = 1.2112585306167603
            fov_y = 0.7428327202796936
        else:
            fov_x = meta_data['camera']['field_of_view']['x_axis_rads']
            fov_y = meta_data['camera']['field_of_view']['y_axis_rads']
        if 'image' not in meta_data.keys():
            img_h = img_size[0]
            img_w = img_size[1]
        else:
            img_h = meta_data['image']['height_px']
            img_w = meta_data['image']['width_px']
        fx = img_w * 0.5 / np.tan(fov_x * 0.5)
        fy = img_h * 0.5 / np.tan(fov_y * 0.5)
        cx = img_w * 0.5
        cy = img_h * 0.5
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)

    def __getitem__(self, id):
        rgb = np.array(Image.open(self.image_paths[id]), dtype = np.float32)
        depth_gt = exr_loader(self.depth_gt_paths[id], ndim = 1, ndim_representation = ['R'])
        depth_gt_mask = np.array(Image.open(self.mask_paths[id]), dtype = np.uint8)
        depth_gt_mask[depth_gt_mask != 0] = 1
        depth = depth_gt.copy() * (1 - depth_gt_mask)
        depth_gt_mask = depth_gt_mask.astype(np.uint8)
        camera_intrinsics = self.get_camera_intrinsics(id, (rgb.shape[0], rgb.shape[1]))
        return process_data(rgb, depth, depth_gt, depth_gt_mask, camera_intrinsics, scene_type = "isolated", camera_type = 0, split = self.split, image_size = self.image_size, depth_min = self.depth_min, depth_max = self.depth_max, depth_norm = self.depth_norm, use_aug = self.use_aug, rgb_aug_prob = self.rgb_aug_prob, with_original = self.with_original)

    def __len__(self):
        return len(self.image_paths)