import os
import json
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset


class TransparentDataset(Dataset):
    def __init__(self, data_dir, split, **kwargs):
        super(TransparentDataset, self).__init__()
        if split not in ['train', 'test']:
            raise AttributeError('Invalid split option.')
        self.data_dir = data_dir
        metadata_file = kwargs.get('metadata', os.path.join(self.data_dir, 'metadata.json'))
        self.metadata = json.load(metadata_file)
        self.perspective_per_scene = self.metadata.get('perspective_per_scene', 240)
        self.scene_num = self.metadata.get('scene_num', 100)

    def __getitem__(self, id):
        scene_id = int(id // self.perspective_per_scene) + 1
        perspective_id = id - (scene_id - 1) * self.perspective_per_scene
        base_path = os.path.join(self.data_dir, 'scene{}'.format(scene_id), '{}'.format(perspective_id))
        rgb = np.array(Image.open(os.path.join(base_path, 'rgb1.png'), dtype = np.float32)) / 255.0
        depth = np.array(Image.open(os.path.join(base_path, 'depth1-gt.png')))
        return rgb, depth
    
    def __len__(self):
        return self.perspective_per_scene * self.scene_num

