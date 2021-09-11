"""
Inference.

Authors: Hongjie Fang.
"""
import os
import cv2
import yaml
import torch
import logging
import warnings
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils.logger import ColoredLogger
from utils.builder import ConfigBuilder
from time import perf_counter


class Inferencer(object):
    """
    Inferencer.
    """
    def __init__(self, cfg_path = os.path.join('configs', 'inference.yaml'), with_info = False, **kwargs):
        """
        Initialization.
        
        Parameters
        ----------

        cfg_path: str, optional, default: 'configs/inference.yaml', the path to the inference configuration file;

        with_info: bool, optional, default: False, whether to display information on the screen.
        """
        warnings.filterwarnings("ignore")
        logging.setLoggerClass(ColoredLogger)
        self.logger = logging.getLogger(__name__)

        with open(cfg_path, 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)

        self.builder = ConfigBuilder(**cfg_params)
        self.with_info = with_info

        if self.with_info:
            self.logger.info('Building models ...')
        
        self.model = self.builder.get_model()
        
        self.cuda_id = self.builder.get_inference_cuda_id()
        self.device = torch.device('cuda:{}'.format(self.cuda_id) if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        if self.with_info:
            self.logger.info('Checking checkpoints ...')
        
        checkpoint_file = self.builder.get_inference_checkpoint_path()
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location = self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            if self.with_info:
                self.logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))
        else:
            raise FileNotFoundError('No checkpoint.')
        
        self.image_size = self.builder.get_inference_image_size()
        self.depth_min, self.depth_max = self.builder.get_inference_depth_min_max()
        self.depth_norm = self.builder.get_inference_depth_norm()

    def inference(self, rgb, depth, target_size = (1280, 720)):
        """
        Inference.

        Parameters
        ----------

        rgb, depth: the initial RGB-D image;

        target_size: tuple of (int, int), optional, default: (1280, 720), the target depth image size.

        Returns
        -------

        The depth image after completion.
        """
        
        rgb = cv2.resize(rgb, self.image_size, interpolation = cv2.INTER_NEAREST)
        depth = cv2.resize(depth, self.image_size, interpolation = cv2.INTER_NEAREST)
        depth = np.where(depth < self.depth_min, 0, depth)
        depth = np.where(depth > self.depth_max, 0, depth)
        depth[np.isnan(depth)] = 0
        depth = depth / self.depth_norm
        rgb = (rgb / 255.0).transpose(2, 0, 1)
        rgb = torch.FloatTensor(rgb).to(self.device).unsqueeze(0)
        depth = torch.FloatTensor(depth).to(self.device).unsqueeze(0)
        with torch.no_grad():
            time_start = perf_counter()
            depth_res = self.model(rgb, depth)
            time_end = perf_counter()
        if self.with_info:
            self.logger.info("Inference finished, time: {:.4f}s.".format(time_end - time_start))
        depth_res = depth_res.squeeze(0).cpu().detach().numpy()
        depth_res = depth_res * self.depth_norm
        depth_res = cv2.resize(depth_res, target_size, interpolation = cv2.INTER_NEAREST)
        return depth_res
    