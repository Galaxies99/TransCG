"""
Inference.

Authors: Hongjie Fang.
"""
import os
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
    def __init__(self, cfg_path = os.path.join('configs', 'default.yaml'), with_info = False, **kwargs):
        """
        Initialization.
        
        Parameters
        ----------
        cfg_path: str, optional, default: 'configs/default.yaml', the path to the configuration file;
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

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        if self.with_info:
            self.logger.info('Checking checkpoints ...')
        
        stats_dir = self.builder.get_stats_dir()
        checkpoint_file = os.path.join(stats_dir, 'checkpoint.tar')
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            if self.with_info:
                self.logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))
        else:
            raise FileNotFoundError('No checkpoint.')

    def inference(self, rgb, depth):
        """
        Inference.

        Parameters
        ----------
        rgb, depth: the initial RGB-D image;

        Returns
        -------
        The depth image after completion.
        """
        rgb = (rgb / 255.0).transpose(2, 0, 1)
        depth = np.where(depth > 10, 1, depth / 10)
        rgb = torch.FloatTensor(rgb, device = self.device)
        depth = torch.FloatTensor(depth, device = self.device)
        with torch.no_grad():
            time_start = perf_counter()
            depth_res = self.model(rgb, depth)
            time_end = perf_counter()
        if self.with_info:
            self.logger.info("Inference finished, time: {:.4f}s.".format(time_end - time_start))
        depth_res = (depth_res * 10).cpu().detach().numpy() 
        return depth_res
    
