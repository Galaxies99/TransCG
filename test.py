"""
Testing scripts.

Authors: Hongjie Fang.
"""
import os
import yaml
import torch
import logging
import warnings
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils.logger import ColoredLogger
from utils.builder import ConfigBuilder
from utils.criterion import Metrics
from time import perf_counter


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    '--cfg', '-c', 
    default = os.path.join('configs', 'default.yaml'), 
    help = 'path to the configuration file', 
    type = str
)
args = parser.parse_args();
cfg_filename = args.cfg

with open(cfg_filename, 'r') as cfg_file:
    cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)

builder = ConfigBuilder(**cfg_params)

logger.info('Building models ...')

model = builder.get_model()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

logger.info('Building dataloaders ...')
test_dataloader = builder.get_dataloader(split = 'test')

logger.info('Checking checkpoints ...')
stats_dir = builder.get_stats_dir()
checkpoint_file = os.path.join(stats_dir, 'checkpoint.tar')
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))
else:
    raise FileNotFoundError('No checkpoint.')

criterion = builder.get_loss()
metrics = Metrics()


def test():
    logger.info('Start testing process.')
    model.eval()
    metrics.clear()
    losses = []
    with tqdm(test_dataloader) as pbar:
        for data in pbar:
            rgb, depth, depth_gt, depth_gt_mask, scene_mask = data
            rgb = rgb.to(device)
            depth = depth.to(device)
            depth_gt = depth_gt.to(device)
            depth_gt_mask = depth_gt_mask.to(device)
            scene_mask = scene_mask.to(device)
            with torch.no_grad():
                time_start = perf_counter()
                res = model(rgb, depth)
                time_end = perf_counter()
                loss = criterion(res, depth_gt, depth_gt_mask, scene_mask)
                metrics.add_record(res, depth_gt, depth_gt_mask, scene_mask)
            pbar.set_description('Test loss: {:.8f}, model time: {:.4f}s'.format(loss.mean().item(), time_end - time_start))
            losses.append(loss.mean().item())
    mean_loss = np.stack(losses).mean()
    logger.info('Finish testing process, mean testing loss: {:.8f}.'.format(mean_loss))
    metrics_result = metrics.final()
    logger.info('Metrics: ')
    logger.info('MSE (w/o mask): {:.6f},    {:.6f}'.format(metrics_result[1], metrics_result[0]))
    logger.info('RMSE (w/o mask): {:.6f},    {:.6f}'.format(metrics_result[3], metrics_result[2]))
    logger.info('REL (w/o mask): {:.6f},    {:.6f}'.format(metrics_result[5], metrics_result[4]))
    logger.info('MAE (w/o mask): {:.6f},    {:.6f}'.format(metrics_result[7], metrics_result[6]))
    logger.info('Threshold 1.05 (w/o mask): {:.6f},    {:.6f}'.format(metrics_result[9], metrics_result[8]))
    logger.info('Threshold 1.10 (w/o mask): {:.6f},    {:.6f}'.format(metrics_result[11], metrics_result[10]))
    logger.info('Threshold 1.25 (w/o mask): {:.6f},    {:.6f}'.format(metrics_result[13], metrics_result[12]))
    return mean_loss, list(metrics_result)


if __name__ == '__main__':
    test()

