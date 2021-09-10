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
from utils.functions import to_device
from time import perf_counter


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)
warnings.simplefilter("ignore", UserWarning)

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

criterion = builder.get_criterion()
metrics = builder.get_metrics()


def test():
    logger.info('Start testing process.')
    model.eval()
    metrics.clear()
    running_time = []
    losses = []
    with tqdm(test_dataloader) as pbar:
        for data_dict in pbar:
            data_dict = to_device(data_dict, device)
            with torch.no_grad():
                time_start = perf_counter()
                res = model(data_dict['rgb'], data_dict['depth'])
                time_end = perf_counter()
                data_dict['pred'] = res
                loss_dict = criterion(data_dict)
                loss = loss_dict['loss']
                _ = metrics.evaluate_batch(data_dict, record = True)
            duration = time_end - time_start
            if 'smooth' in loss_dict.keys():
                pbar.set_description('Loss: {:.8f}, smooth loss: {:.8f}'.format(loss.item(), loss_dict['smooth'].item()))
            else:
                pbar.set_description('Loss: {:.8f}'.format(loss.item()))
            losses.append(loss.item())
            running_time.append(duration)
    mean_loss = np.stack(losses).mean()
    avg_running_time = np.stack(running_time).mean()
    logger.info('Finish testing process, mean testing loss: {:.8f}, average running time: {:.4f}s'.format(mean_loss, avg_running_time))
    metrics_result = metrics.get_results()
    metrics.display_results()
    return mean_loss, metrics_result


if __name__ == '__main__':
    test()

