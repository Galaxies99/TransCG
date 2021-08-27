"""
Training scripts.

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
from utils.criterion import MaskedTransparentLoss, Metrics
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

if builder.multigpu():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
    model.to(device)

logger.info('Building dataloaders ...')
train_dataloader = builder.get_dataloader(split = 'train')
test_dataloader = builder.get_dataloader(split = 'test')

logger.info('Building optimizer and learning rate schedulers ...')
optimizer = builder.get_optimizer(model)
lr_scheduler = builder.get_lr_scheduler(optimizer)

logger.info('Checking checkpoints ...')
start_epoch = 0
max_epoch = builder.get_max_epoch()
stats_dir = builder.get_stats_dir()
checkpoint_file = os.path.join(stats_dir, 'checkpoint.tar')
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    if lr_scheduler is not None:
        lr_scheduler.last_epoch = start_epoch - 1
    logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))

if builder.multigpu():
    model = nn.DataParallel(model)

criterion = MaskedTransparentLoss()
metrics = Metrics()


def train_one_epoch(epoch):
    logger.info('Start training process in epoch {}.'.format(epoch + 1))
    model.train()
    losses = []
    with tqdm(train_dataloader) as pbar:
        for data in pbar:
            optimizer.zero_grad()
            rgb, depth, depth_gt, depth_gt_mask, scene_mask = data
            rgb = rgb.to(device)
            depth = depth.to(device)
            depth_gt = depth_gt.to(device)
            depth_gt_mask = depth_gt_mask.to(device)
            scene_mask = scene_mask.to(device)
            res = model(rgb, depth)
            loss = criterion(res, depth_gt, depth_gt_mask, scene_mask)
            loss.backward()
            optimizer.step()
            pbar.set_description('Epoch {}, loss: {:.8f}'.format(epoch + 1, loss.mean().item()))
            losses.append(loss.mean().item())
    mean_loss = np.stack(losses).mean()
    logger.info('Finish training process in epoch {}, mean training loss: {:.8f}'.format(epoch + 1, mean_loss))


def test_one_epoch(epoch):
    logger.info('Start testing process in epoch {}.'.format(epoch + 1))
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
            pbar.set_description('Epoch {}, loss: {:.8f}, model time: {:.4f}'.format(epoch + 1, loss.mean().item(), time_end - time_start))
            losses.append(loss.mean().item())
    mean_loss = np.stack(losses).mean()
    logger.info('Finish testing process in epoch {}, mean testing loss: {:.8f}.'.format(epoch + 1, mean_loss))
    metrics_result = metrics.final()
    logger.info('Metrics: ')
    logger.info('MSE (w/o mask): {:.6f},    {:.6f}'.format(metrics_result[1], metrics_result[0]))
    logger.info('RMSE (w/o mask): {:.6f},    {:.6f}'.format(metrics_result[3], metrics_result[2]))
    logger.info('REL (w/o mask): {:.6f},    {:.6f}'.format(metrics_result[5], metrics_result[4]))
    logger.info('MAE (w/o mask): {:.6f},    {:.6f}'.format(metrics_result[7], metrics_result[6]))
    logger.info('Threshold 1.05 (w/o mask): {:.6f},    {:.6f}'.format(metrics_result[9], metrics_result[8]))
    logger.info('Threshold 1.10 (w/o mask): {:.6f},    {:.6f}'.format(metrics_result[11], metrics_result[10]))
    logger.info('Threshold 1.25 (w/o mask): {:.6f},    {:.6f}'.format(metrics_result[13], metrics_result[12]))
    return mean_loss, metrics_result


def train(start_epoch):
    min_loss = test_one_epoch(start_epoch - 1)
    min_loss_epoch = start_epoch
    for epoch in range(start_epoch, max_epoch):
        logger.info('--> Epoch {}/{}'.format(epoch + 1, max_epoch))
        train_one_epoch(epoch)
        loss, metrics_result = test_one_epoch(epoch)
        if lr_scheduler is not None:
            lr_scheduler.step()
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict() if builder.multigpu() else model.state_dict(),
            'mean_loss': loss,
            'metrics': list(metrics_result)
        }
        torch.save(save_dict, os.path.join(stats_dir, 'checkpoint-ep{}.tar'.format(epoch)))
        if loss < min_loss:
            min_loss = loss
            min_loss_epoch = epoch + 1
            torch.save(save_dict, os.path.join(stats_dir, 'checkpoint.tar'.format(epoch)))
    logger.info('Training Finished. Min testing loss: {:.6f}, in epoch {}'.format(min_loss, min_loss_epoch))


if __name__ == '__main__':
    train(start_epoch = start_epoch)
