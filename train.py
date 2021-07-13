import os
import yaml
import torch
import logging
import argparse
from tqdm import tqdm
from utils.logger import ColoredLogger
from utils.builder import ConfigBuilder


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)


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
train_dataloader = builder.get_dataloader(split = 'train')
test_dataloader = builder.get_dataloader(split = 'test')

logger.info('Building optimizer and learning rate schedulers ...')
optimizer = builder.get_optimizer(model)
lr_scheduler = builder.get_lr_scheduler(optimizer)

logger.info('Checking checkpoints ...')
start_epoch = 0
max_epoch = builder.get_max_epoch()
stats_dir = builder.get_stats_dir()
if os.path.exists(stats_dir) == False:
    os.makedirs(stats_dir)
checkpoint_file = os.path.join(stats_dir, 'checkpoint.tar')
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    if lr_scheduler is not None:
        lr_scheduler.last_epoch = start_epoch - 1
    logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))


def train_one_epoch(epoch):
    logger.info('Start training process in epoch {}.'.format(epoch + 1))
    model.train()
    losses = []
    with tqdm(train_dataloader) as pbar:
        for data in pbar:
            optimizer.zero_grad()
            x, labels = data
            x = x.to(device)
            labels = labels.to(device)
            res = model(x)
            loss = model.loss(res, labels.view(-1))
            loss.backward()
            optimizer.step()
            pbar.set_description('Epoch {}, loss: {:.8f}'.format(epoch + 1, loss.mean().item()))
            losses.append(loss.mean().item())
    mean_loss = torch.stack(losses).mean()
    logger.info('Finish training process in epoch {}, mean training loss: {:.8f}'.format(epoch + 1, mean_loss))


def test_one_epoch(epoch):
    logger.info('Start testing process in epoch {}.'.format(epoch + 1))
    model.eval()
    losses = []
    with tqdm(test_dataloader) as pbar:
        for data in pbar:
            x, labels = data
            x = x.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                res = model(x)
                loss = model.loss(res, labels.view(-1))
            pbar.set_description('Epoch {}, loss: {:.8f}, accuracy: {:.6f}'.format(epoch + 1, loss.mean().item()))
            losses.append(loss.mean().item())
    mean_loss = torch.stack(losses).mean()
    logger.info('Finish testing process in epoch {}, mean testing loss: {:.8f}.'.format(epoch + 1, mean_loss))
    return mean_loss


def train(start_epoch):
    min_loss = test_one_epoch(start_epoch - 1)
    min_loss_epoch = start_epoch
    for epoch in range(start_epoch, max_epoch):
        logger.info('--> Epoch {}/{}'.format(epoch + 1, max_epoch))
        train_one_epoch(epoch)
        loss = test_one_epoch(epoch)
        if lr_scheduler is not None:
            lr_scheduler.step()
        if loss < min_loss:
            min_loss = loss
            min_loss_epoch = epoch + 1
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
            }
            torch.save(save_dict, os.path.join(stats_dir, 'checkpoint.tar'))
    logger.info('Training Finished. Max accuracy: {:.6f}, in epoch {}'.format(min_loss, min_loss_epoch))


if __name__ == '__main__':
    train(start_epoch = start_epoch)
