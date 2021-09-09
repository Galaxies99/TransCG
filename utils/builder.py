"""
Configuration builder.

Authors: Hongjie Fang.
"""
import os
import logging
from utils.logger import ColoredLogger
from torch.utils.data import ConcatDataset


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)


class ConfigBuilder(object):
    """
    Configuration Builder.

    Features includes:
        
        - build model from configuration;
        
        - build optimizer from configuration;
        
        - build learning rate scheduler from configuration;
        
        - build dataset & dataloader from configuration;
        
        - build statistics directory from configuration;
        
        - build criterion from configuration;

        - build metrics from configuration;
        
        - fetch training parameters (e.g., max_epoch, multigpu) from configuration.
    """
    def __init__(self, **params):
        """
        Set the default configuration for the configuration builder.

        Parameters
        ----------
        
        params: the configuration parameters.
        """
        super(ConfigBuilder, self).__init__()
        self.params = params
        self.model_params = params.get('model', {})
        self.optimizer_params = params.get('optimizer', {})
        self.lr_scheduler_params = params.get('lr_scheduler', {})
        self.dataset_params = params.get('dataset', {'data_dir': 'data'})
        self.dataloader_params = params.get('dataloader', {})
        self.trainer_params = params.get('trainer', {})
        self.metrics_params = params.get('metrics', {})
        self.stats_params = params.get('stats', {})
        self.inference_params = params.get('inference', {})
    
    def get_model(self, model_params = None):
        """
        Get the model from configuration.

        Parameters
        ----------
        
        model_params: dict, optional, default: None. If model_params is provided, then use the parameters specified in the model_params to build the model. Otherwise, the model parameters in the self.params will be used to build the model.
        
        Returns
        -------
        
        A model, which is usually a torch.nn.Module object.
        """
        from models.DFNet import DFNet
        if model_params is None:
            model_params = self.model_params
        type = model_params.get('type', 'DFNet')
        params = model_params.get('params', {})
        if type == 'DFNet':
            model = DFNet(**params)
        else:
            raise NotImplementedError('Invalid model type.')
        return model
    
    def get_optimizer(self, model, optimizer_params = None, resume = False, resume_lr = None):
        """
        Get the optimizer from configuration.
        
        Parameters
        ----------
        
        model: a torch.nn.Module object, the model.
        
        optimizer_params: dict, optional, default: None. If optimizer_params is provided, then use the parameters specified in the optimizer_params to build the optimizer. Otherwise, the optimizer parameters in the self.params will be used to build the optimizer;
        
        resume: bool, optional, default: False, whether to resume training from an existing checkpoint;

        resume_lr: float, optional, default: None, the resume learning rate.
        
        Returns
        -------
        
        An optimizer for the given model.
        """
        from torch.optim import SGD, ASGD, Adagrad, Adamax, Adadelta, Adam, AdamW, RMSprop
        if optimizer_params is None:
            optimizer_params = self.optimizer_params
        type = optimizer_params.get('type', 'AdamW')
        params = optimizer_params.get('params', {})
        if resume:
            network_params = [{'params': model.parameters(), 'initial_lr': resume_lr}]
            params.update(lr = resume_lr)
        else:
            network_params = model.parameters()
        if type == 'SGD':
            optimizer = SGD(network_params, **params)
        elif type == 'ASGD':
            optimizer = ASGD(network_params, **params)
        elif type == 'Adagrad':
            optimizer = Adagrad(network_params, **params)
        elif type == 'Adamax':
            optimizer = Adamax(network_params, **params)
        elif type == 'Adadelta':
            optimizer = Adadelta(network_params, **params)
        elif type == 'Adam':
            optimizer = Adam(network_params, **params)
        elif type == 'AdamW':
            optimizer = AdamW(network_params, **params)
        elif type == 'RMSprop':
            optimizer = RMSprop(network_params, **params)
        else:
            raise NotImplementedError('Invalid optimizer type.')
        return optimizer
    
    def get_lr_scheduler(self, optimizer, lr_scheduler_params = None, resume = False, resume_epoch = None):
        """
        Get the learning rate scheduler from configuration.
        
        Parameters
        ----------
        
        optimizer: an optimizer;
        
        lr_scheduler_params: dict, optional, default: None. If lr_scheduler_params is provided, then use the parameters specified in the lr_scheduler_params to build the learning rate scheduler. Otherwise, the learning rate scheduler parameters in the self.params will be used to build the learning rate scheduler;

        resume: bool, optional, default: False, whether to resume training from an existing checkpoint;

        resume_epoch: int, optional, default: None, the epoch of the checkpoint.
        
        Returns
        -------

        A learning rate scheduler for the given optimizer.
        """
        from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CyclicLR, CosineAnnealingLR, LambdaLR, StepLR
        if lr_scheduler_params is None:
            lr_scheduler_params = self.lr_scheduler_params
        type = lr_scheduler_params.get('type', '')
        params = lr_scheduler_params.get('params', {})
        if resume:
            params.update(last_epoch = resume_epoch)
        if type == 'MultiStepLR':
            scheduler = MultiStepLR(optimizer, **params)
        elif type == 'ExponentialLR':
            scheduler = ExponentialLR(optimizer, **params)
        elif type == 'CyclicLR':
            scheduler = CyclicLR(optimizer, **params)
        elif type == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, **params)
        elif type == 'LambdaLR':
            scheduler = LambdaLR(optimizer, **params)
        elif type == 'StepLR':
            scheduler = StepLR(optimizer, **params)
        elif type == '':
            scheduler = None
        else:
            raise NotImplementedError('Invalid learning rate scheduler type.')
        return scheduler
    
    def get_dataset(self, dataset_params = None, split = 'train'):
        """
        Get the dataset from configuration.

        Parameters
        ----------
        
        dataset_params: dict, optional, default: None. If dataset_params is provided, then use the parameters specified in the dataset_params to build the dataset. Otherwise, the dataset parameters in the self.params will be used to build the dataset;
        
        split: str in ['train', 'test'], optional, default: 'train', the splitted dataset.

        Returns
        -------
        
        A torch.utils.data.Dataset item.
        """
        from datasets.transcg import TransCG
        from datasets.cleargrasp import ClearGraspRealWorld, ClearGraspSynthetic
        from datasets.omniverse_object import OmniverseObject
        from datasets.transparent_object import TransparentObject
        if dataset_params is None:
            dataset_params = self.dataset_params
        dataset_params = dataset_params.get(split, {'type': 'transcg'})
        if type(dataset_params) == dict:
            dataset_type = str.lower(dataset_params.get('type', 'transcg'))
            if dataset_type == 'transcg':
                dataset = TransCG(split = split, **dataset_params)
            elif dataset_type == 'cleargrasp-real':
                dataset = ClearGraspRealWorld(split = split, **dataset_params)
            elif dataset_type == 'cleargrasp-syn':
                dataset = ClearGraspSynthetic(split = split, **dataset_params)
            elif dataset_type == 'omniverse':
                dataset = OmniverseObject(split = split, **dataset_params)
            elif dataset_type == 'transparent-object':
                dataset = TransparentObject(split = split, **dataset_params)
            else:
                raise NotImplementedError('Invalid dataset type: {}.'.format(dataset_type))
            logger.info('Load {} dataset as {}ing set with {} samples.'.format(dataset_type, split, len(dataset)))
        elif type(dataset_params) == list:
            dataset_types = []
            dataset_list = []
            for single_dataset_params in dataset_params:
                dataset_type = str.lower(single_dataset_params.get('type', 'transcg'))
                if dataset_type in dataset_types:
                    raise AttributeError('Duplicate dataset found.')
                else:
                    dataset_types.append(dataset_type)
                if dataset_type == 'transcg':
                    dataset = TransCG(split = split, **single_dataset_params)
                elif dataset_type == 'cleargrasp-real':
                    dataset = ClearGraspRealWorld(split = split, **single_dataset_params)
                elif dataset_type == 'cleargrasp-syn':
                    dataset = ClearGraspSynthetic(split = split, **single_dataset_params)
                elif dataset_type == 'omniverse':
                    dataset = OmniverseObject(split = split, **single_dataset_params)
                elif dataset_type == 'transparent-object':
                    dataset = TransparentObject(split = split, **single_dataset_params)
                else:
                    raise NotImplementedError('Invalid dataset type: {}.'.format(dataset_type))
                dataset_list.append(dataset)
                logger.info('Load {} dataset as {}ing set with {} samples.'.format(dataset_type, split, len(dataset)))
            dataset = ConcatDataset(dataset_list)
        else:
            raise AttributeError('Invalid dataset format.')
        return dataset
    
    def get_dataloader(self, dataset_params = None, split = 'train', batch_size = None, dataloader_params = None):
        """
        Get the dataloader from configuration.

        Parameters
        ----------
        
        dataset_params: dict, optional, default: None. If dataset_params is provided, then use the parameters specified in the dataset_params to build the dataset. Otherwise, the dataset parameters in the self.params will be used to build the dataset;
        
        split: str in ['train', 'test'], optional, default: 'train', the splitted dataset;
        
        batch_size: int, optional, default: None. If batch_size is None, then the batch size parameter in the self.params will be used to represent the batch size (If still not specified, default: 4);
        
        dataloader_params: dict, optional, default: None. If dataloader_params is provided, then use the parameters specified in the dataloader_params to get the dataloader. Otherwise, the dataloader parameters in the self.params will be used to get the dataloader.

        Returns
        -------
        
        A torch.utils.data.DataLoader item.
        """
        from torch.utils.data import DataLoader
        if batch_size is None:
            if split == 'train':
                batch_size = self.trainer_params.get('batch_size', 4)
            else:
                batch_size = self.trainer_params.get('test_batch_size', 1)
        if dataloader_params is None:
            dataloader_params = self.dataloader_params
        dataset = self.get_dataset(dataset_params, split)
        return DataLoader(
            dataset,
            batch_size = batch_size,
            **dataloader_params
        )

    def get_max_epoch(self, trainer_params = None):
        """
        Get the max epoch from configuration.

        Parameters
        ----------
        
        trainer_params: dict, optional, default: None. If trainer_params is provided, then use the parameters specified in the trainer_params to get the maximum epoch. Otherwise, the trainer parameters in the self.params will be used to get the maximum epoch.

        Returns
        -------
        
        An integer, which is the max epoch (default: 50).
        """
        if trainer_params is None:
            trainer_params = self.trainer_params
        return trainer_params.get('max_epoch', 50)
    
    def get_stats_dir(self, stats_params = None):
        """
        Get the statistics directory from configuration.

        Parameters
        ----------
        
        stats_params: dict, optional, default: None. If stats_params is provided, then use the parameters specified in the stats_params to get the statistics directory. Otherwise, the statistics parameters in the self.params will be used to get the statistics directory.

        Returns
        -------
        
        A string, the statistics directory.
        """
        if stats_params is None:
            stats_params = self.stats_params
        stats_dir = stats_params.get('stats_dir', 'stats')
        stats_exper = stats_params.get('stats_exper', 'default')
        stats_res_dir = os.path.join(stats_dir, stats_exper)
        if os.path.exists(stats_res_dir) == False:
            os.makedirs(stats_res_dir)
        return stats_res_dir
    
    def multigpu(self, trainer_params = None):
        """
        Get the multigpu settings from configuration.

        Parameters
        ----------

        trainer_params: dict, optional, default: None. If trainer_params is provided, then use the parameters specified in the trainer_params to get the multigpu flag. Otherwise, the trainer parameters in the self.params will be used to get the multigpu flag.

        Returns
        -------

        A boolean value, whether to use the multigpu training/testing (default: False).
        """
        if trainer_params is None:
            trainer_params = self.trainer_params
        return trainer_params.get('multigpu', False)
    
    def get_resume_lr(self, trainer_params = None):
        """
        Get the resume learning rate from configuration.

        Parameters
        ----------

        trainer_params: dict, optional, default: None. If trainer_params is provided, then use the parameters specified in the trainer_params to get the multigpu flag. Otherwise, the trainer parameters in the self.params will be used to get the multigpu flag.

        Returns
        -------

        A float value, the resume lr (default: 0.001).
        """
        if trainer_params is None:
            trainer_params = self.trainer_params
        return trainer_params.get('resume_lr', False)

    def get_criterion(self, criterion_params = None):
        """
        Get the criterion settings from configuration.

        Parameters
        ----------

        criterion_params: dict, optional, default: None. If criterion_params is provided, then use the parameters specified in the criterion_params to get the criterion. Otherwise, the criterion parameters in the self.params will be used to get the criterion.

        Returns
        -------

        A torch.nn.Module object, the criterion.
        """
        if criterion_params is None:
            criterion_params = self.trainer_params.get('criterion', {})
        loss_type = criterion_params.get('type', 'custom_masked_mse_loss')
        from utils.criterion import Criterion
        criterion = Criterion(**criterion_params)
        return criterion
    
    def get_metrics(self, metrics_params = None):
        """
        Get the metrics settings from configuration.

        Parameters
        ----------

        metrics_params: dict, optional, default: None. If metrics_params is provided, then use the parameters specified in the metrics_params to get the metrics. Otherwise, the metrics parameters in the self.params will be used to get the metrics.
        
        Returns
        -------

        A MetricsRecorder object.
        """
        if metrics_params is None:
            metrics_params = self.metrics_params
        metrics_list = metrics_params.get('types', ['MSE', 'MaskedMSE', 'RMSE', 'MaskedRMSE', 'REL', 'MaskedREL', 'MAE', 'MaskedMAE', 'Threshold@1.05', 'MaskedThreshold@1.05', 'Threshold@1.10', 'MaskedThreshold@1.10', 'Threshold@1.25', 'MaskedThreshold@1.25'])
        metrics_epsilon = metrics_params.get('epsilon', 1e-8)
        from utils.metrics import MetricsRecorder
        metrics = MetricsRecorder(metrics_list = metrics_list, epsilon = metrics_epsilon)
        return metrics
    
    def get_inference_image_size(self, inference_params = None):
        """
        Get the inference image size from inference configuration.

        Parameters
        ----------

        inference_params: dict, optional, default: None. If inference_params is provided, then use the parameters specified in the inference_params to get the inference image size. Otherwise, the inference parameters in the self.params will be used to get the inference image size.
        
        Returns
        -------

        Tuple of (int, int), the image size.
        """
        if inference_params is None:
            inference_params = self.inference_params
        return inference_params.get('image_size', (320, 240))
    
    def get_inference_checkpoint_path(self, inference_params = None):
        """
        Get the inference checkpoint path from inference configuration.

        Parameters
        ----------

        inference_params: dict, optional, default: None. If inference_params is provided, then use the parameters specified in the inference_params to get the inference checkpoint path. Otherwise, the inference parameters in the self.params will be used to get the inference checkpoint path.
        
        Returns
        -------

        str, the checkpoint path.
        """
        if inference_params is None:
            inference_params = self.inference_params
        return inference_params.get('checkpoint_path', os.path.join('checkpoint', 'checkpoint.tar'))
    
    def get_inference_cuda_id(self, inference_params = None):
        """
        Get the inference CUDA ID from inference configuration.

        Parameters
        ----------

        inference_params: dict, optional, default: None. If inference_params is provided, then use the parameters specified in the inference_params to get the inference CUDA ID. Otherwise, the inference parameters in the self.params will be used to get the inference CUDA ID.
        
        Returns
        -------

        int, the CUDA ID.
        """
        if inference_params is None:
            inference_params = self.inference_params
        return inference_params.get('cuda_id', 0)

    def get_inference_depth_min_max(self, inference_params = None):
        """
        Get the min and max depth from inference configuration.

        Parameters
        ----------

        inference_params: dict, optional, default: None. If inference_params is provided, then use the parameters specified in the inference_params to get the inference depth range. Otherwise, the inference parameters in the self.params will be used to get the inference depth range.
        
        Returns
        -------

        Tuple of (int, int) the min and max depth.
        """
        if inference_params is None:
            inference_params = self.inference_params
        depth_min = inference_params.get('depth_min', 0.1)
        depth_max = inference_params.get('depth_max', 1.5)
        return depth_min, depth_max