"""
Configuration builder.

Authors: Hongjie Fang.
"""
import os


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
    
    def get_optimizer(self, model, optimizer_params = None):
        """
        Get the optimizer from configuration.
        
        Parameters
        ----------
        
        model: a torch.nn.Module object, the model.
        
        optimizer_params: dict, optional, default: None. If optimizer_params is provided, then use the parameters specified in the optimizer_params to build the optimizer. Otherwise, the optimizer parameters in the self.params will be used to build the optimizer.
        
        Returns
        -------
        
        An optimizer for the given model.
        """
        from torch.optim import SGD, ASGD, Adagrad, Adamax, Adadelta, Adam, AdamW, RMSprop
        if optimizer_params is None:
            optimizer_params = self.optimizer_params
        type = optimizer_params.get('type', 'AdamW')
        params = optimizer_params.get('params', {})
        if type == 'SGD':
            optimizer = SGD(model.parameters(), **params)
        elif type == 'ASGD':
            optimizer = ASGD(model.parameters(), **params)
        elif type == 'Adagrad':
            optimizer = Adagrad(model.parameters(), **params)
        elif type == 'Adamax':
            optimizer = Adamax(model.parameters(), **params)
        elif type == 'Adadelta':
            optimizer = Adadelta(model.parameters(), **params)
        elif type == 'Adam':
            optimizer = Adam(model.parameters(), **params)
        elif type == 'AdamW':
            optimizer = AdamW(model.parameters(), **params)
        elif type == 'RMSprop':
            optimizer = RMSprop(model.parameters(), **params)
        else:
            raise NotImplementedError('Invalid optimizer type.')
        return optimizer
    
    def get_lr_scheduler(self, optimizer, lr_scheduler_params = None):
        """
        Get the learning rate scheduler from configuration.
        
        Parameters
        ----------
        
        optimizer: an optimizer;
        
        lr_scheduler_params: dict, optional, default: None. If lr_scheduler_params is provided, then use the parameters specified in the lr_scheduler_params to build the learning rate scheduler. Otherwise, the learning rate scheduler parameters in the self.params will be used to build the learning rate scheduler.
        
        Returns
        
        -------
        A learning rate scheduler for the given optimizer.
        """
        from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CyclicLR, CosineAnnealingLR, LambdaLR, StepLR
        if lr_scheduler_params is None:
            lr_scheduler_params = self.lr_scheduler_params
        type = lr_scheduler_params.get('type', '')
        params = lr_scheduler_params.get('params', {})
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
        from dataset.transparent_grasp import TransparentGrasp
        from dataset.cleargrasp import ClearGraspRealWorld
        if dataset_params is None:
            dataset_params = self.dataset_params
        dataset_params = dataset_params.get(split, {})
        type = dataset_params.get('type', 'transparent-grasp')
        if type == 'transparent-grasp':
            dataset = TransparentGrasp(split = split, **dataset_params)
        elif type == 'cleargrasp':
            dataset = ClearGraspRealWorld(split = split, **dataset_params)
        else:
            raise NotImplementedError('Invalid dataset type.')
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
        loss_epsilon = criterion_params.get('epsilon', 1e-8)
        from utils.criterion import Criterion
        criterion = Criterion(loss_type = loss_type, epsilon = loss_epsilon)
        return criterion
    
    def get_metrics(self, metrics_params = None):
        """
        Get the metrics settings from configuration.

        Parameters
        ----------

        metrics_params: dict, optional, default: None. If metrics_params is provided, then use the parameters specified in the metrics_params to get the metrics. Otherwise, the metrics parameters in the self.params will be used to get the metrics
        
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