import os
from dataset.transparent_grasp import TransparentDataset
from model.DFNet import DFNet

class ConfigBuilder(object):
    '''
    Configuration Builder
    '''
    def __init__(self, **params):
        '''
        Set the default configuration for the configuration builder.

        Parameters
        ----------
        params: the configuration parameters.
        '''
        super(ConfigBuilder, self).__init__()
        self.params = params
    
    def get_model(self, model_params = None):
        '''
        Get the model from configuration.

        Parameters
        ----------
        model_params: dict, optional, default: None. If model_params is provided, then use the parameters specified in the model_params to build the model. Otherwise, the model parameters in the self.params will be used to build the model.
        
        Returns
        -------
        A model, which is usually a torch.nn.Module object.
        '''
        if model_params is None:
            model_params = self.params.get('model', {})
        name = model_params.get('type', 'DFNet')
        params = model_params.get('params', {})
        if type == 'DFNet':
            model = DFNet(**params)
        else:
            raise NotImplementedError('Invalid model type.')
        return None
    
    def get_optimizer(self, model, optimizer_params = None):
        '''
        Get the optimizer from configuration.
        
        Parameters
        ----------
        model: a torch.nn.Module object, the model.
        optimizer_params: dict, optional, default: None. If optimizer_params is provided, then use the parameters specified in the optimizer_params to build the optimizer. Otherwise, the optimizer parameters in the self.params will be used to build the optimizer.
        
        Returns
        -------
        An optimizer for the given model.
        '''
        from torch.optim import SGD, ASGD, Adagrad, Adamax, Adadelta, Adam, AdamW, RMSprop
        if optimizer_params is None:
            optimizer_params = self.params.get('optimizer', {})
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
        '''
        Get the learning rate scheduler from configuration.
        
        Parameters
        ----------
        optimizer: an optimizer;
        lr_scheduler_params: dict, optional, default: None. If lr_scheduler_params is provided, then use the parameters specified in the lr_scheduler_params to build the learning rate scheduler. Otherwise, the learning rate scheduler parameters in the self.params will be used to build the learning rate scheduler.
        
        Returns
        -------
        A learning rate scheduler for the given optimizer.
        '''
        from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CyclicLR, CosineAnnealingLR, LambdaLR, StepLR
        if lr_scheduler_params is None:
            lr_scheduler_params = self.params.get('lr_scheduler', {})
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
        '''
        Get the dataset from configuration.

        Parameters
        ----------
        dataset_params: dict, optional, default: None. If dataset_params is provided, then use the parameters specified in the dataset_params to build the dataset. Otherwise, the dataset parameters in the self.params will be used to build the dataset;
        split: str in ['train', 'test'], optional, default: 'train', the splitted dataset.

        Returns
        -------
        A torch.utils.data.Dataset item.
        '''
        if dataset_params is None:
            dataset_params = self.params.get('dataset', {"data_dir": "data"})
        return TransparentDataset(split = split, **dataset_params)
    
    def get_dataloader(self, dataset_params = None, split = 'train', batch_size = None, num_workers = None, shuffle = None):
        '''
        Get the dataloader from configuration.

        Parameters
        ----------
        dataset_params: dict, optional, default: None. If dataset_params is provided, then use the parameters specified in the dataset_params to build the dataset. Otherwise, the dataset parameters in the self.params will be used to build the dataset;
        split: str in ['train', 'test'], optional, default: 'train', the splitted dataset;
        batch_size: int, optional, default: None. If batch_size is None, then the batch size parameter in the self.params will be used to represent the batch size (If still not specified, default: 4);
        num_workers: int, optional, default: None. If num_workers is None, then the worker number parameter in the self.params will be used to represent the worker number (If still not specified, default: 16).
        shuffle: bool, optional, default: None. If shuffle is None, then the shuffle parameter in the self.params will be used to represent the shuffle option (If still not specified, default: True).

        Returns
        -------
        A torch.utils.data.DataLoader item.
        '''
        from torch.utils.data import DataLoader
        if batch_size is None:
            batch_size = self.params.get('trainer', {}).get('batch_size', 4)
        if num_workers is None:
            num_workers = self.params.get('trainer', {}).get('num_workers', 16)
        if shuffle is None:
            shuffle = self.params.get('trainer', {}).get('shuffle', True)
        dataset = self.get_dataset(dataset_params, split)
        return DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers
        )

    def get_max_epoch(self):
        '''
        Get the max epoch from configuration.

        Returns
        -------
        An integer, which is the max epoch (default: 50).
        '''
        return self.params.get('trainer', {}).get('max_epoch', 50)
    
    def get_stats_dir(self, stats_params = None):
        '''
        Get the statistics directory from configuration.

        Parameters
        ----------
        stats_params: dict, optional, default: None. If stats_params is provided, then use the parameters specified in the stats_params to get the statistics directory. Otherwise, the statistics parameters in the self.params will be used to get the statistics directory.

        Returns
        -------
        A string, the statistics directory.
        '''
        if stats_params is None:
            stats_params = self.params.get('stats', {})
        stats_dir = stats_params.get('stats_dir', 'stats')
        stats_exper = stats_params.get('stats_exper', 'default')
        stats_res_dir = os.path.join(stats_dir, stats_exper)
        return stats_res_dir
