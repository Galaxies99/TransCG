"""
Define metrics.

Authors: Hongjie Fang.
"""
import cv2
import torch
import logging
import numpy as np
from utils.logger import ColoredLogger
from utils.functions import display_results


class Metrics(object):
    """
    Define metrics for evaluation, metrics include:

        - MSE, masked MSE;

        - RMSE, masked RMSE;

        - REL, masked REL;

        - MAE, masked MAE;

        - Threshold, masked threshold.
    """
    def __init__(self, epsilon = 1e-8, depth_scale = 10.0, **kwargs):
        """
        Initialization.

        Parameters
        ----------

        epsilon: float, optional, default: 1e-8, the epsilon used in the metric calculation.
        """
        super(Metrics, self).__init__()
        self.epsilon = epsilon
        self.depth_scale = depth_scale
    
    def MSE(self, pred, gt, zero_mask, *args, **kwargs):
        """
        MSE metric.

        Parameters
        ----------

        pred: tensor, required, the predicted depth image;

        gt: tensor, required, the ground-truth depth image;

        zero_mask: tensor, required, the invalid pixel mask.

        Returns
        -------

        The MSE metric.
        """
        sample_mse = torch.sum(((pred - gt) ** 2) * zero_mask.float(), dim = [1, 2]) / torch.sum(zero_mask.float(), dim = [1, 2]) * self.depth_scale * self.depth_scale
        return torch.mean(sample_mse).item()
    
    def RMSE(self, pred, gt, zero_mask, *args, **kwargs):
        """
        RMSE metric.

        Parameters
        ----------

        pred: tensor, required, the predicted depth image;

        gt: tensor, required, the ground-truth depth image;

        zero_mask: tensor, required, the invalid pixel mask.


        Returns
        -------

        The RMSE metric.
        """
        sample_mse = torch.sum(((pred - gt) ** 2) * zero_mask.float(), dim = [1, 2]) / torch.sum(zero_mask.float(), dim = [1, 2]) * self.depth_scale * self.depth_scale
        return torch.mean(torch.sqrt(sample_mse)).item()
    
    def MaskedMSE(self, pred, gt, zero_mask, gt_mask, *args, **kwargs):
        """
        Masked MSE metric.

        Parameters
        ----------

        pred: tensor, required, the predicted depth image;

        gt: tensor, required, the ground-truth depth image;

        zero_mask: tensor, required, the invalid pixel mask;

        gt_mask: tensor, required, the ground-truth mask.

        Returns
        -------

        The masked MSE metric.
        """
        mask = gt_mask & zero_mask
        sample_masked_mse = torch.sum(((pred - gt) ** 2) * mask.float(), dim = [1, 2]) / torch.sum(mask.float(), dim = [1, 2]) * self.depth_scale * self.depth_scale
        return torch.mean(sample_masked_mse).item()
    
    def MaskedRMSE(self, pred, gt, zero_mask, gt_mask, *args, **kwargs):
        """
        Masked RMSE metric.

        Parameters
        ----------

        pred: tensor, required, the predicted depth image;

        gt: tensor, required, the ground-truth depth image;

        zero_mask: tensor, required, the invalid pixel mask;

        gt_mask: tensor, required, the ground-truth mask.

        Returns
        -------

        The masked RMSE metric.
        """
        mask = gt_mask & zero_mask
        sample_masked_mse = torch.sum(((pred - gt) ** 2) * mask.float(), dim = [1, 2]) / torch.sum(mask.float(), dim = [1, 2]) * self.depth_scale * self.depth_scale
        return torch.mean(torch.sqrt(sample_masked_mse)).item()
    
    def REL(self, pred, gt, zero_mask, *args, **kwargs):
        """
        REL metric.

        Parameters
        ----------

        pred: tensor, required, the predicted depth image;

        gt: tensor, required, the ground-truth depth image;

        zero_mask: tensor, required, the invalid pixel mask.

        Returns
        -------

        The REL metric.
        """
        sample_rel = torch.sum((torch.abs(pred - gt) / (gt + self.epsilon)) * zero_mask.float(), dim = [1, 2]) / torch.sum(zero_mask.float(), dim = [1, 2])
        return torch.mean(sample_rel).item()
    
    def MaskedREL(self, pred, gt, zero_mask, gt_mask, *args, **kwargs):
        """
        Masked REL metric.

        Parameters
        ----------

        pred: tensor, required, the predicted depth image;

        gt: tensor, required, the ground-truth depth image;

        zero_mask: tensor, required, the invalid pixel mask;

        gt_mask: tensor, required, the ground-truth mask.

        Returns
        -------

        The masked REL metric.
        """
        mask = gt_mask & zero_mask
        sample_masked_rel = torch.sum((torch.abs(pred - gt) / (gt + self.epsilon)) * mask.float(), dim = [1, 2]) / torch.sum(mask.float(), dim = [1, 2])
        return torch.mean(sample_masked_rel).item()
    
    def MAE(self, pred, gt, zero_mask, *args, **kwargs):
        """
        MAE metric.

        Parameters
        ----------
        pred: tensor, required, the predicted depth image;

        gt: tensor, required, the ground-truth depth image;

        zero_mask: tensor, required, the invalid pixel mask.

        Returns
        -------
        
        The MAE metric.
        """
        sample_mae = torch.sum(torch.abs(pred - gt) * zero_mask.float(), dim = [1, 2]) / torch.sum(zero_mask.float(), dim = [1, 2]) * self.depth_scale
        return torch.mean(sample_mae).item()
    
    def MaskedMAE(self, pred, gt, zero_mask, gt_mask, *args, **kwargs):
        """
        Masked MAE metric.

        Parameters
        ----------

        pred: tensor, required, the predicted depth image;

        gt: tensor, required, the ground-truth depth image;

        zero_mask: tensor, required, the invalid pixel mask;

        gt_mask: tensor, required, the ground-truth mask.

        Returns
        -------

        The masked MAE metric.
        """
        mask = gt_mask & zero_mask
        sample_masked_mae = torch.sum(torch.abs(pred - gt) * mask.float(), dim = [1, 2]) / torch.sum(mask.float(), dim = [1, 2]) * self.depth_scale
        return torch.mean(sample_masked_mae).item()

    def Threshold(self, pred, gt, zero_mask, *args, **kwargs):
        """
        Threshold metric.

        Parameters
        ----------

        pred: tensor, required, the predicted depth image;

        gt: tensor, required, the ground-truth depth image;

        zero_mask: tensor, required, the invalid pixel mask;

        delta: float, optional, default: 1.25, the threshold value, should be specified as "delta = xxx".

        Returns
        -------

        The threshold metric.
        """
        delta = kwargs.get('delta', 1.25)
        thres = torch.maximum(pred / (gt + self.epsilon), gt / pred)
        res = ((thres < delta) & zero_mask).float().sum(dim = [1, 2]) / torch.sum(zero_mask.float(), dim = [1, 2])
        return torch.mean(res).item() * 100
    
    def MaskedThreshold(self, pred, gt, zero_mask, gt_mask, *args, **kwargs):
        """
        Masked threshold metric.

        Parameters
        ----------

        pred: tensor, required, the predicted depth image;

        gt: tensor, required, the ground-truth depth image;

        zero_mask: tensor, required, the invalid pixel mask;

        gt_mask: tensor, required, the ground-truth mask;

        delta: float, optional, default: 1.25, the threshold value, should be specified as "delta = xxx".

        Returns
        -------

        The masked threshold metric.
        """
        delta = kwargs.get('delta', 1.25)
        mask = gt_mask & zero_mask
        thres = torch.maximum(pred / (gt + self.epsilon), gt / pred)
        res = ((thres < delta) & mask).float().sum(dim = [1, 2]) / torch.sum(mask.float(), dim = [1, 2])
        return torch.mean(res).item() * 100


class MetricsRecorder(object):
    """
    Metrics Recorder.
    """
    def __init__(self, metrics_list, epsilon = 1e-8, depth_scale = 10.0, **kwargs):
        """
        Initialization.

        Parameters
        ----------

        metrics_list: list of str, required, the metrics name list used in the metric calcuation.

        epsilon: float, optional, default: 1e-8, the epsilon used in the metric calculation.
        """
        super(MetricsRecorder, self).__init__()
        logging.setLoggerClass(ColoredLogger)
        self.logger = logging.getLogger(__name__)
        self.epsilon = epsilon
        self.depth_scale = depth_scale
        self.metrics = Metrics(epsilon = epsilon, depth_scale = depth_scale)
        self.metrics_list = []
        for metric in metrics_list:
            try:
                if "Threshold@" in metric:
                    # (Masked) Threshold metrics.
                    split_list = metric.split('@')
                    if len(split_list) != 2:
                        raise AttributeError('Invalid metric.')
                    delta = float(split_list[1])
                    metric_func = getattr(self.metrics, split_list[0])
                    self.metrics_list.append([metric, metric_func, {'delta': delta}])
                else:
                    # Other metrics.
                    metric_func = getattr(self.metrics, metric)
                    self.metrics_list.append([metric, metric_func, {}])
            except Exception:
                self.logger.warning('Unable to parse metric "{}", thus the metric is ignored.'.format(metric))
                pass
        self._clear_recorder_dict()

    def clear(self):
        """
        Clear the record dict of the metric recorder.
        """
        self._clear_recorder_dict()

    def _clear_recorder_dict(self):
        """
        Internal Function: clear the record dict of the metric recorder.
        """
        self.metrics_recorder_dict = {}
        for metric_line in self.metrics_list:
            metric_name, _, _ = metric_line
            self.metrics_recorder_dict[metric_name] = 0
        self.metrics_recorder_dict['samples'] = 0
    
    def _update_recorder_dict(self, metrics_dict):
        """
        Internal Function: update the recorder dict of the metric recorder with a metrics dict of a batch of samples.
        """
        for metric_line in self.metrics_list:
            metric_name, _, _ = metric_line
            self.metrics_recorder_dict[metric_name] += metrics_dict[metric_name] * metrics_dict['samples']
        self.metrics_recorder_dict['samples'] += metrics_dict['samples']
    
    def evaluate_batch(self, pred, gt, gt_mask, use_gt_mask, record = True, *args, **kwargs):
        """
        Evaluate a batch of the samples.

        Parameters
        ----------

        (pred, gt, gt_mask, use_gt_mask): a record, representing predicted depth image, ground-truth depth image, groud-truth mask and whether to use ground-truth mask respectively.

        record: bool, optional, default: True, whether to record the metrics of the batch of samples in the metric recorder.

        Returns
        -------

        The metrics dict of the batch of samples.
        """
        num_samples = gt.shape[0]
        zero_mask = torch.where(torch.abs(gt) < self.epsilon, False, True)
        metrics_dict = {'samples': num_samples}
        for metric_line in self.metrics_list:
            metric_name, metric_func, metric_kwargs = metric_line
            metrics_dict[metric_name] = metric_func(pred, gt, zero_mask, gt_mask, use_gt_mask, **metric_kwargs)
        if record:
            self._update_recorder_dict(metrics_dict)
        return metrics_dict

    def get_results(self):
        """
        Get the final results of metrics dict.
        """
        final_metrics_dict = self.metrics_recorder_dict.copy()
        for metric_line in self.metrics_list:
            metric_name, _, _ = metric_line
            final_metrics_dict[metric_name] /= final_metrics_dict['samples']
        return final_metrics_dict
    
    def display_results(self):
        """
        Display the metrics recorder dict.        
        """
        display_results(self.get_results(), self.logger)
