"""
Define criterions and metrics.

Authors: Hongjie Fang.
"""
import torch
import torch.nn as nn
import numpy as np


class MSELoss(nn.Module):
    """
    MSE loss.
    """
    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()
        self.epsilon = kwargs.get('epsilon', 1e-10)
    
    def forward(self, pred, gt, gt_mask, use_gt_mask):
        """
        MSE loss.

        Parameters
        ----------
        pred: NHW
        gt: NHW
        gt_mask: NHW
        use_gt_mask: N
        """
        mask = torch.where(gt < self.epsilon, False, True)
        delta = (pred - gt) ** 2
        loss = torch.sum(delta * mask.float(), dim = [1, 2]) / torch.sum(mask.float(), dim = [1, 2])
        return torch.mean(loss)


class MaskedMSELoss(nn.Module):
    """
    Masked MSE loss.
    """
    def __init__(self, **kwargs):
        super(MaskedMSELoss, self).__init__()
        self.epsilon = kwargs.get('epsilon', 1e-10)
    
    def forward(self, pred, gt, gt_mask, use_gt_mask):
        """
        Masked MSE loss.
        
        Parameters
        ----------
        pred: NHW
        gt: NHW
        gt_mask: NHW
        use_gt_mask: N
        """
        zero_mask = torch.where(gt < self.epsilon, False, True)
        mask = gt_mask & zero_mask
        delta = (pred - gt) ** 2
        loss = torch.sum(delta * mask.float(), dim = [1, 2]) / torch.sum(mask.float(), dim = [1, 2])
        return torch.mean(loss)


class MaskedTransparentLoss(nn.Module):
    """
    Custom loss: masked transparent loss.
        - For cluttered scenes: MSE loss;
        - For isolated scenes: masked MSE loss.
    """
    def __init__(self, **kwargs):
        super(MaskedTransparentLoss, self).__init__()
        self.epsilon = kwargs.get('epsilon', 1e-10)
    
    def forward(self, pred, gt, gt_mask, use_gt_mask):
        """
        Masked transparent loss.
        
        Parameters
        ----------
        pred: NHW
        gt: NHW
        gt_mask: NHW
        use_gt_mask: N
        """
        zero_mask = torch.where(gt < self.epsilon, False, True)
        _, use_gt_mask = torch.broadcast_tensors(gt_mask.transpose(0, 2), use_gt_mask.view(-1))
        gt_mask = ~ (~ gt_mask & use_gt_mask.transpose(0, 2))
        mask = gt_mask & zero_mask
        delta = (pred - gt) ** 2
        loss = torch.sum(delta * mask.float(), dim = [1, 2]) / torch.sum(mask.float(), dim = [1, 2])
        return torch.mean(loss)


class Metrics(nn.Module):
    """
    Define metrics for evaluation, metrics include:
        - MSE, masked MSE;
        - RMSE, masked RMSE;
        - REL, masked REL;
        - MAE, masked MAE;
        - Threshold, masked threshold.
    """
    def __init__(self, **kwargs):
        super(Metrics, self).__init__()
        self.epsilon = kwargs.get('epsilon', 1e-10)
        self.clear()
    
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
        sample_mse = torch.sum(((pred - gt) ** 2) * zero_mask.float(), dim = [1, 2]) / torch.sum(zero_mask.float(), dim = [1, 2]) * 100
        return torch.mean(sample_mse)
    
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
        sample_mse = torch.sum(((pred - gt) ** 2) * zero_mask.float(), dim = [1, 2]) / torch.sum(zero_mask.float(), dim = [1, 2]) * 100
        return torch.mean(torch.sqrt(sample_mse))
    
    def MaskedMSE(self, pred, gt, gt_mask, zero_mask, *args, **kwargs):
        """
        Masked MSE metric.

        Parameters
        ----------
        pred: tensor, required, the predicted depth image;
        gt: tensor, required, the ground-truth depth image;
        gt_mask: tensor, required, the ground-truth mask;
        zero_mask: tensor, required, the invalid pixel mask.

        Returns
        -------
        The masked MSE metric.
        """
        mask = gt_mask & zero_mask
        sample_masked_mse = torch.sum(((pred - gt) ** 2) * mask.float(), dim = [1, 2]) / torch.sum(mask.float(), dim = [1, 2]) * 100
        return torch.mean(sample_masked_mse)
    
    def MaskedRMSE(self, pred, gt, gt_mask, zero_mask, *args, **kwargs):
        """
        Masked RMSE metric.

        Parameters
        ----------
        pred: tensor, required, the predicted depth image;
        gt: tensor, required, the ground-truth depth image;
        gt_mask: tensor, required, the ground-truth mask;
        zero_mask: tensor, required, the invalid pixel mask.

        Returns
        -------
        The masked RMSE metric.
        """
        mask = gt_mask & zero_mask
        sample_masked_mse = torch.sum(((pred - gt) ** 2) * mask.float(), dim = [1, 2]) / torch.sum(mask.float(), dim = [1, 2]) * 100
        return torch.mean(torch.sqrt(sample_masked_mse))
    
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
        return torch.mean(sample_rel)
    
    def MaskedREL(self, pred, gt, gt_mask, zero_mask, *args, **kwargs):
        """
        Masked REL metric.

        Parameters
        ----------
        pred: tensor, required, the predicted depth image;
        gt: tensor, required, the ground-truth depth image;
        gt_mask: tensor, required, the ground-truth mask;
        zero_mask: tensor, required, the invalid pixel mask.

        Returns
        -------
        The masked REL metric.
        """
        mask = gt_mask & zero_mask
        sample_masked_rel = torch.sum((torch.abs(pred - gt) / (gt + self.epsilon)) * mask.float(), dim = [1, 2]) / torch.sum(mask.float(), dim = [1, 2])
        return torch.mean(sample_masked_rel)
    
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
        sample_mae = torch.sum(torch.abs(pred - gt) * zero_mask.float(), dim = [1, 2]) / torch.sum(zero_mask.float(), dim = [1, 2]) * 10
        return torch.mean(sample_mae)
    
    def MaskedMAE(self, pred, gt, gt_mask, zero_mask, *args, **kwargs):
        """
        Masked MAE metric.

        Parameters
        ----------
        pred: tensor, required, the predicted depth image;
        gt: tensor, required, the ground-truth depth image;
        gt_mask: tensor, required, the ground-truth mask;
        zero_mask: tensor, required, the invalid pixel mask.

        Returns
        -------
        The masked MAE metric.
        """
        mask = gt_mask & zero_mask
        sample_masked_mae = torch.sum(torch.abs(pred - gt) * mask.float(), dim = [1, 2]) / torch.sum(mask.float(), dim = [1, 2]) * 10
        return torch.mean(sample_masked_mae)

    def Threshold(self, pred, gt, zero_mask, delta, *args, **kwargs):
        """
        Threshold metric.

        Parameters
        ----------
        pred: tensor, required, the predicted depth image;
        gt: tensor, required, the ground-truth depth image;
        zero_mask: tensor, required, the invalid pixel mask;
        delta: float, required, the threshold value.

        Returns
        -------
        The threshold metric.
        """
        thres = torch.maximum(pred / (gt + self.epsilon), gt / pred)
        res = ((thres < delta) & zero_mask).float().sum(dim = [1, 2]) / torch.sum(zero_mask.float(), dim = [1, 2])
        return torch.mean(res)
    
    def MaskedThreshold(self, pred, gt, gt_mask, zero_mask, delta, *args, **kwargs):
        """
        Masked threshold metric.

        Parameters
        ----------
        pred: tensor, required, the predicted depth image;
        gt: tensor, required, the ground-truth depth image;
        gt_mask: tensor, required, the ground-truth mask;
        zero_mask: tensor, required, the invalid pixel mask;
        delta: float, required, the threshold value.

        Returns
        -------
        The masked threshold metric.
        """
        mask = gt_mask & zero_mask
        thres = torch.maximum(pred / (gt + self.epsilon), gt / pred)
        res = ((thres < delta) & mask).float().sum(dim = [1, 2]) / torch.sum(mask.float(), dim = [1, 2])
        return torch.mean(res)
    
    def clear(self):
        """
        Clear the record of the metric calculator.
        """
        self.mse_list = []
        self.rmse_list = []
        self.masked_mse_list = []
        self.masked_rmse_list = []
        self.rel_list = []
        self.masked_rel_list = []
        self.mae_list = []
        self.masked_mae_list = []
        self.th105_list = []
        self.th110_list = []
        self.th125_list = []
        self.masked_th105_list = []
        self.masked_th110_list = []
        self.masked_th125_list = []
        self.num_samples = 0
    
    def add_record(self, pred, gt, gt_mask, use_gt_mask, *args, **kwargs):
        """
        Add a record to the metric calculator.

        Parameters
        ----------
        (pred, gt, gt_mask, use_gt_mask): a record, representing predicted depth image, ground-truth depth image, groud-truth mask and whether to use ground-truth mask respectively.
        """
        num_samples = gt.shape[0]
        zero_mask = torch.where(torch.abs(gt) < self.epsilon, False, True)
        self.mse_list.append(self.MSE(pred, gt, zero_mask).item() * num_samples)
        self.rmse_list.append(self.RMSE(pred, gt, zero_mask).item() * num_samples)
        self.masked_mse_list.append(self.MaskedMSE(pred, gt, gt_mask, zero_mask).item() * num_samples)
        self.masked_rmse_list.append(self.MaskedRMSE(pred, gt, gt_mask, zero_mask).item() * num_samples)
        self.rel_list.append(self.REL(pred, gt, zero_mask).item() * num_samples)
        self.masked_rel_list.append(self.MaskedREL(pred, gt, gt_mask, zero_mask).item() * num_samples)
        self.mae_list.append(self.MAE(pred, gt, zero_mask).item() * num_samples)
        self.masked_mae_list.append(self.MaskedMAE(pred, gt, gt_mask, zero_mask).item() * num_samples)
        self.th105_list.append(self.Threshold(pred, gt, zero_mask, 1.05).item() * num_samples)
        self.th110_list.append(self.Threshold(pred, gt, zero_mask, 1.10).item() * num_samples)
        self.th125_list.append(self.Threshold(pred, gt, zero_mask, 1.25).item() * num_samples)
        self.masked_th105_list.append(self.MaskedThreshold(pred, gt, gt_mask, zero_mask, 1.05).item() * num_samples)
        self.masked_th110_list.append(self.MaskedThreshold(pred, gt, gt_mask, zero_mask, 1.10).item() * num_samples)
        self.masked_th125_list.append(self.MaskedThreshold(pred, gt, gt_mask, zero_mask, 1.25).item() * num_samples)
        self.num_samples += num_samples
    
    def final(self):
        """
        Return the final metrics based on the records.
        
        Returns
        -------
        A tuple consisting of 14 metrics, namely:
            - MSE o/w mask;
            - RMSE o/w mask;
            - REL o/w mask;
            - MAE o/w mask;
            - Threshold@1.05 o/w mask;
            - Threshold@1.10 o/w mask;
            - Threshold@1.25 o/w mask.
        """
        return (
            np.sum(np.stack(self.mse_list)) / self.num_samples, 
            np.sum(np.stack(self.masked_mse_list)) / self.num_samples,
            np.sum(np.stack(self.rmse_list)) / self.num_samples,
            np.sum(np.stack(self.masked_rmse_list)) / self.num_samples,
            np.sum(np.stack(self.rel_list)) / self.num_samples,
            np.sum(np.stack(self.masked_rel_list)) / self.num_samples,
            np.sum(np.stack(self.mae_list)) / self.num_samples,
            np.sum(np.stack(self.masked_mae_list)) / self.num_samples,
            np.sum(np.stack(self.th105_list)) / self.num_samples,
            np.sum(np.stack(self.masked_th105_list)) / self.num_samples,
            np.sum(np.stack(self.th110_list)) / self.num_samples,
            np.sum(np.stack(self.masked_th110_list)) / self.num_samples,
            np.sum(np.stack(self.th125_list)) / self.num_samples,
            np.sum(np.stack(self.masked_th125_list)) / self.num_samples
        )
    