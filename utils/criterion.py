import torch
import torch.nn as nn
import numpy as np


class MSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()
        self.epsilon = kwargs.get('epsilon', 1e-10)
    
    def forward(self, pred, gt, gt_mask, use_gt_mask):
        """
        MSE Loss.

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
    def __init__(self, **kwargs):
        super(MaskedMSELoss, self).__init__()
        self.epsilon = kwargs.get('epsilon', 1e-10)
    
    def forward(self, pred, gt, gt_mask, use_gt_mask):
        """
        Masked MSE Loss.
        
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
    def __init__(self, **kwargs):
        super(MaskedTransparentLoss, self).__init__()
        self.epsilon = kwargs.get('epsilon', 1e-10)
    
    def forward(self, pred, gt, gt_mask, use_gt_mask):
        """
        Masked Transparent Loss:
        - For cluttered scenes: MSE loss;
        - For isolated scenes: masked MSE loss.
        
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
    def __init__(self, **kwargs):
        super(Metrics, self).__init__()
        self.epsilon = kwargs.get('epsilon', 1e-10)
        self.clear()
    
    def MSE(self, pred, gt, zero_mask, *args, **kwargs):
        sample_mse = torch.sum(((pred - gt) ** 2) * zero_mask.float(), dim = [1, 2]) / torch.sum(zero_mask.float(), dim = [1, 2]) * 100
        return torch.mean(sample_mse)
    
    def RMSE(self, pred, gt, zero_mask, *args, **kwargs):
        sample_mse = torch.sum(((pred - gt) ** 2) * zero_mask.float(), dim = [1, 2]) / torch.sum(zero_mask.float(), dim = [1, 2]) * 100
        return torch.mean(torch.sqrt(sample_mse))
    
    def MaskedMSE(self, pred, gt, gt_mask, zero_mask, *args, **kwargs):
        mask = gt_mask & zero_mask
        sample_masked_mse = torch.sum(((pred - gt) ** 2) * mask.float(), dim = [1, 2]) / torch.sum(mask.float(), dim = [1, 2]) * 100
        return torch.mean(sample_masked_mse)
    
    def MaskedRMSE(self, pred, gt, gt_mask, zero_mask, *args, **kwargs):
        mask = gt_mask & zero_mask
        sample_masked_mse = torch.sum(((pred - gt) ** 2) * mask.float(), dim = [1, 2]) / torch.sum(mask.float(), dim = [1, 2]) * 100
        return torch.mean(torch.sqrt(sample_masked_mse))
    
    def REL(self, pred, gt, zero_mask, *args, **kwargs):
        sample_rel = torch.sum((torch.abs(pred - gt) / (gt + self.epsilon)) * zero_mask.float(), dim = [1, 2]) / torch.sum(zero_mask.float(), dim = [1, 2])
        return torch.mean(sample_rel)
    
    def MaskedREL(self, pred, gt, gt_mask, zero_mask, *args, **kwargs):
        mask = gt_mask & zero_mask
        sample_masked_rel = torch.sum((torch.abs(pred - gt) / (gt + self.epsilon)) * mask.float(), dim = [1, 2]) / torch.sum(mask.float(), dim = [1, 2])
        return torch.mean(sample_masked_rel)
    
    def MAE(self, pred, gt, zero_mask, *args, **kwargs):
        sample_mae = torch.sum(torch.abs(pred - gt) * zero_mask.float(), dim = [1, 2]) / torch.sum(zero_mask.float(), dim = [1, 2]) * 10
        return torch.mean(sample_mae)
    
    def MaskedMAE(self, pred, gt, gt_mask, zero_mask, *args, **kwargs):
        mask = gt_mask & zero_mask
        sample_masked_mae = torch.sum(torch.abs(pred - gt) * mask.float(), dim = [1, 2]) / torch.sum(mask.float(), dim = [1, 2]) * 10
        return torch.mean(sample_masked_mae)

    def Threshold(self, pred, gt, zero_mask, delta, *args, **kwargs):
        thres = torch.maximum(pred / (gt + self.epsilon), gt / pred)
        res = ((thres < delta) & zero_mask).float().sum(dim = [1, 2]) / torch.sum(zero_mask.float(), dim = [1, 2])
        return torch.mean(res)
    
    def MaskedThreshold(self, pred, gt, gt_mask, zero_mask, delta, *args, **kwargs):
        mask = gt_mask & zero_mask
        thres = torch.maximum(pred / (gt + self.epsilon), gt / pred)
        res = ((thres < delta) & mask).float().sum(dim = [1, 2]) / torch.sum(mask.float(), dim = [1, 2])
        return torch.mean(res)
    
    def clear(self):
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
    