import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()
    
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
        loss = torch.mean((pred - gt) ** 2)
        return loss


class MaskedMSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(MaskedMSELoss, self).__init__()
    
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
        delta = (pred - gt) ** 2
        loss = torch.sum(delta * gt_mask.float(), dim = [1, 2]) / torch.sum(gt_mask.float(), dim = [1, 2])
        return torch.mean(loss)


class MaskedTransparentLoss(nn.Module):
    def __init__(self, **kwargs):
        super(MaskedTransparentLoss, self).__init__()
    
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
        _, use_gt_mask = torch.broadcast_tensors(gt_mask.transpose(0, 2), use_gt_mask.view(-1))
        gt_mask = ~ (~ gt_mask & use_gt_mask.transpose(0, 2))
        delta = (pred - gt) ** 2
        loss = torch.sum(delta * gt_mask.float(), dim = [1, 2]) / torch.sum(gt_mask.float(), dim = [1, 2])
        return torch.mean(loss)


class Metrics(nn.Module):
    def __init__(self, **kwargs):
        super(Metrics, self).__init__()
        self.clear()
    
    def MSE(self, pred, gt, *args, **kwargs):
        sample_mse = torch.mean((pred - gt) ** 2, dim = [1, 2]) * 100
        return torch.mean(sample_mse)
    
    def RMSE(self, pred, gt, *args, **kwargs):
        sample_mse = torch.mean((pred - gt) ** 2, dim = [1, 2]) * 100
        return torch.mean(torch.sqrt(sample_mse))
    
    def MaskedMSE(self, pred, gt, gt_mask, *args, **kwargs):
        sample_masked_mse = torch.sum(((pred - gt) ** 2) * gt_mask.float(), dim = [1, 2]) / torch.sum(gt_mask.float(), dim = [1, 2]) * 100
        return torch.mean(sample_masked_mse)
    
    def MaskedRMSE(self, pred, gt, gt_mask, *args, **kwargs):
        sample_masked_mse = torch.sum(((pred - gt) ** 2) * gt_mask.float(), dim = [1, 2]) / torch.sum(gt_mask.float(), dim = [1, 2]) * 100
        return torch.mean(torch.sqrt(sample_masked_mse))
    
    def REL(self, pred, gt, *args, **kwargs):
        sample_rel = torch.mean(torch.abs(pred - gt) / gt, dim = [1, 2])
        return torch.mean(sample_rel)
    
    def MaskedREL(self, pred, gt, gt_mask, *args, **kwargs):
        sample_masked_rel = torch.sum((torch.abs(pred - gt) / gt) * gt_mask.float(), dim = [1, 2]) / torch.sum(gt_mask.float(), dim = [1, 2])
        return torch.mean(sample_masked_rel)
    
    def MAE(self, pred, gt, *args, **kwargs):
        sample_mae = torch.mean(torch.abs(pred - gt), dim = [1, 2]) * 10
        return torch.mean(sample_mae)
    
    def MaskedMAE(self, pred, gt, gt_mask, *args, **kwargs):
        sample_masked_mae = torch.sum(torch.abs(pred - gt) * gt_mask.float(), dim = [1, 2]) / torch.sum(gt_mask.float(), dim = [1, 2]) * 10
        return torch.mean(sample_masked_mae)

    def Threshold(self, pred, gt, delta, *args, **kwargs):
        thres = torch.maximum(pred / gt, gt / pred)
        res = (thres < delta).float().sum(dim = [1, 2]) / (gt.shape[1] * gt.shape[2])
        return torch.mean(res)
    
    def MaskedThreshold(self, pred, gt, gt_mask, delta, *args, **kwargs):
        thres = torch.maximum(pred / gt, gt / pred)
        res = ((thres < delta) & gt_mask).float().sum(dim = [1, 2]) / torch.sum(gt_mask.float(), dim = [1, 2])
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
        self.mse_list.append(self.MSE(pred, gt) * num_samples)
        self.rmse_list.append(self.RMSE(pred, gt) * num_samples)
        self.masked_mse_list.append(self.MaskedMSE(pred, gt, gt_mask) * num_samples)
        self.masked_rmse_list.append(self.MaskedRMSE(pred, gt, gt_mask) * num_samples)
        self.rel_list.append(self.REL(pred, gt) * num_samples)
        self.masked_rel_list.append(self.MaskedREL(pred, gt, gt_mask) * num_samples)
        self.mae_list.append(self.MAE(pred, gt) * num_samples)
        self.masked_mae_list.append(self.MaskedMAE(pred, gt, gt_mask) * num_samples)
        self.th105_list.append(self.Threshold(pred, gt, 1.05) * num_samples)
        self.th110_list.append(self.Threshold(pred, gt, 1.10) * num_samples)
        self.th125_list.append(self.Threshold(pred, gt, 1.25) * num_samples)
        self.masked_th105_list.append(self.MaskedThreshold(pred, gt, gt_mask, 1.05) * num_samples)
        self.masked_th110_list.append(self.MaskedThreshold(pred, gt, gt_mask, 1.10) * num_samples)
        self.masked_th125_list.append(self.MaskedThreshold(pred, gt, gt_mask, 1.25) * num_samples)
        self.num_samples += num_samples
    
    def final(self):
        return (
            torch.sum(torch.stack(self.mse_list)).item() / self.num_samples, 
            torch.sum(torch.stack(self.masked_mse_list)).item() / self.num_samples,
            torch.sum(torch.stack(self.rmse_list)).item() / self.num_samples,
            torch.sum(torch.stack(self.masked_rmse_list)).item() / self.num_samples,
            torch.sum(torch.stack(self.rel_list)).item() / self.num_samples,
            torch.sum(torch.stack(self.masked_rel_list)).item() / self.num_samples,
            torch.sum(torch.stack(self.mae_list)).item() / self.num_samples,
            torch.sum(torch.stack(self.masked_mae_list)).item() / self.num_samples,
            torch.sum(torch.stack(self.th105_list)).item() / self.num_samples,
            torch.sum(torch.stack(self.masked_th105_list)).item() / self.num_samples,
            torch.sum(torch.stack(self.th110_list)).item() / self.num_samples,
            torch.sum(torch.stack(self.masked_th110_list)).item() / self.num_samples,
            torch.sum(torch.stack(self.th125_list)).item() / self.num_samples,
            torch.sum(torch.stack(self.masked_th125_list)).item() / self.num_samples
        )
    