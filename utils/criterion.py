"""
Define criterions.

Authors: Hongjie Fang.
"""
import torch
import torch.nn as nn
import numpy as np


class Criterion(nn.Module):
    """
    Various type of criterions.
    """
    def __init__(self, type, epsilon = 1e-8, huber_k = 0.01, **kwargs):
        super(Criterion, self).__init__()
        self.epsilon = epsilon
        self.l2_loss = self.mse_loss
        self.masked_l2_loss = self.masked_mse_loss
        self.custom_masked_l2_loss = self.custom_masked_mse_loss
        self.huber_k = huber_k
        self.forward = getattr(self, type)
        self._mse = self._l2
    
    def _l1(self, pred, gt):
        """
        L1 loss in pixel-wise representations.
        """
        return torch.abs(pred - gt)

    def _l2(self, pred, gt):
        """
        L2 loss in pixel-wise representations.
        """
        return (pred - gt) ** 2
    
    def _huber(self, pred, gt):
        """
        Huber loss in pixel-wise representations.
        """
        delta = torch.abs(pred - gt)
        return torch.where(delta <= self.huber_k, delta ** 2 / 2, self.huber_k * delta - self.huber_k ** 2 / 2)
    
    def mse_loss(self, pred, gt, *args, **kwargs):
        """
        MSE loss.

        Parameters
        ----------
        
        pred: tensor of shape NHW, the prediction;
        
        gt: tensor of shape NHW, the ground-truth.

        Returns
        -------

        The MSE loss.
        """
        mask = torch.where(gt < self.epsilon, False, True)
        delta = self._l2(pred, gt)
        mask_base = torch.sum(mask.float(), dim = [1, 2])
        mask_base = torch.where(mask_base < self.epsilon, mask_base + self.epsilon, mask_base)
        loss = torch.sum(delta * mask.float(), dim = [1, 2]) / mask_base
        return torch.mean(loss)
    
    def masked_mse_loss(self, pred, gt, gt_mask, *args, **kwargs):
        """
        Masked MSE loss.
        
        Parameters
        ----------
        
        pred: tensor of shape NHW, the prediction;
        
        gt: tensor of shape NHW, the ground-truth;
        
        gt_mask: tensor of shape NHW, the ground-truth mask.

        Returns
        -------

        The masked MSE loss.
        """
        zero_mask = torch.where(gt < self.epsilon, False, True)
        mask = gt_mask & zero_mask
        delta = self._l2(pred, gt)
        mask_base = torch.sum(mask.float(), dim = [1, 2])
        mask_base = torch.where(mask_base < self.epsilon, mask_base + self.epsilon, mask_base)
        loss = torch.sum(delta * mask.float(), dim = [1, 2]) / mask_base
        return torch.mean(loss)
    
    def custom_masked_mse_loss(self, pred, gt, gt_mask, use_gt_mask, *args, **kwargs):
        """
        Custom masked MSE loss.
        
        Parameters
        ----------
        
        pred: tensor of shape NHW, the prediction;
        
        gt: tensor of shape NHW, the ground-truth;
        
        gt_mask: tensor of shape NHW, the ground-truth mask;
        
        use_gt_mask: tensor of shape N, whether to use the ground-truth mask.

        Returns
        -------

        The custom masked MSE loss.
        """
        zero_mask = torch.where(gt < self.epsilon, False, True)
        _, use_gt_mask = torch.broadcast_tensors(gt_mask.transpose(0, 2), use_gt_mask.view(-1))
        gt_mask = ~ (~ gt_mask & use_gt_mask.transpose(0, 2))
        mask = gt_mask & zero_mask
        delta = self._l2(pred, gt)
        mask_base = torch.sum(mask.float(), dim = [1, 2])
        mask_base = torch.where(mask_base < self.epsilon, mask_base + self.epsilon, mask_base)
        loss = torch.sum(delta * mask.float(), dim = [1, 2]) / mask_base
        return torch.mean(loss)
    
    def l1_loss(self, pred, gt, *args, **kwargs):
        """
        L1 loss.

        Parameters
        ----------
        
        pred: tensor of shape NHW, the prediction;
        
        gt: tensor of shape NHW, the ground-truth.

        Returns
        -------

        The L1 loss.
        """
        mask = torch.where(gt < self.epsilon, False, True)
        delta = self._l1(pred, gt)
        mask_base = torch.sum(mask.float(), dim = [1, 2])
        mask_base = torch.where(mask_base < self.epsilon, mask_base + self.epsilon, mask_base)
        loss = torch.sum(delta * mask.float(), dim = [1, 2]) / mask_base
        return torch.mean(loss)
    
    def masked_l1_loss(self, pred, gt, gt_mask, *args, **kwargs):
        """
        Masked L1 loss.
        
        Parameters
        ----------
        
        pred: tensor of shape NHW, the prediction;
        
        gt: tensor of shape NHW, the ground-truth;
        
        gt_mask: tensor of shape NHW, the ground-truth mask.

        Returns
        -------

        The masked L1 loss.
        """
        zero_mask = torch.where(gt < self.epsilon, False, True)
        mask = gt_mask & zero_mask
        delta = self._l1(pred, gt)
        mask_base = torch.sum(mask.float(), dim = [1, 2])
        mask_base = torch.where(mask_base < self.epsilon, mask_base + self.epsilon, mask_base)
        loss = torch.sum(delta * mask.float(), dim = [1, 2]) / mask_base
        return torch.mean(loss)
    
    def custom_masked_l1_loss(self, pred, gt, gt_mask, use_gt_mask, *args, **kwargs):
        """
        Custom masked L1 loss.
        
        Parameters
        ----------
        
        pred: tensor of shape NHW, the prediction;
        
        gt: tensor of shape NHW, the ground-truth;
        
        gt_mask: tensor of shape NHW, the ground-truth mask;
        
        use_gt_mask: tensor of shape N, whether to use the ground-truth mask.

        Returns
        -------

        The custom masked L1 loss.
        """
        zero_mask = torch.where(gt < self.epsilon, False, True)
        _, use_gt_mask = torch.broadcast_tensors(gt_mask.transpose(0, 2), use_gt_mask.view(-1))
        gt_mask = ~ (~ gt_mask & use_gt_mask.transpose(0, 2))
        mask = gt_mask & zero_mask
        delta = self._l1(pred, gt)
        mask_base = torch.sum(mask.float(), dim = [1, 2])
        mask_base = torch.where(mask_base < self.epsilon, mask_base + self.epsilon, mask_base)
        loss = torch.sum(delta * mask.float(), dim = [1, 2]) / mask_base
        return torch.mean(loss)
    
    def huber_loss(self, pred, gt, *args, **kwargs):
        """
        Huber loss.

        Parameters
        ----------
        
        pred: tensor of shape NHW, the prediction;
        
        gt: tensor of shape NHW, the ground-truth.

        Returns
        -------

        The huber loss.
        """
        mask = torch.where(gt < self.epsilon, False, True)
        delta = self._huber(pred, gt)
        mask_base = torch.sum(mask.float(), dim = [1, 2])
        mask_base = torch.where(mask_base < self.epsilon, mask_base + self.epsilon, mask_base)
        loss = torch.sum(delta * mask.float(), dim = [1, 2]) / mask_base
        return torch.mean(loss)

    def masked_huber_loss(self, pred, gt, gt_mask, *args, **kwargs):
        """
        Masked Huber loss.
        
        Parameters
        ----------
        
        pred: tensor of shape NHW, the prediction;
        
        gt: tensor of shape NHW, the ground-truth;
        
        gt_mask: tensor of shape NHW, the ground-truth mask.

        Returns
        -------

        The masked huber loss.
        """
        zero_mask = torch.where(gt < self.epsilon, False, True)
        mask = gt_mask & zero_mask
        delta = self._huber(pred, gt)
        mask_base = torch.sum(mask.float(), dim = [1, 2])
        mask_base = torch.where(mask_base < self.epsilon, mask_base + self.epsilon, mask_base)
        loss = torch.sum(delta * mask.float(), dim = [1, 2]) / mask_base
        return torch.mean(loss)
    
    def custom_masked_huber_loss(self, pred, gt, gt_mask, use_gt_mask, *args, **kwargs):
        """
        Custom masked huber loss.
        
        Parameters
        ----------
        
        pred: tensor of shape NHW, the prediction;
        
        gt: tensor of shape NHW, the ground-truth;
        
        gt_mask: tensor of shape NHW, the ground-truth mask;
        
        use_gt_mask: tensor of shape N, whether to use the ground-truth mask.

        Returns
        -------

        The custom masked huber loss.
        """
        zero_mask = torch.where(gt < self.epsilon, False, True)
        _, use_gt_mask = torch.broadcast_tensors(gt_mask.transpose(0, 2), use_gt_mask.view(-1))
        gt_mask = ~ (~ gt_mask & use_gt_mask.transpose(0, 2))
        mask = gt_mask & zero_mask
        delta = self._huber(pred, gt)
        mask_base = torch.sum(mask.float(), dim = [1, 2])
        mask_base = torch.where(mask_base < self.epsilon, mask_base + self.epsilon, mask_base)
        loss = torch.sum(delta * mask.float(), dim = [1, 2]) / mask_base
        return torch.mean(loss)