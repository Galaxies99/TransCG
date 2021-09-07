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
    def __init__(self, loss_type, epsilon = 1e-8, **kwargs):
        super(Criterion, self).__init__()
        self.epsilon = epsilon
        self.l2_loss = self.mse_loss
        self.masked_l2_loss = self.masked_mse_loss
        self.custom_masked_l2_loss = self.custom_masked_mse_loss
        self.forward = getattr(self, loss_type)
    
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
        delta = (pred - gt) ** 2
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
        delta = (pred - gt) ** 2
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
        delta = (pred - gt) ** 2
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
        delta = torch.abs(pred - gt)
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
        delta = torch.abs(pred - gt)
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
        delta = torch.abs(pred - gt)
        mask_base = torch.sum(mask.float(), dim = [1, 2])
        mask_base = torch.where(mask_base < self.epsilon, mask_base + self.epsilon, mask_base)
        loss = torch.sum(delta * mask.float(), dim = [1, 2]) / mask_base
        return torch.mean(loss)
