"""
Define criterions.

Authors: Hongjie Fang.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.functions import get_surface_normal_from_depth


class Criterion(nn.Module):
    """
    Various type of criterions.
    """
    def __init__(self, type, combined_smooth = False, **kwargs):
        super(Criterion, self).__init__()
        self.epsilon = kwargs.get('epsilon', 1e-8)
        self.type = str.lower(type)
        if 'huber' in self.type:
            self.huber_k = kwargs.get('huber_k', 0.1)
        self.combined_smooth = combined_smooth
        if combined_smooth:
            self.combined_beta = kwargs.get('combined_beta', 0.5)
        self.l2_loss = self.mse_loss
        self.masked_l2_loss = self.masked_mse_loss
        self.custom_masked_l2_loss = self.custom_masked_mse_loss
        self.main_loss = getattr(self, type)
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
    
    def mse_loss(self, data_dict, *args, **kwargs):
        """
        MSE loss.

        Parameters
        ----------
        
        data_dict: the data dict for computing L2 loss.

        Returns
        -------

        The MSE loss.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['zero_mask']
        return self._l2(pred, gt)[mask].mean()
    
    def masked_mse_loss(self, data_dict, *args, **kwargs):
        """
        Masked MSE loss.
        
        Parameters
        ----------
        
        data_dict: the data dict for computing L2 loss.

        Returns
        -------

        The masked MSE loss.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['initial_loss_mask']
        return self._l2(pred, gt)[mask].mean()
    
    def custom_masked_mse_loss(self, data_dict, *args, **kwargs):
        """
        Custom masked MSE loss.
        
        Parameters
        ----------
        
        data_dict: the data dict for computing L2 loss.

        Returns
        -------

        The custom masked MSE loss.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['loss_mask']
        return self._l2(pred, gt)[mask].mean()
    
    def l1_loss(self, data_dict, *args, **kwargs):
        """
        L1 loss.

        Parameters
        ----------
        
        data_dict: the data dict for computing L1 loss.

        Returns
        -------

        The L1 loss.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['zero_mask']
        return self._l1(pred, gt)[mask].mean()
    
    def masked_l1_loss(self, data_dict, *args, **kwargs):
        """
        Masked L1 loss.
        
        Parameters
        ----------
        
        data_dict: the data dict for computing L1 loss.

        Returns
        -------

        The masked L1 loss.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['initial_loss_mask']
        return self._l1(pred, gt)[mask].mean()
    
    def custom_masked_l1_loss(self, data_dict, *args, **kwargs):
        """
        Custom masked L1 loss.
        
        Parameters
        ----------

        data_dict: the data dict for computing L1 loss.

        Returns
        -------

        The custom masked L1 loss.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['loss_mask']
        return self._l1(pred, gt)[mask].mean()
    
    def huber_loss(self, data_dict, *args, **kwargs):
        """
        Huber loss.

        Parameters
        ----------
        
        data_dict: the data dict for computing huber loss.

        Returns
        -------

        The huber loss.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['zero_mask']
        return self._huber(pred, gt)[mask].mean()

    def masked_huber_loss(self, data_dict, *args, **kwargs):
        """
        Masked Huber loss.
        
        Parameters
        ----------
        
        data_dict: the data dict for computing huber loss.

        Returns
        -------

        The masked huber loss.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['initial_loss_mask']
        return self._huber(pred, gt)[mask].mean()
    
    def custom_masked_huber_loss(self, data_dict, *args, **kwargs):
        """
        Custom masked huber loss.
        
        Parameters
        ----------
        
        data_dict: the data dict for computing huber loss.

        Returns
        -------

        The custom masked huber loss.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['loss_mask']
        return self._huber(pred, gt)[mask].mean()
    
    def smooth_loss(self, data_dict, *args, **kwargs):
        """
        Smooth loss: surface normal loss.

        Parameters
        ----------
        
        data_dict: the data dict for computing smooth loss.

        Returns
        -------
        
        The smooth loss.
        """
        # Fetch information from data dict
        pred = data_dict['pred']
        fx, fy, cx, cy = data_dict['fx'], data_dict['fy'], data_dict['cx'], data_dict['cy']
        depth_gt_sn = data_dict['depth_gt_sn']
        _, original_h, original_w = data_dict['depth_original'].shape
        mask = data_dict['loss_mask_dilated']
        # Calculate smooth loss.
        pred_sn = get_surface_normal_from_depth(pred, fx, fy, cx, cy, original_size = (original_w, original_h))
        sn_loss = 1 - F.cosine_similarity(pred_sn, depth_gt_sn, dim = 1)
        # masking
        return sn_loss[mask].mean()

    def forward(self, data_dict):
        """
        Calculate criterion given data dict.
        
        Parameters
        ----------
        
        data_dict: the data dict for computing loss.

        Returns
        -------
        
        The pre-defined loss.
        """
        loss_dict = {
            self.type: self.main_loss(data_dict)
        }
        if self.combined_smooth:
            loss_dict['smooth'] = self.smooth_loss(data_dict)
            loss_dict['loss'] = loss_dict[self.type] + self.combined_beta * loss_dict['smooth']
        else:
            loss_dict['loss'] = loss_dict[self.type]
        return loss_dict
