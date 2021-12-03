#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 09:48:47 2021

@author: vincent
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from benchmarks import Benchmark
from misc import CovarianceMatrix


class Boyd_MPO(nn.Module):
    
    def __init__(self,
                 n_assets=20,
                 horizon=6,
                 risk_aversion=1,
                 transaction_cost=1,
                 min_weight=0,
                 max_weight=1,
                 ):
        """
        

        Parameters
        ----------
        n_assets : int, optional
            Number of assets. The default is 20.
        horizon : int, optional
            Lookforward horizon. The default is 6.
        risk_aversion : int/float, optional
            risk aversion parameter The default is 1.
        transaction_cost : int/float, optional
            transaction cost. The default is 1.
        min_weight: float, optional
            minimum allowed weight. The default is 0.
        max_weight: float, optional
            maximum allowed weight. The default is 1.

        Returns
        -------
        None.

        """
        super().__init__()
        
        rets = cp.Parameter(shape=(n_assets, horizon))
        covmat = cp.Parameter(shape=(n_assets, n_assets))
        prev_w = cp.Parameter(shape=n_assets)
        
        w = cp.Variable(shape=(n_assets, horizon))
        
        port_ret = cp.sum([w[:, t].T @ rets[:, t] for t in range(horizon)])
        risks = cp.sum(cp.sum_squares(w.T @ covmat))
        transaction = cp.sum([cp.sum(cp.abs(w[:, t] - prev_w)) if t==0 else cp.sum(cp.abs(w[:, t] - w[:, t-1]))
                              for t in range(horizon)])
        
        obj = cp.Maximize(port_ret - risks - transaction)
        constraints = [cp.sum(w, axis=0) == 1, w>=min_weight, w<=max_weight]
        
        prob = cp.Problem(obj, constraints)
        
        self.cvxpy_layer = CvxpyLayer(prob, [rets, covmat, prev_w], [w])
        
    def forward(self, rets, covmat, prev_w):
        """
        

        Parameters
        ----------
        rets : torch.Tensor
            Of shape (n_assets, horizon) corresponding to the expected return.
        covmat : torch.Tensor
            Of shape (n_assets, n_assets) corresponding to the covariance matrix.
        prev_w : torch.Tensor
            Of shape (n_assets) corresponding to the weight of the last time period.

        Returns
        -------
        weights: torch.Tensor
            Predicted optimal weight of shape=(n_assets, horizon).

        """
        
        return self.cvxpy_layer(rets, covmat, prev_w)[0]
    
class IEEE_MPO(nn.Module, Benchmark):
    
    def __init__(self, *,
                 n_input_channels=1,
                 hidden_channels=20,
                 horizon=6,
                 kernel_size=3,
                 lookback=36,
                 n_assets=20,
                 risk_aversion=1,
                 transaction_cost=1,
                 min_weight=0,
                 max_weight=1):
        """
        

        Parameters
        ----------
        * : TYPE
            DESCRIPTION.
        n_input_channels : int, optional
            Number of features. The default is 1.
        hidden_channels : int, optional
            Number of features map. The default is 20.
        horizon : int, optional
            The horizon period. The default is 6.
        kernel_size : int, optional
            Size of the 1-D convolution kernel. The default is 3.
        lookback : int, optional
            The lookback period. The default is 36.
        n_assets : int, optional
            Number of assets in portfolio. The default is 20.
        risk_aversion : int/float, optional
            risk aversion parameter. The default is 1.
        transaction_cost : int/float, optional
            transaction costs. The default is 1.
        min_weight : float, optional
            Minimum portfolio weight. The default is 0.
        max_weight : float, optional
            Maximum portfolio weight. The default is 1.

        Returns
        -------
        None.

        """
        super().__init__()
        self.covariance_layer = CovarianceMatrix()
        self.norm_layer = nn.InstanceNorm2d(n_input_channels, affine=True)
        self.cnn1 = nn.Conv1d(in_channels=n_input_channels, out_channels=hidden_channels, kernel_size=kernel_size)
        kernel_size = lookback - kernel_size + 1
        self.cnn2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size)
        self.linear = nn.Linear(in_features=hidden_channels, out_features=horizon)
        self.activation = nn.Tanh()
        self.allocate_layer = Boyd_MPO(n_assets=n_assets, horizon=horizon, risk_aversion=risk_aversion,
                                       transaction_cost=transaction_cost, min_weight=min_weight, max_weight=max_weight)
    
    def forward(self, x, prev_w):
        """
        

        Parameters
        ----------
        x : torch.Tensor
            Of shape (batch_size, n_channels, lookback, n_assets) corresponding to the features.
            It is assume that channels 0 is the returns. 
        prev_w : torch.Tensor
            Of shape (batch_size, n_assets) corresponding to the previous weights.

        Returns
        -------
        weights : torch.Tensor
            Of shape (batch_size, horizon, n_samples) corresponding to the predicted optimal weights.

        """
        
        x = self.norm_layer(x)
        
        #covariance matrix
        rets = x[:, 0, :, :]
        covmat = self.covariance_layer(rets)
        
        #expected returns
        
        batch_size, n_channels, lookback, n_assets = x.shape
        x = self.norm_layer(x)
        exp_rets_list = []
        for i in range(n_assets):
            z = x[:, :, :, i] #(batch_size, n_channels, lookback)
            z = self.cnn1(z) #(batch_size, hidden_channels, 32)
            z = F.leaky_relu(z)
            z = self.cnn2(z) #(batch_size, hidden_channels, 1)
            z = torch.flatten(z, start_dim=1) #(batch_size, hidden_channels)
            z = F.leaky_relu(z)
            z = self.linear(z) #(batch_size, horizon)
            z = self.activation(z)
            exp_rets_list.append(z)
        _, horizon = z.shape
        exp_rets = torch.cat(exp_rets_list).reshape(n_assets, batch_size, horizon)
        exp_rets = exp_rets.permute(1, 0, 2)  #(batch_size, n_assets, horizon)
        
        weights = self.allocate_layer(exp_rets, covmat, prev_w) #(batch_size, n_sample, horizon)
        weights = weights.permute(0, 2, 1) #(batch_size, horizon, n_sample)
        
        return weights
        
        