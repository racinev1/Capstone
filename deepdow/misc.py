#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 14:28:06 2021

@author: vincent
"""
import torch.nn as nn
import torch

class CovarianceMatrix(nn.Module):
    """Covariance matrix or its square root.
    Parameters
    ----------
    sqrt : bool
        If True, then returning the square root.
    shrinkage_strategy : None or {'diagonal', 'identity', 'scaled_identity'}
        Strategy of combining the sample covariance matrix with some more stable matrix.
    shrinkage_coef : float or None
        If ``float`` then in the range [0, 1] representing the weight of the convex combination. If `shrinkage_coef=1`
        then using purely the sample covariance matrix. If `shrinkage_coef=0` then using purely the stable matrix.
        If None then needs to be provided dynamically when performing forward pass.
    """

    def __init__(self, sqrt=True, shrinkage_strategy='diagonal', shrinkage_coef=0.5):
        """Construct."""
        super().__init__()

        self.sqrt = sqrt

        if shrinkage_strategy is not None:
            if shrinkage_strategy not in {'diagonal', 'identity', 'scaled_identity'}:
                raise ValueError('Unrecognized shrinkage strategy {}'.format(shrinkage_strategy))

        self.shrinkage_strategy = shrinkage_strategy
        self.shrinkage_coef = shrinkage_coef

    def forward(self, x, shrinkage_coef=None):
        """Perform forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, dim, n_assets). The middle dimension `dim`
            represents the observations we compute the covariance matrix over.
        shrinkage_coef : None or torch.Tensor
            If None then using the `self.shrinkage_coef` supplied at construction for each sample. Otherwise a
            tensor of shape `(n_shapes,)`.
        Returns
        -------
        covmat : torch.Tensor
            Of shape (n_samples, n_assets, n_assets).
        """
        n_samples = x.shape[0]
        dtype, device = x.dtype, x.device

        if not ((shrinkage_coef is None) ^ (self.shrinkage_coef is None)):
            raise ValueError('Not clear which shrinkage coefficient to use')

        if shrinkage_coef is not None:
            shrinkage_coef_ = shrinkage_coef  # (n_samples,)
        else:
            shrinkage_coef_ = self.shrinkage_coef * torch.ones(n_samples, dtype=dtype, device=device)

        wrapper = self.compute_sqrt if self.sqrt else lambda h: h

        return torch.stack([wrapper(self.compute_covariance(x[i].T.clone(),
                                                            shrinkage_strategy=self.shrinkage_strategy,
                                                            shrinkage_coef=shrinkage_coef_[i]))
                            for i in range(n_samples)], dim=0)

    @staticmethod
    def compute_covariance(m, shrinkage_strategy=None, shrinkage_coef=0.5):
        """Compute covariance matrix for a single sample.
        Parameters
        ----------
        m : torch.Tensor
            Of shape (n_assets, n_channels).
        shrinkage_strategy : None or {'diagonal', 'identity', 'scaled_identity'}
            Strategy of combining the sample covariance matrix with some more stable matrix.
        shrinkage_coef : torch.Tensor
            A ``torch.Tensor`` scalar (probably in the range [0, 1]) representing the weight of the
            convex combination.
        Returns
        -------
        covmat_single : torch.Tensor
            Covariance matrix of shape (n_assets, n_assets).
        """
        fact = 1.0 / (m.size(1) - 1)
        m -= torch.mean(m, dim=1, keepdim=True)  # !!!!!!!!!!! INPLACE
        mt = m.t()

        s = fact * m.matmul(mt)  # sample covariance matrix

        if shrinkage_strategy is None:
            return s

        elif shrinkage_strategy == 'identity':
            identity = torch.eye(len(s), device=s.device, dtype=s.dtype)

            return shrinkage_coef * s + (1 - shrinkage_coef) * identity

        elif shrinkage_strategy == 'scaled_identity':
            identity = torch.eye(len(s), device=s.device, dtype=s.dtype)
            scaled_identity = identity * torch.diag(s).mean()

            return shrinkage_coef * s + (1 - shrinkage_coef) * scaled_identity

        elif shrinkage_strategy == 'diagonal':
            diagonal = torch.diag(torch.diag(s))

            return shrinkage_coef * s + (1 - shrinkage_coef) * diagonal

    @staticmethod
    def compute_sqrt(m):
        """Compute the square root of a single positive definite matrix.
        Parameters
        ----------
        m : torch.Tensor
            Tensor of shape `(n_assets, n_assets)` representing the covariance matrix - needs to be PSD.
        Returns
        -------
        m_sqrt : torch.Tensor
            Tensor of shape `(n_assets, n_assets)` representing the square root of the covariance matrix.
        """
        _, s, v = m.svd()

        good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
        components = good.sum(-1)
        common = components.max()
        unbalanced = common != components.min()
        if common < s.size(-1):
            s = s[..., :common]  # pragma: no cover
            v = v[..., :common]  # pragma: no cover
            if unbalanced:  # pragma: no cover
                good = good[..., :common]  # pragma: no cover
        if unbalanced:
            s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))  # pragma: no cover

        return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)
