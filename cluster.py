#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 09:55:49 2021

@author: vincent
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


class Cluster(AgglomerativeClustering):
    """
    Class to cluster similar stocks together
    """
    def __init__(self, n_clusters=2, *,
                 affinity='euclidean',
                 memory=None,
                 connectivity=None,
                 compute_full_tree='auto',
                 linkage='ward',
                 distance_threshold=None):
        """
        Parameters
        ----------
        see https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
        for details on the parameters

        Returns
        -------
        None.

        """
        super().__init__(n_clusters,
                         affinity=affinity,
                         memory=memory,
                         connectivity=connectivity,
                         compute_full_tree=compute_full_tree,
                         linkage=linkage,
                         distance_threshold=distance_threshold)
    def __call__(self, X):
        """
        Clusters stocks

        Parameters
        ----------
        X : array like of shape (N,N) or (N,T).
            Price of return of each stock

        Returns
        -------
        labels : np.ndarray
            The cluster for each stock with shape N.

        """
        if self.affinity == 'precomputed':
            dist = self.corr_distance(X)
            labels = super().fit_predict(dist)
        else:
            labels = super().fit_predict(X)
        return labels
    def plot_dendogram(self, X):
        """
        Function to plot the dendrogram

        Parameters
        ----------
        X : array like of shape (N,N) or (N,T).
            Price of return of each stock

        Returns
        -------
        None.

        """
        if self.affinity == 'precomputed':
            dist = self.corr_distance(X)
            link = sch.linkage(dist, method='single')
            sch.dendrogram(link)
            plt.show()
        else:
            raise NotImplementedError('plot_dendogram is only implemented for '
                                      'affinity=precomputed')
    @staticmethod
    def corr_distance(X):
        """
        Compute a proper distance metric based on the correlation.
        Parameters
        ----------
        X : array like of shape (N,N) or (N,T).
            Price of return of each stock

        Returns
        -------
        dist : np.ndarray
            Distance metric between each stock. shape=(N, N)

        """
        #compute correlation
        corr = np.corrcoef(X)
        #compute proper distance metric
        dist = np.sqrt((1 - corr)/2)
        return dist

            