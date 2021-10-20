#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 09:56:36 2021

@author: vincent
"""

import numpy as np
import pandas as pd


def argmax(labels, criterion):
    """
    For each cluster finds the stock with the highest criterion within that
    cluster

    Parameters
    ----------
    labels : pd.Series
        Labels identifying the cluster that each stock .
    criterion : pd.Series
        Criterion to use to select the stock within a cluster

    Returns
    -------
    selected_stocks : pd.Series
        The selected stocks. The index corresponds to the cluster id and the
        value corresponds to the stock selected for that cluster

    """
    cluster_id = labels.unique()
    selected_stocks = pd.Series(index=cluster_id)
    for cluster in cluster_id:
        stocks_in_cluster = labels.loc[labels == cluster].index
        criterion_cluster = criterion.loc[stocks_in_cluster]
        selected_stocks.loc[cluster] = criterion_cluster.idxmax()
    selected_stocks.sort_index(inplace=True)
    return selected_stocks
