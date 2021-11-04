#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 08:49:15 2021

@author: vincent
"""
import numpy as np

def geometric_return(ret, fees=0.0, n_years=10):
    """
    Function to compute the geometric average return

    Parameters
    ----------
    ret : array-like
        fund return.
    fees: float, optional
        fees charged by the fund. The default is zero.
    n_years : int, optional
        Number of years in which the return are obtained. The default is 10.

    Returns
    -------
    float
        Geometric average return.

    """
    
    return np.prod((ret + 1)*(1-fees))**(1/n_years) - 1
def semi_variance(ret):
    """
    Function to compute the semi-variance

    Parameters
    ----------
    ret : array-like
        fund-return.

    Returns
    -------
    float.
        Portfolio Semi-Variance

    """
    return np.mean(np.min(ret, 0)**2)
def maximum_drawdown(ret):
    """
    Function to compute the maximum drawdown

    Parameters
    ----------
    ret : array-like
        fund return.

    Returns
    -------
    float.
        Maximum Drawdown

    """
    price = np.cumprod(ret + 1)
    roll_max = price.cummax()
    drawdown = price/roll_max - 1
    max_drawdown = np.max(-drawdown)
    return max_drawdown
def find_aum(liquidity, turnover, fees):
    """
    Find the portfolio AUM based on Pastor model

    Parameters
    ----------
    liquidity : array-like
        Portfolio liquidity.
    turnover : array-like
        Portfolio turnover.
    fees : float
        Fees charged to clients.

    Returns
    -------
    aum: float
        The portfolio AUM.

    """
    ln_aum = 8.065*np.log(liquidity) - 5.065*np.log(fees) - 0.815*np.log(turnover)
    aum = np.exp(ln_aum)
    return aum