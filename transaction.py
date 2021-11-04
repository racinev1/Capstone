#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 10:18:30 2021

@author: vincent
"""
import numpy as np
class TransactionCost:
    """
    Class to find the transaction cost using the bisection method.
    The transaction cost is define as the
    ratio of the portfolio value after transaction cost to the portfolio value
    before the transactions
    """
    def __init__(self, *,
                 selling_cost=5e-4, purchasing_cost=5e-4,
                 low=0, high=1, max_iter=100, tol=1e-8):
        """
        

        Parameters
        ----------
        * : TYPE
            DESCRIPTION.
        selling_cost : float, optional
            Selling transaction cost. The default is 5e-4.
        purchasing_cost : float, optional
            Purchase transaction cost. The default is 5e-4.
        low: float, optional
            Lower bound for the transaction cost. The default is 0.
        high: float, optional
            Upper bound for the transaction cost. The default is 1.
        max_iter: int, optional
            Maximum number of iteration
        tol: float, optional
            Value to consider objective as optimal.

        Returns
        -------
        None.

        """
        self.selling_cost = selling_cost
        self.purchasing_cost = purchasing_cost
        self.low = low
        self.high = high
        self.max_iter = max_iter
        self.tol = tol
    def __call__(self, current_weight, desired_weight):
        return self.bisection(current_weight, desired_weight)
    @staticmethod
    def objective(mu, current_weights, desired_weights, *,
                  selling_cost=5e-4, purchasing_cost=5e-4):
        """
        Computes the objective function. The optimal mu is found when the objective
        equals 0.

        Parameters
        ----------
        mu : float
            Variable that we desired to find. Corresponds to the ratio of the
            portfolio value after transaction cost to the value before the
            transaction
        current_weights : np.ndarray
            The current weight of the portfolio.
        desired_weights : np.ndarray
            The weights that we wish to rebalance to.
        * : TYPE
            DESCRIPTION.
        selling_cost : float, optional
            Selling transaction cost. The default is 5e-4.
        purchasing_cost : float, optional
            Purchasing transaction cost. The default is 5e-4.

        Returns
        -------
        obj : float
            The value of the objective function.

        """
        obj = mu - (1 - selling_cost*np.sum(np.max(current_weights - mu*desired_weights))
                    - purchasing_cost*np.sum(np.max(mu*desired_weights - current_weights)))
        return obj
    @staticmethod
    def samesign(x1, x2):
        """
        Helper function to determine if x1 and x2 are of the same sign.

        Parameters
        ----------
        x1 : float
            Arbitrary number.
        x2 : float
            Arbitrary number.

        Returns
        -------
        boolean
            True if x1 and x2 are of the same sign. False otherwise.

        """
        return x1 * x2 >= 0
    def bisection(self, current_weight, desired_weight):
        """
        Finds the the transaction cost.

        Parameters
        ----------
        current_weight : np.ndarray
            current weight of portfolio.
        desired_weight : np.ndarray
            desired weight of portfolio after transaction.

        Returns
        -------
        midpoint : float
            Transaction cost.

        """
        low = self.low
        high = self.high
        for _ in range(self.max_iter):
            midpoint = (high + low)/2
            low_func_val = self.objective(low, current_weight, desired_weight,
                                          selling_cost=self.selling_cost,
                                          purchasing_cost=self.purchasing_cost)
            func_val = self.objective(midpoint, current_weight, desired_weight,
                                      selling_cost=self.selling_cost,
                                      purchasing_cost=self.purchasing_cost)
            if np.abs(func_val) <= self.tol:
                return midpoint
            if self.samesign(low_func_val, func_val):
                low = midpoint
            else:
                high = midpoint
        return midpoint