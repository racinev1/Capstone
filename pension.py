#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:28:10 2021

@author: vincent
"""
import numpy as np

def find_payment(income,
                 amount_in_portfolio,
                 work_time,
                 retirement_time,
                 *,
                 r_growth=1.06**(1/12),
                 r_safe=1.03**(1/12),
                 inflation=1.03**(1/12)):
    """
    Finds the required monthly payment to achieve given retirement goal

    Parameters
    ----------
    income : int/float
        Desired monthly income during retirement. Should correspond to the value
        desired at the first month of retirement.
    amount_in_portfolio : int/float
        Current amount accumulated in portfolio.
    work_time : int
        Number of months remaining before retirement.
    retirement_time : int
        Number of months expected in retirement.
    * : TYPE
        DESCRIPTION.
    r_growth : float, optional
        Expected monthly growth rate of the portfolio during work period.
        The default is 1.06**(1/12).
    r_safe : float, optional
        Expected monthly growth rate of the portfolio during retirement period.
        The default is 1.03**(1/12).
    inflation : float, optional
        The inflation rate. The default is 1.03**(1/12).

    Returns
    -------
    monthly_payment : float
        The required monthly payment to achieve given investment goal.

    """
    desired_npv_at_retirement = np.sum([income*(inflation/r_safe)**t
                                       for t in range(retirement_time)])
    monthly_payment = ((desired_npv_at_retirement -amount_in_portfolio*r_growth**work_time)
                       /np.sum([r_growth**t for t in range(work_time)]))
    return monthly_payment
def find_income(payments,
                amount_in_portfolio,
                work_time,
                retirement_time,
                *,
                r_growth=1.06**(1/12),
                r_safe=1.03**(1/12),
                inflation=1.03**(1/12)):
    """
    Finds the monthly income given monthly payments

    Parameters
    ----------
    payments : None or array-like
        Monthly payments during work time. None corresponds to no additional payments.
        If array-like should of shape (work_time, )
    amount_in_portfolio : int/float
        Current amount accumulated in portfolio.
    work_time : int
        Number of months remaining before retirement.
    retirement_time : int
        Number of months expected in retirement.
    * : TYPE
        DESCRIPTION.
    r_growth : float, optional
        Expected monthly growth rate of the portfolio during work period.
        The default is 1.06**(1/12).
    r_safe : float, optional
        Expected monthly growth rate of the portfolio during retirement period.
        The default is 1.03**(1/12).
    inflation : float, optional
        The inflation rate. The default is 1.03**(1/12).

    Returns
    -------
    monthly_income : float
        The monthly income achieved with current investment plan. This corresponds to the
        first monthly payment

    """ 
    if not payments:
        npv_at_retirement = amount_in_portfolio ** work_time
    else:
        npv_at_retirement = amount_in_portfolio + np.sum([payments[t] * r_growth ** t
                                                          for t in range(work_time)])
    monthly_income = npv_at_retirement / np.sum([(inflation/r_safe)**t
                                         for t in range(retirement_time)])
    return monthly_income
    
                                         
                                         
    