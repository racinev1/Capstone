#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:26:29 2021

@author: vincent
"""
import cvxpy as cp
import numpy as np
import pandas as pd
from cluster import Cluster
from selection import argmax
from transaction import TransactionCost

class RollingBacktest:
    
    def __init__(self, *,
                 cardinality=20,
                 market_cardinality=500,
                 selection='mktcap',
                 weighting='vw',
                 look_ahead=0,
                 history=60,
                 max_look_ahead=12):
        """
        

        Parameters
        ----------
        * : TYPE
            DESCRIPTION.
        cardinality : int, optional
            Number of stocks to include in the portfolio. The default is 20.
        market_cardinality: int, optional
            Number of stocks in the market. The default is 500. 
        selection : str optional
            Selection policy. Must be one of ['mktcap', 'cluster'].
            The default is 'mktcap'.
        weighting : str, optional
            Weighting policy. Must be one of ['vw', 'ew', 'ow'] The default is 'vw'.
        look_ahead : int, optional
            The look-ahead period. The default is 0.
        history : int, optional
            The number of months to look back to make prediction. The default is 60.
        max_look_ahead : int, optional
            The maximum look ahead period so that each backtest is over the same period.
            The default is 12.

        Returns
        -------
        None.

        """
        self.cardinality = cardinality
        self.market_cardinality = market_cardinality
        self.selection = selection
        self.weighting = weighting
        self.look_ahead = look_ahead
        self.history = history
        self.max_look_ahead = max_look_ahead
        
    def rolling_backtest(self, train_df):
        """
        Performs a rolling backtest on train_df  

        Parameters
        ----------
        train_df : pd.DataFrame
            index=['date', 'PERMNO'], columns=['RET', 'LAGMKTCAP'].

        Returns
        -------
        results: pd.DataFrame
            index=date, columns=[ret, liquidity, turnover, transaction_cost]

        """
        portfolio_ret = []
        portfolio_turnover = []
        portfolio_transaction_cost = []
        portfolio_liquidity = []
        TC = TransactionCost()
        date_index = train_df.index.get_level_values('date').unique()
        for i, date in enumerate(date_index[self.history:-self.max_look_ahead]):
            j = i + self.history
            look_ahead_df = train_df.loc[date_index[j]: date_index[j + self.look_ahead]].copy()
            universe_df = train_df.loc[date].copy()
            history_df = self.find_history(universe_df, train_df, date_index[i],
                                           date, market_cardinality=self.market_cardinality)
            #only consider stocks that have data over the history period
            current_df = universe_df.loc[history_df.index].copy()
            selected_stocks = self.stock_selection(current_df, history_df,
                                                   cardinality=self.cardinality,
                                                   selection=self.selection)
            suggested_w = self.weighting_policiy(current_df, history_df, selected_stocks,
                                                     weighting=self.weighting)
            if i == 0:
                current_w = suggested_w.copy()
                current_w.name = 'current_w'
            weights = pd.merge(suggested_w, current_w, how='outer',
                               left_index=True, right_index=True)
            weights.replace(np.nan, 0, inplace=True)
            mu = TC(weights['current_w'].values, weights['suggested_w'].values)
            rebalance = self.oracle(look_ahead_df, suggested_w, current_w, mu)
            if rebalance:
                w = suggested_w.copy()
                portfolio_turnover.append(0.5*np.sum(np.abs(
                    weights['current_w'] - weights['suggested_w'])))
                portfolio_transaction_cost.append(1 - mu)
            else:
                w = current_w.copy()
                portfolio_turnover.append(0)
                portfolio_transaction_cost.append(0)
            portfolio_liquidity.append(self.find_portfolio_liquidity(universe_df, w,
                                                                 market_cardinality=self.market_cardinality))
            portfolio = universe_df.loc[w.index]
            portfolio_ret.append(np.average(portfolio['RET'], weights=w))
            current_w = w*(1 + portfolio['RET'])/np.sum(w*(1 + portfolio['RET']))
            current_w.name = 'current_w'
        results = pd.DataFrame({'ret': portfolio_ret, 'turnover': portfolio_turnover,
                                'liquidity': portfolio_liquidity,
                                'transaction_cost': portfolio_transaction_cost},
                               index=date_index[self.history:-self.max_look_ahead])
        return results
            
    @staticmethod  
    def find_portfolio_liquidity(current_df, w, market_cardinality=500):
        """
        Computes the portfolio liquidity as defined in Pastor 2020
    
        Parameters
        ----------
        current_df : pd.DataFrame
            .
        w : pd.Series
            weights of portfolio.
        market_cardinality : int, optional
            The number of stocks in the portfolio. The default is 500.
    
        Returns
        -------
        liquidity : float
            Portfolio liquidity.
    
        """
        universe = current_df.nlargest(market_cardinality, 'LAGMKTCAP')['LAGMKTCAP']
        universe_weights = universe/np.sum(universe)
        port_weights = pd.Series(np.zeros(len(universe)), index=universe.index)
        try:
            port_weights.loc[w.index] = w
        except KeyError:
            universe_set = set(universe.index)
            portfolio_set = set(w.index)
            common_index = np.asarray(list(universe_set.intersection(portfolio_set)))
            port_weights.loc[common_index] = w.loc[common_index]
        illiquidity = np.sum(np.square(port_weights)/universe_weights)
        liquidity = 1/illiquidity
        return liquidity
    @staticmethod 
    def oracle(look_ahead_df, suggested_w, current_w, mu):
        """
        Oracle to determin if it is optimal to rebalance

        Parameters
        ----------
        look_ahead_df : pd.DataFrame
            DESCRIPTION.
        suggested_w : pd.Series
            DESCRIPTION.
        current_w : pd.Series
            DESCRIPTION.
        mu : float
            Transaction cost.

        Returns
        -------
        bool
            True if you should reblance. False otherwise.

        """
        try:
            current_portfolio_return = compute_return(look_ahead_df, current_w)
        except KeyError:
            current_portfolio_return = -1000
        try:
            suggested_portfolio_return = compute_return(look_ahead_df, suggested_w)
        except KeyError:
            suggested_portfolio_return = -1000
        if current_portfolio_return >= mu * suggested_portfolio_return:
            return False
        return True

    @staticmethod 
    def find_history(current_df, train_df, history_date, date, market_cardinality=500):
        """
        

        Parameters
        ----------
        current_df : pd.DataFrame
            index=['date', 'PERMNO'], columns=['RET', 'LAGMKTCAP'] at current date                
        train_df : pd.DataFrame
            index=['date', 'PERMNO'], columns=['RET', 'LAGMKTCAP']
        history_date : pd.Timestamp
            Date corresponding to the start of the history period.
        date : pd.Timestamp
            Current date.
        market_cardinality : int, optional
            Number of stocks in the universe. The default is 500.

        Returns
        -------
        history_df : pd.DataFrame
            pivot table of return with index='PERMNO' and columns='date'.

        """
        #find the stocks in the universe
        universe_stock = current_df.nlargest(market_cardinality, 'LAGMKTCAP').index
        #obtain the return from history_date:date
        history_df = train_df.loc[history_date:date, 'RET'].copy()
        #make a pivot table with index='PERMNO' and columns='date
        history_df = pd.DataFrame(history_df).reset_index()
        history_df = history_df.pivot(index='PERMNO', columns='date', values='RET')
        #Keep only stocks in the universe
        history_df = history_df.loc[universe_stock].copy()
        #drop stock with missing history
        history_df.dropna(axis=0, inplace=True)
        return history_df
    @staticmethod 
    def stock_selection(current_df,
                        history_df,
                        cardinality=20,
                        selection='mktcap'):
        """
        Function to select stocks

        Parameters
        ----------
        current_df : TYPE
            DESCRIPTION.
        history_df : TYPE
            DESCRIPTION.
        cardinality : TYPE, optional
            DESCRIPTION. The default is 20.
        selection : TYPE, optional
            DESCRIPTION. The default is 'mktcap'.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        selected_stocks : TYPE
            DESCRIPTION.

        """
        if selection not in ['mktcap', 'cluster']:
            raise NotImplementedError(f'selection={selection} is not in '
                                      '["mktcap", "cluster"]')
        if selection == 'mktcap':
            selected_stocks = current_df.nlargest(cardinality, 'LAGMKTCAP').index.values
        else:
            clustering = Cluster(n_clusters=cardinality,
                                 affinity='precomputed', 
                                 linkage='complete')
            labels = clustering(history_df)
            labels = pd.Series(labels, index=history_df.index)
            criterion = current_df['LAGMKTCAP']
            selected_stocks = argmax(labels, criterion)
            selected_stocks = selected_stocks.values
        return selected_stocks
    @staticmethod
    def weighting_policiy(current_df,
                          history_df,
                          selected_stocks,
                          weighting='vw'):
        """
        Function to weight stocks

        Parameters
        ----------
        current_df : TYPE
            DESCRIPTION.
        history_df : TYPE
            DESCRIPTION.
        selected_stocks : TYPE
            DESCRIPTION.
        weighting : TYPE, optional
            DESCRIPTION. The default is 'vw'.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        suggested_w : TYPE
            DESCRIPTION.

        """
        if weighting not in ['vw', 'ew', 'ow']:
            raise NotImplementedError(f'weighting={weighting} is not in '
                                      '["vw", "ew", "ow"]')
        if weighting == 'vw':
            suggested_w = current_df.loc[selected_stocks, 'LAGMKTCAP']
            suggested_w = suggested_w / np.sum(suggested_w)
        elif weighting == 'ew':
            suggested_w = pd.Series(1/len(selected_stocks), index=selected_stocks)
        else:
            data = history_df.loc[selected_stocks].copy()
            cov = np.cov(data)
            optimal_weight = minimum_variance(cov)
            suggested_w = pd.Series(optimal_weight, index=selected_stocks)
        suggested_w.name = 'suggested_w'
        return suggested_w
def minimum_variance(cov):
    """
    Finds the minimum variance portfolio weights

    Parameters
    ----------
    cov : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    _n = cov.shape[0]
    _l = (1/_n) - 0.5*(1/_n)
    _u = (1/_n) + 0.5*(1/_n)
    _x = cp.Variable(shape=_n)
    obj = cp.Minimize(cp.quad_form(_x, cov))
    constraints = [cp.sum(_x) == 1]
    constraints += [_x >= _l]
    constraints += [_x <= _u]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return _x.value
def compute_return(look_ahead_df, weights):
    """
    Computes the return of a portfolio specificed by weights over the look ahead period.

    Parameters
    ----------
    look_ahead_df : TYPE
        DESCRIPTION.
    weights : TYPE
        DESCRIPTION.

    Returns
    -------
    port_ret : TYPE
        DESCRIPTION.

    """
    date_index = look_ahead_df.index.get_level_values('date').unique()
    port_ret = 1
    for date in date_index:
        portfolio = look_ahead_df.loc[date]
        portfolio = portfolio.loc[weights.index]
        ret = np.average(portfolio['RET'], weights=weights)
        port_ret *= (1 + ret)
        weights = weights * (1 + portfolio['RET'])/np.sum(weights*(1 + portfolio['RET']))
    return port_ret
        
        