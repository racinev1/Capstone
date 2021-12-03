#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 09:34:27 2021

@author: vincent
"""

import numpy as np
import pandas as pd
import optuna
import torch.optim as optim
from sklearn.model_selection import KFold

from data import RigidDataLoader
from experiments import Run
from nn import IEEE_MPO
from visualize import generate_metrics_table



class HPTuning:
    """class for HP tuning on training set"""
    
    def __init__(self, dataset, indices, loss, *,
                 n_trials = 50,
                 n_input_channels=1,
                 n_assets=20,
                 horizon=6,
                 lookback=36,
                 n_epochs=5,
                 batch_size=32,
                 n_splits=3,
                 callbacks=None):
        
        self.dataset = dataset
        self.indices = indices
        self.loss = loss
        self.n_trials = n_trials
        self.n_input_channels = n_input_channels
        self.n_assets = n_assets
        self.horizon = horizon
        self.lookback = lookback
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.callbacks = callbacks
        
    def __call__(self):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        trial = study.best_trial
        
        return trial.params
        
        
    
    def define_model(self, trial):
        
        hidden_channels = trial.suggest_int("hidden_channels", 1, 20)
        kernel_size = trial.suggest_int("kernel_size", 3, 12)
        risk_aversion = trial.suggest_float("risk_aversion", 1e-5, 1000, log=True)
        transaction_cost = trial.suggest_float("transaction_cost", 1e-5, 1, log=True)
        min_weight = trial.suggest_float("min_weight", 0, 1/self.n_assets)
        max_weight = trial.suggest_float("max_weight", 1/self.n_assets, 1)
        
        model = IEEE_MPO(n_input_channels=self.n_input_channels,
                         hidden_channels=hidden_channels,
                         horizon=self.horizon,
                         kernel_size=kernel_size,
                         lookback=self.lookback,
                         n_assets=self.n_assets,
                         risk_aversion=risk_aversion,
                         transaction_cost=transaction_cost,
                         min_weight=min_weight,
                         max_weight=max_weight)
        return model
    
    def objective(self, trial):
        

        
        # Generate the optimizers.
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        
        
        scores = []
        for indices_train, indices_test in self.cross_validate(self.indices, self.n_splits):
            #Generate the neural network
            network = self.define_model(trial)
            network = network.train()
            optimizer = optim.Adam(network.parameters(), lr=lr)
            index = np.concatenate([indices_train, 
                                    np.asarray([indices_train[-1] + 1])])
            PVM = pd.DataFrame(1/self.n_assets, index = index, columns=np.arange(self.n_assets))
            dataloader_train = RigidDataLoader(self.dataset,
                                               indices=indices_train,
                                               batch_size=self.batch_size)
            
            dataloader_test = RigidDataLoader(self.dataset,
                                              indices=indices_test,
                                              batch_size=self.batch_size)
            
            run = Run(network,
                      self.loss,
                      dataloader_train,
                      PVM,
                      val_dataloaders={'test': dataloader_test},
                      optimizer=optimizer,
                      callbacks=self.callbacks)
            
            _ = run.launch(self.n_epochs)
            
            benchmarks = {'network': network}
            metrics = {'loss': self.loss}
            
            index = np.concatenate([np.asarray(indices_test),
                                    np.asarray([indices_test[-1] + 1])])
            
            #Initialize all PVM to equal weight portfolios
            PVM_dict = {b_name: pd.DataFrame(1/self.n_assets, index=index, columns=np.arange(self.n_assets))
                        for b_name in benchmarks}
            
            metrics_table = generate_metrics_table(benchmarks,
                                                   dataloader_test,
                                                   metrics,
                                                   PVM_dict)
            
            scores.append(metrics_table['value'].mean())
            
        return np.mean(scores)
            
            
        
    @staticmethod 
    def cross_validate(indices, n_splits):
        kfold = KFold(n_splits=n_splits)
        for indices_train, indices_test in kfold.split(indices):
            yield indices_train, indices_test
        
        
        
        
        
        
        