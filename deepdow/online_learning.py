#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:22:22 2021

@author: vincent
"""
import numpy as np
import pandas as pd
import torch


from experiments import Run
from data import RigidDataLoader

class OnlineLearning:
    
    def __init__(self,
                 network,
                 dataset,
                 indices,
                 optimizer,
                 loss,
                 *,
                 retrain_period=12,
                 n_epochs=20,
                 batch_size=4,
                 n_assets=20,
                 callbacks=None):
        self.network = network
        self.dataset = dataset
        self.indices = indices
        self.optimizer = optimizer
        self.loss = loss
        self.retrain_period = retrain_period
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_assets = n_assets
        self.callbacks = callbacks
        
    
    def __call__(self):
        
        splits = self.split_given_size(self.indices, self.retrain_period)
        
        optimal_weights = pd.DataFrame(0, index=self.indices, columns=np.arange(self.n_assets))
        
        for idx in splits:
            
            dataloader = RigidDataLoader(self.dataset,
                                         indices=idx,
                                         batch_size=self.batch_size)
            index = np.concatenate([idx, np.asarray([idx[-1] + 1])])
            PVM = pd.DataFrame(1/self.n_assets, index=index, columns=np.arange(self.n_assets))
            
            self.network.eval()
            
            #evalute on actual test set
            for batch_ix, (X_batch, y_batch, timestamps, _) in enumerate(dataloader):
                
                prev_w = torch.FloatTensor(PVM.loc[timestamps].values)
                
                weights = self.network(X_batch.float(), prev_w)
                
                actions = weights[:, 0, :].detach().cpu().numpy()
                
                PVM.loc[np.asarray(timestamps) + 1] = actions
            
            #update weights
            optimal_weights.loc[idx] = PVM.loc[idx + 1].values
            
            #reinitialize PVM for training purpose
            PVM = pd.DataFrame(1/self.n_assets, index=index, columns=np.arange(self.n_assets))
            #update model based on observation from last period
            self.network.train()
            
            run = Run(self.network,
                      self.loss,
                      dataloader,
                      PVM,
                      optimizer=self.optimizer,
                      callbacks=self.callbacks)
            
            run.launch(self.n_epochs)
            
            
        
        return optimal_weights 
        
        
    @staticmethod
    def split_given_size(array, size):
        return np.split(array, np.arange(size, len(array), size))
