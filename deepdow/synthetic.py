#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:54:55 2021

@author: vincent
"""

import numpy as np


def sin_single(n_timesteps, amplitude=1, freq=0.25, phase=0):
    """Generate sine waves.
    Parameters
    ----------
    n_timesteps : int
        Number of timesteps.
    amplitude : float
        The peak value.
    freq : float
        Frequency - number of oscillations per timestep.
    phase : float
        Offset.
    Returns
    -------
    y : np.ndarray
        1D array of shape `(n_timesteps,)`.
    """
    x = np.arange(n_timesteps)

    return amplitude * np.sin(2 * np.pi * freq * x + phase)
