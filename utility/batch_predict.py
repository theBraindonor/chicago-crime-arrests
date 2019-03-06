#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Functions for calling estimator predictions in batches
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

import numpy as np


def batch_predict(estimator, x, transformer=None, increment=500, verbose=True):
    y = None
    if len(x) > increment:
        step_size = len(x) // increment
    else:
        step_size = 1
    steps = 0
    for chunk in np.array_split(x, step_size):
        if verbose:
            print('.', end='')
            if steps > 0 and steps % 100 == 0:
                print(steps*increment)
            steps += 1
        if transformer is not None:
            chunk = transformer.transform(chunk)
        y_chunk = estimator.predict(chunk)
        if y is None:
            y = y_chunk
        else:
            y = np.concatenate((y, y_chunk), axis=0)
    if verbose:
        print('')
    return y


def batch_predict_proba(estimator, x, threshold=None, transformer=None, increment=500, verbose=True):
    y = None
    if len(x) > increment:
        step_size = len(x) // increment
    else:
        step_size = 1
    steps = 0
    for chunk in np.array_split(x, step_size):
        if verbose:
            print('.', end='')
            if steps > 0 and steps % 100 == 0:
                print(steps*increment)
            steps += 1
        if transformer is not None:
            chunk = transformer.transform(chunk)
        y_chunk = estimator.predict_proba(chunk)
        if threshold is not None:
            y_chunk = (y_chunk[:, 1] >= threshold).astype(bool)
        if y is None:
            y = y_chunk
        else:
            y = np.concatenate((y, y_chunk), axis=0)
    if verbose:
        print('')
    return y

