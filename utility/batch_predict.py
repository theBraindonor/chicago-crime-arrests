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


def batch_predict(estimator, x):
    y = None
    for chunk in np.array_split(x, 100):
        y_chunk = estimator.predict(chunk)
        if y is None:
            y = y_chunk
        else:
            y = np.concatenate((y, y_chunk), axis=0)
    return y


def batch_predict_proba(estimator, x):
    y = None
    for chunk in np.array_split(x, 100):
        y_chunk = estimator.predict_proba(chunk)
        if y is None:
            y = y_chunk
        else:
            y = np.concatenate((y, y_chunk), axis=0)
    return y
