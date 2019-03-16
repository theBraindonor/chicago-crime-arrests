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


def batch_fit_classifier(estimator, x, y, transformer=None, increment=500, verbose=True):
    if len(x) > increment:
        step_size = len(x) // increment
    else:
        step_size = 1
    steps = 0
    x_chunks = np.array_split(x, step_size)
    y_chunks = np.array_split(y, step_size)
    classes = np.unique(y)
    for i in range(len(x_chunks)):
        if verbose:
            print('.', end='')
            if i > 0 and i % 100 == 0:
                print(i*increment)
        x_chunk = x_chunks[i]
        y_chunk = y_chunks[i]
        if transformer is not None:
            x_chunk = transformer.transform(x_chunk)
        estimator.partial_fit(x_chunk, y_chunk, classes=classes)
    if verbose:
        print('')

