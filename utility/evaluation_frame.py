#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The evaluation frame allows for easy pickling of the test values and predicted values.  This allows for
    easy scoring without having to refit model or to pickle the model itself.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

import pickle


class EvaluationFrame:
    def __init__(self, y_actual, y_predict):
        self._y_actual = y_actual
        self._y_predict = y_predict

    @property
    def y_actual(self):
        return self._y_actual

    @y_actual.setter
    def y_actual(self, value):
        self._y_actual = value

    @property
    def y_predict(self):
        return self._y_predict

    @y_predict.setter
    def y_predict(self, value):
        self._y_predict = value

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
