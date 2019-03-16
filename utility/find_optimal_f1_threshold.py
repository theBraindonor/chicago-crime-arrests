#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Find an optimal f1 score for a classifier.  Used as an exercise to explore prediction thresholds.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

from sklearn.metrics import f1_score


def find_optimal_f1_threshold(frame):
    y_actual = frame.y_actual
    y_predict_proba = frame.y_predict

    best_threshold_score = 0
    best_threshold = 0
    for i in range(0, 9):
        threshold = 0.1 + (i/10)
        y_predict = (y_predict_proba[:, 1] >= threshold).astype(bool)
        threshold_score = f1_score(y_actual, y_predict)
        if threshold_score > best_threshold_score:
            best_threshold_score = threshold_score
            best_threshold = threshold

    for i in range(0, 19):
        threshold = best_threshold + ((i - 9)/100)
        y_predict = (y_predict_proba[:, 1] >= threshold).astype(bool)
        threshold_score = f1_score(y_actual, y_predict)
        if threshold_score > best_threshold_score:
            best_threshold_score = threshold_score
            best_threshold = threshold

    for i in range(0, 19):
        threshold = best_threshold + ((i - 9)/1000)
        y_predict = (y_predict_proba[:, 1] >= threshold).astype(bool)
        threshold_score = f1_score(y_actual, y_predict)
        if threshold_score > best_threshold_score:
            best_threshold_score = threshold_score
            best_threshold = threshold

    for i in range(0, 19):
        threshold = best_threshold + ((i - 9)/10000)
        y_predict = (y_predict_proba[:, 1] >= threshold).astype(bool)
        threshold_score = f1_score(y_actual, y_predict)
        if threshold_score > best_threshold_score:
            best_threshold_score = threshold_score
            best_threshold = threshold

    return best_threshold, best_threshold_score

