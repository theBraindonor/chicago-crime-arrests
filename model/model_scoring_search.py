#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Search the model scores and range their output.  Will focus on optimizing the threshold to maximize f1
    score on all models and then sort from there.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

import pandas as pd
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score

from utility import find_optimal_f1_threshold, Logger, use_project_path

input_predictions = [
    'neural_network_basic',
    'neural_network_basic_fs',
    'sgd_huber_loss_over_sampled',
    'sgd_huber_loss_over_sampled_fs',
    'xgboost_basic',
    'xgboost_basic_fs'
]

if __name__ == '__main__':
    use_project_path()
    logger = Logger('model/output/model_scoring_search.txt')

    scores = []

    for predictions in input_predictions:
        with open('model/output/%s_predict_proba.p' % predictions, 'rb') as file:
            frame = pickle.load(file)
        optimal_threshold, optimal_f1_score = find_optimal_f1_threshold(frame)
        y_actual = frame.y_actual
        y_predict_proba = frame.y_predict
        y_predict = (y_predict_proba[:, 1] >= optimal_threshold).astype(bool)

        score = [
            predictions,
            optimal_threshold,
            optimal_f1_score,
            accuracy_score(y_actual, y_predict),
            precision_score(y_actual, y_predict),
            recall_score(y_actual, y_predict)
        ]
        scores.append(score)
        print(score)

    score_frame = pd.DataFrame(scores, columns=['model', 'threshold', 'f1', 'accuracy', 'precision', 'recall'])

    logger.log("Sorted For Accuracy:")
    score_frame.sort_values('accuracy', ascending=False, inplace=True)
    logger.log(score_frame)

    logger.log("Sorted For F1:")
    score_frame.sort_values('f1', ascending=False, inplace=True)
    logger.log(score_frame)
