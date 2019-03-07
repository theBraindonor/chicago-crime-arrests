#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Search the experiment scores and range their output.  Focuses on the initial models before removing
    the false signals.
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
    'decision_tree_basic_full',
    'decision_tree_combine_sampled_full',
    'decision_tree_over_sampled_full',
    'decision_tree_under_sampled_full',
    'xgboost_basic_full',
    'xgboost_combine_sampled_full',
    'xgboost_over_sampled_full',
    'xgboost_under_sampled_full',
]

if __name__ == '__main__':
    use_project_path()
    logger = Logger('model/experiment/output/full_model_scoring_search.txt')

    scores = []

    for predictions in input_predictions:
        with open('model/experiment/output/%s_predict_proba.p' % predictions, 'rb') as file:
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





