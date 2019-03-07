#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Search the experiment scores and range their output.  Will focus on optimizing the threshold to maximize f1
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
    'ada_boost_basic',
    'ada_boost_combine_sampled',
    'ada_boost_over_sampled',
    'ada_boost_under_sampled',
    'decision_tree_basic',
    'decision_tree_combine_sampled',
    'decision_tree_over_sampled',
    'decision_tree_under_sampled',
    'extra_trees_basic',
    'extra_trees_combine_sampled',
    'extra_trees_over_sampled',
    'extra_trees_under_sampled',
    'gaussian_naive_bayes_basic',
    'gaussian_naive_bayes_combine_sampled',
    'gaussian_naive_bayes_over_sampled',
    'gaussian_naive_bayes_under_sampled',
    'gradient_boosting_basic',
    'gradient_boosting_combine_sampled',
    'gradient_boosting_over_sampled',
    'gradient_boosting_under_sampled',
    'random_forest_basic',
    'random_forest_combine_sampled',
    'random_forest_over_sampled',
    'random_forest_under_sampled',
    'sgd_huber_loss_basic',
    'sgd_huber_loss_combine_sampled',
    'sgd_huber_loss_over_sampled',
    'sgd_huber_loss_under_sampled',
    'sgd_log_loss_basic',
    'sgd_log_loss_combine_sampled',
    'sgd_log_loss_over_sampled',
    'sgd_log_loss_under_sampled',
    'xgboost_basic',
    'xgboost_combine_sampled',
    'xgboost_over_sampled',
    'xgboost_under_sampled',
]

if __name__ == '__main__':
    use_project_path()
    logger = Logger('model/experiment/output/model_scoring_search.txt')

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
