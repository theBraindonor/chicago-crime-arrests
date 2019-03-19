#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Experiment with a neural network model with a variety of balancing techniques on the cleaned data set
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.neural_network import MLPClassifier

from utility import Runner
from model import load_clean_sample_data_frame, binned_geo_one_hot_data_mapper


sample = None
fit_increment = 10000
max_iters = 5


def test_neural_network():
    runner = Runner(
        'model/experiment/output/neural_network_basic',
        load_clean_sample_data_frame(),
        'arrest',
        MLPClassifier(hidden_layer_sizes=(750,125,), verbose=True)
    )
    runner.run_classification_experiment(
        sample=sample,
        record_predict_proba=True,
        transformer=binned_geo_one_hot_data_mapper,
        fit_increment=fit_increment,
        max_iters=max_iters,
        n_jobs=1
    )

    runner = Runner(
        'model/experiment/output/neural_network_under_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        MLPClassifier(hidden_layer_sizes=(750,125,), verbose=True)
    )
    runner.run_classification_experiment(
        sample=sample,
        record_predict_proba=True,
        transformer=binned_geo_one_hot_data_mapper,
        fit_increment=fit_increment,
        max_iters=max_iters,
        n_jobs=1,
        sampling=RandomUnderSampler()
    )

    runner = Runner(
        'model/experiment/output/neural_network_over_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        MLPClassifier(hidden_layer_sizes=(750,125,), verbose=True)
    )
    runner.run_classification_experiment(
        sample=sample,
        record_predict_proba=True,
        transformer=binned_geo_one_hot_data_mapper,
        fit_increment=fit_increment,
        max_iters=max_iters,
        n_jobs=1,
        sampling=SMOTE()
    )

    runner = Runner(
        'model/experiment/output/neural_network_combine_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        MLPClassifier(hidden_layer_sizes=(750,125,), verbose=True)
    )
    runner.run_classification_experiment(
        sample=sample,
        record_predict_proba=True,
        transformer=binned_geo_one_hot_data_mapper,
        fit_increment=fit_increment,
        max_iters=max_iters,
        n_jobs=1,
        sampling=SMOTEENN()
    )


if __name__ == '__main__':
    test_neural_network()
