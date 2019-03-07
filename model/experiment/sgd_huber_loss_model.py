#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Experiment with a stochastic gradient descent model with modified huber loss and a variety of balancing techniques
    on the cleaned data set
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.linear_model import SGDClassifier

from utility import Runner
from model import load_clean_sample_data_frame, binned_geo_one_hot_data_mapper


sample = None
fit_increment = 10000
max_iters = 10


def test_sgd_huber_loss():
    runner = Runner(
        'model/experiment/output/sgd_huber_loss_basic',
        load_clean_sample_data_frame(),
        'arrest',
        SGDClassifier(loss='modified_huber')
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
        'model/experiment/output/sgd_huber_loss_under_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        SGDClassifier(loss='modified_huber')
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
        'model/experiment/output/sgd_huber_loss_over_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        SGDClassifier(loss='modified_huber')
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
        'model/experiment/output/sgd_huber_loss_combine_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        SGDClassifier(loss='modified_huber')
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
    test_sgd_huber_loss()
