#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Experiment with an ada boost model with a variety of balancing techniques on the cleaned data set
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from skopt.space import Categorical, Integer, Real

from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline

from utility import HyperParameters, Runner
from model import load_clean_sample_data_frame, ordinal_data_mapper

sample = None
iterations = 12

hyper_parameters = HyperParameters({
    'ada__n_estimators': Integer(50, 200),
    'ada__learning_rate': Real(0.5, 1.5),
    'ada__algorithm': Categorical(['SAMME', 'SAMME.R'])
})

ada_boost_pipeline = Pipeline([
    ('mapper', ordinal_data_mapper),
    ('ada', AdaBoostClassifier())
])


def test_ada_boost():
    runner = Runner(
        'model/experiment/output/ada_boost_basic',
        load_clean_sample_data_frame(),
        'arrest',
        ada_boost_pipeline,
        hyper_parameters=hyper_parameters
    )
    runner.run_classification_search_experiment(
        'roc_auc',
        sample=sample,
        n_iter=iterations,
        record_predict_proba=True
    )

    runner = Runner(
        'model/experiment/output/ada_boost_under_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        ada_boost_pipeline,
        hyper_parameters=hyper_parameters
    )
    runner.run_classification_search_experiment(
        'roc_auc',
        sample=sample,
        n_iter=iterations,
        record_predict_proba=True,
        sampling=RandomUnderSampler()
    )

    runner = Runner(
        'model/experiment/output/ada_boost_over_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        ada_boost_pipeline,
        hyper_parameters=hyper_parameters
    )
    runner.run_classification_search_experiment(
        'roc_auc',
        sample=sample,
        n_iter=iterations,
        record_predict_proba=True,
        sampling=SMOTE()
    )

    runner = Runner(
        'model/experiment/output/ada_boost_combine_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        ada_boost_pipeline,
        hyper_parameters=hyper_parameters
    )
    runner.run_classification_search_experiment(
        'roc_auc',
        sample=sample,
        n_iter=iterations,
        record_predict_proba=True,
        sampling=SMOTEENN()
    )


if __name__ == '__main__':
    test_ada_boost()
