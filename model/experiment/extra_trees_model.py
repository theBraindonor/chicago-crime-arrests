#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Experiment with an extra trees model with a variety of balancing techniques on the cleaned data set
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

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline

from utility import HyperParameters, Runner
from model import load_clean_sample_data_frame, ordinal_data_mapper

sample = None
iterations = 24

hyper_parameters = HyperParameters({
    'et__n_estimators': Integer(10, 100),
    'et__criterion': Categorical(['gini', 'entropy']),
    'et__max_depth': Integer(4, 24),
    'et__min_samples_leaf': Real(0.000001, 0.001),
    'et__min_samples_split': Real(0.000002, 0.002)
})

extra_trees_pipeline = Pipeline([
    ('mapper', ordinal_data_mapper),
    ('et', ExtraTreesClassifier())
])


def test_extra_trees():
    runner = Runner(
        'model/experiment/output/extra_trees_basic',
        load_clean_sample_data_frame(),
        'arrest',
        extra_trees_pipeline,
        hyper_parameters=hyper_parameters
    )
    runner.run_classification_search_experiment(
        'roc_auc',
        sample=sample,
        n_iter=iterations,
        record_predict_proba=True
    )

    runner = Runner(
        'model/experiment/output/extra_trees_under_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        extra_trees_pipeline,
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
        'model/experiment/output/extra_trees_over_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        extra_trees_pipeline,
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
        'model/experiment/output/extra_trees_combine_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        extra_trees_pipeline,
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
    test_extra_trees()
