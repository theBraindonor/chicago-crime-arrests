#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Experiment with a random forest model with a variety of balancing techniques on the cleaned data set
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from utility import HyperParameters, Runner
from model import load_clean_sample_data_frame, ordinal_data_mapper

sample = None
iterations = 12

hyper_parameters = HyperParameters({
    'rf__n_estimators': Integer(50, 150),
    'rf__criterion': Categorical(['gini', 'entropy']),
    'rf__max_depth': Integer(4, 18),
    'rf__max_features': Categorical(['sqrt', 'log2'])
})

random_forest_pipeline = Pipeline([
    ('mapper', ordinal_data_mapper),
    ('rf', RandomForestClassifier())
])


def test_random_forest():
    runner = Runner(
        'model/experiment/output/random_forest_basic',
        load_clean_sample_data_frame(),
        'arrest',
        random_forest_pipeline,
        hyper_parameters=hyper_parameters
    )
    runner.run_classification_search_experiment(
        'roc_auc',
        sample=sample,
        n_iter=iterations,
        record_predict_proba=True
    )

    runner = Runner(
        'model/experiment/output/random_forest_under_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        random_forest_pipeline,
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
        'model/experiment/output/random_forest_over_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        random_forest_pipeline,
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
        'model/experiment/output/random_forest_combine_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        random_forest_pipeline,
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
    test_random_forest()
