#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Experiment with a decision tree model with a variety of balancing techniques on the cleaned data set
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

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from utility import HyperParameters, Runner
from model import load_clean_sample_data_frame, ordinal_data_mapper

sample = None
iterations = 24

hyper_parameters = HyperParameters({
    'dt__criterion': Categorical(['gini', 'entropy']),
    'dt__max_depth': Integer(4, 24),
    'dt__min_samples_leaf': Real(0.000001, 0.001),
    'dt__min_samples_split': Real(0.000002, 0.002)
})

decision_tree_basic = Pipeline([
    ('mapper', ordinal_data_mapper),
    ('dt', DecisionTreeClassifier())
])


def test_decision_tree():
    runner = Runner(
        'model/experiment/output/decision_tree_basic',
        load_clean_sample_data_frame(),
        'arrest',
        decision_tree_basic,
        hyper_parameters=hyper_parameters
    )
    runner.run_classification_search_experiment(
        'roc_auc',
        sample=sample,
        n_iter=iterations,
        record_predict_proba=True
    )

    runner = Runner(
        'model/experiment/output/decision_tree_under_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        decision_tree_basic,
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
        'model/experiment/output/decision_tree_over_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        decision_tree_basic,
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
        'model/experiment/output/decision_tree_combine_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        decision_tree_basic,
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
    test_decision_tree()
