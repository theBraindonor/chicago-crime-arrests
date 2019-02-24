#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Experiment with an XGBoost model with a variety of balancing techniques on the cleaned data set
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

import xgboost as xgb

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from skopt.space import Integer, Real

from sklearn.pipeline import Pipeline

from utility import HyperParameters, Runner
from model import load_clean_sample_data_frame, ordinal_data_mapper

sample = None
iterations = 24

hyper_parameters = HyperParameters(search_space={
    'xgb__n_estimators': Integer(100, 500),
    'xgb__learning_rate': Real(0.1, 0.3),
    'xgb__gamma': Real(0.0001, 100.0, prior='log-uniform'),
    'xgb__max_depth': Integer(3, 7),
    'xgb__colsample_bytree': Real(0.4, 0.8),
    'xgb__colsample_bylevel': Real(0.4, 0.8),
    'xgb__colsample_bynode': Real(0.4, 0.8)
})

xgboost_basic = Pipeline([
    ('mapper', ordinal_data_mapper),
    ('xgb', xgb.XGBClassifier(tree_method='hist'))
])


def test_xgboost():
    runner = Runner(
        'model/experiment/output/xgboost_basic',
        load_clean_sample_data_frame(),
        'arrest',
        xgboost_basic,
        hyper_parameters
    )
    runner.run_classification_search_experiment(
        'roc_auc',
        sample=sample,
        n_iter=iterations,
        record_predict_proba=True
    )

    runner = Runner(
        'model/experiment/output/xgboost_under_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        xgboost_basic,
        hyper_parameters
    )
    runner.run_classification_search_experiment(
        'roc_auc',
        sample=sample,
        n_iter=iterations,
        record_predict_proba=True,
        sampling=RandomUnderSampler()
    )

    runner = Runner(
        'model/experiment/output/xgboost_over_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        xgboost_basic,
        hyper_parameters
    )
    runner.run_classification_search_experiment(
        'roc_auc',
        sample=sample,
        n_iter=iterations,
        record_predict_proba=True,
        sampling=SMOTE()
    )

    runner = Runner(
        'model/experiment/output/xgboost_combine_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        xgboost_basic,
        hyper_parameters
    )
    runner.run_classification_search_experiment(
        'roc_auc',
        sample=sample,
        n_iter=iterations,
        record_predict_proba=True,
        sampling=SMOTEENN()
    )


if __name__ == '__main__':
    test_xgboost()
