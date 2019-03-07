#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Build an XGBoost model of arrests in the Chicago crime data.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

import xgboost as xgb

from skopt.space import Integer, Real

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from sklearn_pandas import DataFrameMapper

from utility import HyperParameters, Runner
from model import load_clean_data_frame

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

# Features were selected based on feature importance from experiments.
data_mapper = DataFrameMapper([
    (['month'], None),
    (['hour'], None),
    (['iucr'], None),
    (['type'], None),
    (['location'], None),
    (['beat'], None),
    (['fbi_code'], None),
    (['latitude'], None),
    (['longitude'], None)
])

xgboost_pipeline = Pipeline([
    ('mapper', data_mapper),
    ('xgb', xgb.XGBClassifier(tree_method='hist'))
])

def build_xgboost_model():
    runner = Runner(
        'model/output/xgboost_basic',
        load_clean_data_frame(),
        'arrest',
        xgboost_pipeline,
        hyper_parameters
    )
    runner.run_classification_search_experiment(
        'roc_auc',
        sample=sample,
        n_iter=iterations,
        record_predict_proba=True
    )
    joblib.dump(
        runner.trained_estimator,
        'model/output/xgboost_basic.joblib'
    )

if __name__ == '__main__':
    build_xgboost_model()
