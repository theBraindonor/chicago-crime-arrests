#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Build a neural network model of arrests in the Chicago crime data.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.neural_network import MLPClassifier

from sklearn_pandas import DataFrameMapper

from utility import Runner
from model import load_clean_data_frame, binned_geo_one_hot_data_mapper

sample = None
fit_increment = 10000
max_iters = 5

# Features were taken from average feature importance listed in all supported experiments
mapper = DataFrameMapper([
    (['iucr'], [OneHotEncoder(handle_unknown='ignore')]),
    (['type'], [OneHotEncoder(handle_unknown='ignore')]),
    (['location'], [OneHotEncoder(handle_unknown='ignore')]),
    (['fbi_code'], [OneHotEncoder(handle_unknown='ignore')]),
    (['hour'], [OneHotEncoder(handle_unknown='ignore')]),
    (['property_crime'], None),
    (['longitude'], [KBinsDiscretizer(n_bins=256)]),
    (['latitude'], [KBinsDiscretizer(n_bins=256)]),
    (['weekday'], [OneHotEncoder(handle_unknown='ignore')]),
    (['domestic'], [OneHotEncoder(handle_unknown='ignore')])
])

nn = MLPClassifier(hidden_layer_sizes=(1000,200,), verbose=True)

pipeline = Pipeline([
    ('mapper', binned_geo_one_hot_data_mapper),
    ('nn', nn)
])

nn_fs = MLPClassifier(hidden_layer_sizes=(1000,200,), verbose=True)

pipeline_fs = Pipeline([
    ('mapper', mapper),
    ('nn', nn_fs)
])


def build_neural_network():
    runner = Runner(
        'model/output/neural_network_basic',
        load_clean_data_frame(),
        'arrest',
        nn
    )
    runner.run_classification_experiment(
        sample=sample,
        record_predict_proba=True,
        transformer=binned_geo_one_hot_data_mapper,
        fit_increment=fit_increment,
        max_iters=max_iters,
        cv=None,
        n_jobs=1
    )
    joblib.dump(
        pipeline,
        'model/output/neural_network_basic.joblib'
    )

    runner = Runner(
        'model/output/neural_network_basic_fs',
        load_clean_data_frame(),
        'arrest',
        nn_fs
    )
    runner.run_classification_experiment(
        sample=sample,
        record_predict_proba=True,
        transformer=mapper,
        fit_increment=fit_increment,
        max_iters=max_iters,
        cv=None,
        n_jobs=1
    )
    joblib.dump(
        pipeline_fs,
        'model/output/neural_network_basic_fs.joblib'
    )


if __name__ == '__main__':
    build_neural_network()
