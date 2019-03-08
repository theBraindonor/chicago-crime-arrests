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

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.neural_network import MLPClassifier

from sklearn_pandas import DataFrameMapper

from utility import Runner
from model import load_clean_data_frame

sample = None
fit_increment = 10000
max_iters = 20

mapper = DataFrameMapper([
    (['iucr'], [OneHotEncoder(handle_unknown='ignore')]),
    (['type'], [OneHotEncoder(handle_unknown='ignore')]),
    (['location'], [OneHotEncoder(handle_unknown='ignore')]),
    (['fbi_code'], [OneHotEncoder(handle_unknown='ignore')]),
    (['hour'], [OneHotEncoder(handle_unknown='ignore')]),
    (['property_crime'], None),
    (['longitude'], [KBinsDiscretizer(n_bins=512)]),
    (['latitude'], [KBinsDiscretizer(n_bins=512)]),
    (['weekday'], [OneHotEncoder(handle_unknown='ignore')]),
    (['domestic'], [OneHotEncoder(handle_unknown='ignore')])
])

nn = MLPClassifier(hidden_layer_sizes=(1000,200,), verbose=True)

pipeline = Pipeline([
    ('mapper', mapper),
    ('nn', nn)
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
        transformer=mapper,
        fit_increment=fit_increment,
        max_iters=max_iters,
        cv=None,
        n_jobs=1
    )
    joblib.dump(
        pipeline,
        'model/output/neural_network_basic.joblib'
    )


if __name__ == '__main__':
    build_neural_network()
