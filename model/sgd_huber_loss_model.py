#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Build a stochastic gradient descent model of arrests in the Chicago crime data.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

from imblearn.over_sampling import SMOTE

from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

from sklearn_pandas import DataFrameMapper

from utility import Runner
from model import load_clean_data_frame, binned_geo_one_hot_data_mapper


sample = None
fit_increment = 10000
max_iters = 20

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

sgd = SGDClassifier(loss='modified_huber')

pipeline = Pipeline([
    ('mapper', binned_geo_one_hot_data_mapper),
    ('sgd', sgd)
])

sgd_fs = SGDClassifier(loss='modified_huber')

pipeline_fs = Pipeline([
    ('mapper', mapper),
    ('sgd', sgd_fs)
])


def build_sgd_huber_loss():
    runner = Runner(
        'model/output/sgd_huber_loss_over_sampled',
        load_clean_data_frame(),
        'arrest',
        sgd
    )
    runner.run_classification_experiment(
        sample=sample,
        record_predict_proba=True,
        transformer=binned_geo_one_hot_data_mapper,
        sampling=SMOTE(),
        fit_increment=fit_increment,
        max_iters=max_iters,
        n_jobs=1,
        cv=None
    )
    joblib.dump(
        pipeline,
        'model/output/sgd_huber_loss_over_sampled.joblib'
    )

    runner = Runner(
        'model/output/sgd_huber_loss_over_sampled_fs',
        load_clean_data_frame(),
        'arrest',
        sgd_fs
    )
    runner.run_classification_experiment(
        sample=sample,
        record_predict_proba=True,
        transformer=mapper,
        sampling=SMOTE(),
        fit_increment=fit_increment,
        max_iters=max_iters,
        n_jobs=1,
        cv=None
    )
    joblib.dump(
        pipeline_fs,
        'model/output/sgd_huber_loss_over_sampled_fs.joblib'
    )


if __name__ == '__main__':
    build_sgd_huber_loss()
