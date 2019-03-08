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

from imblearn.over_sampling import SMOTE

from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

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

sgd = SGDClassifier(loss='modified_huber')

pipeline = Pipeline([
    ('mapper', mapper),
    ('sgd', sgd)
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
        transformer=mapper,
        fit_increment=fit_increment,
        max_iters=max_iters,
        n_jobs=1,
        sampling=SMOTE(),
        cv=None
    )
    joblib.dump(
        pipeline,
        'model/output/sgd_huber_loss_over_sampled.joblib'
    )


if __name__ == '__main__':
    build_sgd_huber_loss()
