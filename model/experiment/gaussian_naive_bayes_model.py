#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Experiment with a gaussian naive bayes model with a variety of balancing techniques on the cleaned data set
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer

from sklearn_pandas import DataFrameMapper

from utility import Runner
from model import load_clean_sample_data_frame

# Due to limitations on sample size with naive bayes, we really can't go beyond 25k records
sample = 25000

# We also have a significantly reduced feature space.  This takes the top features as reported by the best
# performing model during experimentation, XGBoost.
mapper = DataFrameMapper([
    (['hour'], [OneHotEncoder(handle_unknown='ignore')]),
    (['iucr'], [OneHotEncoder(handle_unknown='ignore')]),
    (['beat'], [OneHotEncoder(handle_unknown='ignore')]),
    (['month'], [OneHotEncoder(handle_unknown='ignore')]),
    (['weekday'], [OneHotEncoder(handle_unknown='ignore')]),
    (['location'], [OneHotEncoder(handle_unknown='ignore')]),
    (['latitude'], [KBinsDiscretizer(n_bins=256)]),
    (['longitude'], [KBinsDiscretizer(n_bins=256)]),
])

gaussian_naive_bayes = Pipeline([
    ('mapper', mapper),
    ('gnb', GaussianNB())
])


def test_gaussian_naive_bayes():
    runner = Runner(
        'model/experiment/output/gaussian_naive_bayes_basic',
        load_clean_sample_data_frame(),
        'arrest',
        gaussian_naive_bayes
    )
    runner.run_classification_experiment(
        sample=sample,
        record_predict_proba=True
    )

    runner = Runner(
        'model/experiment/output/gaussian_naive_bayes_under_samples',
        load_clean_sample_data_frame(),
        'arrest',
        gaussian_naive_bayes
    )
    runner.run_classification_experiment(
        sample=sample,
        record_predict_proba=True,
        sampling=RandomUnderSampler()
    )

    runner = Runner(
        'model/experiment/output/gaussian_naive_bayes_over_samples',
        load_clean_sample_data_frame(),
        'arrest',
        gaussian_naive_bayes
    )
    runner.run_classification_experiment(
        sample=sample,
        record_predict_proba=True,
        sampling=SMOTE()
    )

    runner = Runner(
        'model/experiment/output/gaussian_naive_bayes_combine_sampled',
        load_clean_sample_data_frame(),
        'arrest',
        gaussian_naive_bayes
    )
    runner.run_classification_experiment(
        sample=sample,
        record_predict_proba=True,
        sampling=SMOTEENN()
    )


if __name__ == '__main__':
    test_gaussian_naive_bayes()
