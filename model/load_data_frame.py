#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Loading the data frame for the model and making any pandas adjustments.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

import pandas as pd

from utility import use_project_path

clean_sample_data_filename = 'data_scratch/sampled_cleaned_crimes.csv'
clean_data_filename = 'data_scratch/cleaned_crimes.csv'
sample_data_filename = 'data_scratch/sampled_crimes.csv'
data_filename = 'data_scratch/preprocessed_crimes.csv'


def load_data(filename):
    use_project_path()
    data_frame = pd.read_csv(filename, dtype='float')
    return data_frame


def load_sample_data_frame():
    return load_data(sample_data_filename)


def load_data_frame():
    return load_data(data_filename)


def load_clean_sample_data_frame():
    return load_data(clean_sample_data_filename)


def load_clean_data_frame():
    return load_data(clean_data_filename)
