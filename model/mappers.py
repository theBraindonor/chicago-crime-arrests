#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Common Data Mappers for Experiments
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer

from sklearn_pandas import DataFrameMapper

ordinal_data_mapper = DataFrameMapper([
    (['month'], [MinMaxScaler()]),
    (['weekday'], [MinMaxScaler()]),
    (['hour'], [MinMaxScaler()]),
    (['iucr'], [MinMaxScaler()]),
    (['type'], [MinMaxScaler()]),
    (['description'], [MinMaxScaler()]),
    (['location'], [MinMaxScaler()]),
    (['domestic'], [MinMaxScaler()]),
    (['beat'], [MinMaxScaler()]),
    (['district'], [MinMaxScaler()]),
    (['ward'], [MinMaxScaler()]),
    (['community'], [MinMaxScaler()]),
    (['fbi_code'], [MinMaxScaler()]),
    (['latitude'], [StandardScaler()]),
    (['longitude'], [StandardScaler()])
])

one_hot_data_mapper = DataFrameMapper([
    (['month'], [OneHotEncoder(handle_unknown='ignore')]),
    (['weekday'], [OneHotEncoder(handle_unknown='ignore')]),
    (['hour'], [OneHotEncoder(handle_unknown='ignore')]),
    (['iucr'], [OneHotEncoder(handle_unknown='ignore')]),
    (['type'], [OneHotEncoder(handle_unknown='ignore')]),
    (['description'], [OneHotEncoder(handle_unknown='ignore')]),
    (['location'], [OneHotEncoder(handle_unknown='ignore')]),
    (['domestic'], [OneHotEncoder(handle_unknown='ignore')]),
    (['beat'], [OneHotEncoder(handle_unknown='ignore')]),
    (['district'], [OneHotEncoder(handle_unknown='ignore')]),
    (['ward'], [OneHotEncoder(handle_unknown='ignore')]),
    (['community'], [OneHotEncoder(handle_unknown='ignore')]),
    (['fbi_code'], [OneHotEncoder(handle_unknown='ignore')]),
    (['latitude'], [StandardScaler()]),
    (['longitude'], [StandardScaler()])
])

binned_geo_one_hot_data_mapper = DataFrameMapper([
    (['month'], [OneHotEncoder(handle_unknown='ignore')]),
    (['weekday'], [OneHotEncoder(handle_unknown='ignore')]),
    (['hour'], [OneHotEncoder(handle_unknown='ignore')]),
    (['iucr'], [OneHotEncoder(handle_unknown='ignore')]),
    (['type'], [OneHotEncoder(handle_unknown='ignore')]),
    (['description'], [OneHotEncoder(handle_unknown='ignore')]),
    (['location'], [OneHotEncoder(handle_unknown='ignore')]),
    (['domestic'], [OneHotEncoder(handle_unknown='ignore')]),
    (['beat'], [OneHotEncoder(handle_unknown='ignore')]),
    (['district'], [OneHotEncoder(handle_unknown='ignore')]),
    (['ward'], [OneHotEncoder(handle_unknown='ignore')]),
    (['community'], [OneHotEncoder(handle_unknown='ignore')]),
    (['fbi_code'], [OneHotEncoder(handle_unknown='ignore')]),
    (['latitude'], [KBinsDiscretizer(n_bins=1024)]),
    (['longitude'], [KBinsDiscretizer(n_bins=1024)])
])
