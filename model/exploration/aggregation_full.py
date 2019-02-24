#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Pre-Process the raw data from data.cityofchicago.org
    Will remove missing records and translate any non-numeric code values
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

import csv
import pandas as pd

from data import load_codes, column_names
from utility import use_project_path

if __name__ == '__main__':
    use_project_path()

    raw_record_count = -1

    iucr_codes = load_codes('data_scratch/iucr_codes.json')
    iucr_matrix = []
    for key in iucr_codes.keys():
        iucr_matrix.append([key, 0, 0, 0])

    type_codes = load_codes('data_scratch/type_codes.json')
    type_matrix = []
    for key in type_codes.keys():
        type_matrix.append([key, 0, 0, 0])

    description_codes = load_codes('data_scratch/description_codes.json')
    description_matrix = []
    for key in description_codes.keys():
        description_matrix.append([key, 0, 0, 0])

    location_codes = load_codes('data_scratch/location_codes.json')
    location_matrix = []
    for key in location_codes.keys():
        location_matrix.append([key, 0, 0, 0])

    fbi_codes = load_codes('data_scratch/fbi_codes.json')
    fbi_matrix = []
    for key in fbi_codes.keys():
        fbi_matrix.append([key, 0, 0, 0])

    total_crimes = 0
    total_arrests = 0
    total_rate = 0

    with open('data_scratch/preprocessed_crimes.csv') as input_file:
        reader = csv.reader(input_file)

        for i, line in enumerate(reader):
            raw_record_count += 1
            if raw_record_count == 0:
                continue

            arrest = int(line[column_names.index('arrest')])
            iucr = int(line[column_names.index('iucr')])
            type = int(line[column_names.index('type')])
            description = int(line[column_names.index('description')])
            location = int(line[column_names.index('location')])
            fbi = int(line[column_names.index('fbi_code')])

            iucr_matrix[iucr - 1][1] += 1
            iucr_matrix[iucr - 1][2] += arrest

            type_matrix[type - 1][1] += 1
            type_matrix[type - 1][2] += arrest

            description_matrix[description - 1][1] += 1
            description_matrix[description - 1][2] += arrest

            location_matrix[location - 1][1] += 1
            location_matrix[location - 1][2] += arrest

            fbi_matrix[fbi - 1][1] += 1
            fbi_matrix[fbi - 1][2] += arrest

            total_crimes += 1
            total_arrests += arrest

    data_frame = pd.DataFrame(iucr_matrix, columns=['iucr', 'crimes', 'arrests', 'arrest_rate'])
    data_frame['arrest_rate'] = data_frame['arrests'] / data_frame['crimes']
    data_frame.sort_values('arrest_rate', ascending=False, inplace=True)
    data_frame.to_csv('model/exploration/output/iucr_aggregation_full.csv', index=False)

    data_frame = pd.DataFrame(type_matrix, columns=['type', 'crimes', 'arrests', 'arrest_rate'])
    data_frame['arrest_rate'] = data_frame['arrests'] / data_frame['crimes']
    data_frame.sort_values('arrest_rate', ascending=False, inplace=True)
    data_frame.to_csv('model/exploration/output/type_aggregation_full.csv', index=False)

    data_frame = pd.DataFrame(description_matrix, columns=['description', 'crimes', 'arrests', 'arrest_rate'])
    data_frame['arrest_rate'] = data_frame['arrests'] / data_frame['crimes']
    data_frame.sort_values('arrest_rate', ascending=False, inplace=True)
    data_frame.to_csv('model/exploration/output/description_aggregation_full.csv', index=False)

    data_frame = pd.DataFrame(location_matrix, columns=['location', 'crimes', 'arrests', 'arrest_rate'])
    data_frame['arrest_rate'] = data_frame['arrests'] / data_frame['crimes']
    data_frame.sort_values('arrest_rate', ascending=False, inplace=True)
    data_frame.to_csv('model/exploration/output/location_aggregation_full.csv', index=False)

    data_frame = pd.DataFrame(fbi_matrix, columns=['fbi', 'crimes', 'arrests', 'arrest_rate'])
    data_frame['arrest_rate'] = data_frame['arrests'] / data_frame['crimes']
    data_frame.sort_values('arrest_rate', ascending=False, inplace=True)
    data_frame.to_csv('model/exploration/output/fbi_aggregation_full.csv', index=False)

    data_frame = pd.DataFrame([[total_crimes, total_arrests, 0]], columns=['crimes', 'arrests', 'arrest_rate'])
    data_frame['arrest_rate'] = data_frame['arrests'] / data_frame['crimes']
    data_frame.sort_values('arrest_rate', ascending=False, inplace=True)
    data_frame.to_csv('model/exploration/output/total_aggregation_full.csv', index=False)

