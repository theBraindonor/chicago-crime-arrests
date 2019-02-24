#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Remove the crime types that are arrest based, meaning that they are reported by police at the time of arrest.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

import csv
import random

from utility import use_project_path, Logger
from data import column_names, load_codes

if __name__ == "__main__":
    use_project_path()

    logger = Logger('data_scratch/remove_arrest_based_crime.txt')

    sample_seed = 1027
    sample_rate = 0.10
    raw_record_count = -1
    processed_records = 0

    random.seed(sample_seed)

    logger.time_log('Starting Data Cleaning.')

    type_codes = load_codes('data_scratch/type_codes.json')
    arrest_based_crimes = [
        type_codes['PROSTITUTION'],
        type_codes['NARCOTICS'],
        type_codes['PUBLIC INDECENCY'],
        type_codes['GAMBLING'],
        type_codes['LIQUOR LAW VIOLATION']
    ]

    with open('data_scratch/preprocessed_crimes.csv') as input_file:
        with open('data_scratch/cleaned_crimes.csv', 'w', newline='') as output_file:
            reader = csv.reader(input_file)
            writer = csv.writer(output_file)

            writer.writerow(column_names)

            for i, line in enumerate(reader):
                raw_record_count += 1
                if raw_record_count == 0:
                    continue

                if int(line[column_names.index('type')]) not in arrest_based_crimes:
                    writer.writerow(line)
                    processed_records += 1

    logger.time_log('Data Cleaning Complete.\n')
    logger.log('Total Records: %s' % raw_record_count)
    logger.log('Clean Records: %s' % processed_records)
