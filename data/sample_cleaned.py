#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Sample the pre-processed data file into a smaller form that can be worked on more easily.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

import csv
import random

from utility import use_project_path, Logger
from data import column_names

if __name__ == "__main__":
    use_project_path()

    logger = Logger('data_scratch/sample_cleaned.txt')

    sample_seed = 1027
    sample_rate = 0.10
    raw_record_count = -1
    processed_records = 0

    random.seed(sample_seed)

    logger.time_log('Starting Data Sampling.')

    with open('data_scratch/cleaned_crimes.csv') as input_file:
        with open('data_scratch/sampled_cleaned_crimes.csv', 'w', newline='') as output_file:
            reader = csv.reader(input_file)
            writer = csv.writer(output_file)

            writer.writerow(column_names)

            for i, line in enumerate(reader):
                raw_record_count += 1
                if raw_record_count == 0:
                    continue

                if random.random() < sample_rate:
                    writer.writerow(line)
                    processed_records += 1

    logger.time_log('Data Sampling Complete.\n')
    logger.log('Total Records: %s' % raw_record_count)
    logger.log('Sampled Records: %s' % processed_records)
