#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Gather non-numeric category variables and transform them into numeric values.
    All new values are saved to json files.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

import csv
import json

from utility import Logger, use_project_path


def serialize_codes(original_codes, filename, logger):
    """
    Code the incoming dictionary and save it to a json file
    :param original_codes: Dictionary with the keys we want to save
    :param filename: Destination filename
    :param logger: logging utility instance
    """
    new_codes = {}
    code_count = 1
    for code in original_codes.keys():
        new_codes[code] = code_count
        code_count += 1
    with open(filename, 'w') as file:
        json.dump(new_codes, file, indent=1)
    logger.log('Saving %s category values in %s' % (code_count, filename))


if __name__ == '__main__':
    use_project_path()

    logger = Logger('Data_Scratch/categories.txt')

    raw_record_count = -1

    iucr_codes = {}
    type_codes = {}
    description_codes = {}
    location_codes = {}
    fbi_codes = {}

    logger.time_log('Starting Category Transformation.')
    with open('data_scratch/Crimes_-_2001_to_present.csv') as input_file:
        with open('data_scratch/preprocessed_crimes.csv', 'w', newline='') as output_file:
            reader = csv.reader(input_file)

            for i, line in enumerate(reader):
                raw_record_count += 1
                if raw_record_count == 0:
                    continue

                iucr = line[4]
                type = line[5]
                description = line[6]
                location = line[7]
                fbi_code = line[14]

                iucr_codes[iucr] = 1
                type_codes[type] = 1
                description_codes[description] = 1
                location_codes[location] = 1
                fbi_codes[fbi_code] = 1

            logger.time_log('Category Transformation Complete.\n')
            logger.log('Processed %s total records' % raw_record_count)
            serialize_codes(iucr_codes, 'data_scratch/iucr_codes.json', logger)
            serialize_codes(type_codes, 'data_scratch/type_codes.json', logger)
            serialize_codes(description_codes, 'data_scratch/description_codes.json', logger)
            serialize_codes(location_codes, 'data_scratch/location_codes.json', logger)
            serialize_codes(fbi_codes, 'data_scratch/fbi_codes.json', logger)
