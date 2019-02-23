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
import datetime
import json

from utility import Logger, use_project_path


def load_codes(filename):
    with open(filename, 'r') as file:
        return json.load(file)


column_names = [
    'arrest',
    'year',
    'month',
    'weekday',
    'hour',
    'iucr',
    'type',
    'description',
    'location',
    'domestic',
    'beat',
    'district',
    'ward',
    'community',
    'fbi_code',
    'latitude',
    'longitude'
]

if __name__ == '__main__':
    use_project_path()

    logger = Logger('Data_Scratch/preprocess.txt')
    iucr_codes = load_codes('data_scratch/iucr_codes.json')
    type_codes = load_codes('data_scratch/type_codes.json')
    description_codes = load_codes('data_scratch/description_codes.json')
    location_codes = load_codes('data_scratch/location_codes.json')
    fbi_codes = load_codes('data_scratch/fbi_codes.json')

    missing_columns = {}
    for column in column_names:
        missing_columns[column] = 1

    raw_record_count = -1
    processed_records = 0

    logger.time_log('Starting Pre-Processing.')
    with open('data_scratch/Crimes_-_2001_to_present.csv') as input_file:
        with open('data_scratch/preprocessed_crimes.csv', 'w', newline='') as output_file:
            reader = csv.reader(input_file)
            writer = csv.writer(output_file)

            writer.writerow(column_names)

            for i, line in enumerate(reader):
                raw_record_count += 1
                if raw_record_count == 0:
                    continue

                arrest = line[8]
                date = line[2]
                iucr = line[4]
                type = line[5]
                description = line[6]
                location = line[7]
                domestic = line[9]
                beat = line[10]
                district = line[11]
                ward = line[12]
                community = line[13]
                fbi_code = line[14]
                latitude = line[19]
                longitude = line[20]

                year = None
                month = None
                weekday = None
                hour = None
                if date:
                    date_object = datetime.datetime.strptime(date, '%m/%d/%Y %I:%M:%S %p')
                    year = date_object.year
                    month = date_object.month
                    weekday = date_object.weekday()
                    hour = date_object.hour

                if arrest == 'true':
                    arrest = 1
                elif arrest == 'false':
                    arrest = 0

                if domestic == 'true':
                    domestic = 1
                elif domestic == 'false':
                    domestic = 0

                has_missing = False
                if arrest is None or arrest is '':
                    has_missing = True
                    missing_columns['arrest'] += 1
                if year is None or year is '':
                    has_missing = True
                    missing_columns['year'] += 1
                if month is None or month is '':
                    has_missing = True
                    missing_columns['month'] += 1
                if weekday is None or weekday is '':
                    has_missing = True
                    missing_columns['weekday'] += 1
                if hour is None or hour is '':
                    has_missing = True
                    missing_columns['hour'] += 1
                if iucr is None or iucr is '':
                    has_missing = True
                    missing_columns['iucr'] += 1
                if type is None or type is '':
                    has_missing = True
                    missing_columns['type'] += 1
                if description is None or description is '':
                    has_missing = True
                    missing_columns['description'] += 1
                if location is None or location is '':
                    has_missing = True
                    missing_columns['location'] += 1
                if domestic is None or domestic is '':
                    has_missing = True
                    missing_columns['domestic'] += 1
                if beat is None or beat is '':
                    has_missing = True
                    missing_columns['beat'] += 1
                if district is None or district is '':
                    has_missing = True
                    missing_columns['district'] += 1
                if ward is None or ward is '':
                    has_missing = True
                    missing_columns['ward'] += 1
                if community is None or community is '':
                    has_missing = True
                    missing_columns['community'] += 1
                if fbi_code is None or fbi_code is '':
                    has_missing = True
                    missing_columns['fbi_code'] += 1
                if latitude is None or latitude is '':
                    has_missing = True
                    missing_columns['latitude'] += 1
                if longitude is None or longitude is '':
                    has_missing = True
                    missing_columns['longitude'] += 1

                if not has_missing:
                    writer.writerow([
                        arrest,
                        year,
                        month,
                        weekday,
                        hour,
                        iucr_codes[iucr],
                        type_codes[type],
                        description_codes[description],
                        location_codes[location],
                        domestic,
                        beat,
                        district,
                        ward,
                        community,
                        fbi_codes[fbi_code],
                        latitude,
                        longitude
                    ])
                    processed_records += 1

    logger.time_log('Pre-Processing Complete.\n')
    logger.log('Total Records: %s' % raw_record_count)
    logger.log('Processed Records: %s\n' % processed_records)
    logger.log('Missing Data:')
    for column in missing_columns.keys():
        logger.log('    %s: %s' % (column, missing_columns[column]))
