# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Common logger that will log to stdout and a file.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

import pandas as pd

from datetime import datetime


class Logger:
    """
    Class to provide an easy common logging routine.  Allows for utilities to capture output easily without
    having to resort to redirecting stdout.
    """
    def __init__(self, filename):
        """
        Create a new logger with an open file handle to the given filename.
        Will overwrite the file if it already exists.
        :param filename:
        """
        self.filename = filename
        self.file = open(self.filename, 'w')

    def close(self):
        """
        Close the file handle of the logger.
        :return:
        """
        self.file.close()

    def log(self, message):
        """
        Log a message.  Will update pandas formatting to provide extra-wide display.
        :param message:
        :return:
        """
        with pd.option_context(
                'display.max_rows', None, 'display.max_columns', None,
                'display.width', 1024, 'display.max_colwidth', 256):
            self.file.write('%s\n' % message)
            print(message)

    def time_log(self, message):
        """
        Log a message with a timestamp.
        :param message:
        :return:
        """
        self.log("[%s] %s" % (str(datetime.now()), message))
