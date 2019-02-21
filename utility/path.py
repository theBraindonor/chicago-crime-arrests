#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

import re
import os


def use_project_path():
    """
    Based on the path of this file, we change directory to the project root.  This is used in the scripts to ensure
    path resolution is done the same when files are run through the IDE and command line.
    :return:
    """
    path = re.sub(
        '[\\\\/]utility[\\\\/]path\\.py$',
        '',
        __file__
    )
    os.chdir(path)
