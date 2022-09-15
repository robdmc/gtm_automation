#! /usr/bin/env python

import dash_lib as dl
import datetime
import pandas as pd
import time
import easier as ezr


logger = ezr.get_logger('dash_data_runner')


def run():
    dd = dl.DashData()
    dd.run()
    logger.info('success all dashboard data')




