#! /usr/bin/env python

import os
import datetime
import click
import gtmarket as gtm
from rocketry import Rocketry
import easier as ezr

# Will want to write to logs for daemon process
logger = ezr.get_logger('snap_sfdc')


class ObjectSnapper:
    DATA_DIR = os.path.realpath(os.path.expanduser('~/data/gtm_snapshots'))

    def __init__(self, kind, use_cache=False):
        """
        This class does the file management for putting snapshot dataframes
        in the right place.  I leverages the loaders from the gtmarket package.
        """
        self.kind = kind
        loading_mapper = {
            'opportunity': gtm.OppLoader,
            'account': gtm.AccountLoader,
            'order': gtm.OrderProducts,
        }

        allowed_kinds = sorted(loading_mapper.keys())
        if kind not in allowed_kinds:
            raise ValueError(f'{kind} not in {allowed_kinds}')

        self.loader = loading_mapper[kind]()
        if use_cache:
            self.loader.enable_pickle_cache()
        else:
            self.loader.disable_pickle_cache()

    @property
    def df(self):
        return self.loader.df_raw

    @property
    def directory(self):
        return os.path.join(self.DATA_DIR, str(datetime.datetime.now().date()))

    @property
    def file_name(self):
        return {
            'opportunity': 'opportunities.csv',
            'account': 'accounts.csv',
            'order': 'orders.csv',
        }[self.kind]

    @property
    def file_path(self):
        return os.path.join(self.directory, self.file_name)

    def _ensure_directory(self):
        os.makedirs(self.directory, exist_ok=True)

    def _create_file(self):
        df = self.df
        df.insert(0, 'snapshot_date', datetime.datetime.now().date())
        df.to_csv(self.file_path, index=False)

    def run(self):
        self._ensure_directory()
        self._create_file()


def run():
    # Set this to True when debugging to minimize salesforce hits
    use_cache = False

    # These are the names of the objects we want to snapshot
    object_names = ['opportunity', 'account', 'order']

    # This is the day we are snapshotting on
    today = str(datetime.datetime.now().date())

    # Loop over all names
    for object_name in object_names:
        # When running in daemon mode, you don't want to barf on error, just
        # write errors to log
        try:
            # Run the snapper and tell the log about it
            snapper = ObjectSnapper(object_name, use_cache=use_cache)
            snapper.run()
            logger.info(f'date: {today}  sucess for: {object_name}')
        except:  # noqa
            # If an error happened, tell the log about it
            logger.exception(f'date: {today} error for: {object_name}\n')

