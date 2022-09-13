#! /usr/bin/env python

# import os
# import datetime
import click
import importlib
# import gtmarket as gtm
# from rocketry import Rocketry
# import easier as ezr

# Will want to write to logs for daemon process
# logger = ezr.get_logger('snapper')
# app = Rocketry()



# def snap():
#     # Set this to True when debugging to minimize salesforce hits
#     use_cache = False

#     # These are the names of the objects we want to snapshot
#     object_names = ['opportunity', 'account', 'order']

#     # This is the day we are snapshotting on
#     today = str(datetime.datetime.now().date())

#     # Loop over all names
#     for object_name in object_names:
#         # When running in daemon mode, you don't want to barf on error, just
#         # write errors to log
#         try:
#             # Run the snapper and tell the log about it
#             snapper = ObjectSnapper(object_name, use_cache=use_cache)
#             snapper.run()
#             logger.info(f'date: {today}  sucess for: {object_name}')
#         except:  # noqa
#             # If an error happened, tell the log about it
#             logger.exception(f'date: {today} error for: {object_name}\n')


# @app.task('every 6 hours')
# def snap_runner():
#     """
#     Define a task to run in deamon mode
#     """
#     snap()


def run():
    module_list = [
        # 'tasks.snapshot_sfdc',
        # 'tasks.snapshot_predictor',
        'tasks.snapshot_dash_data',
    ]
    for module_name in module_list:
        mod = importlib.import_module(module_name)
        mod.run()


@click.command()
@click.option(
    '-d', '--daemon',
    default=False,
    is_flag=True,
    help='Run in daemon mode with scheduled downloads')
def main(daemon):
    """
    Handle the cli arguments and run in the appropriate mode
    """
    run()
    # if daemon:
    #     logger.info('Starting daemon mode')
    #     app.run()
    # else:
    #     logger.info('Running single snapshot')
    #     snap()


if __name__ == '__main__':
    main()
