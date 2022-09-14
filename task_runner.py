#! /usr/bin/env python

import click
import importlib
from rocketry import Rocketry
import easier as ezr

# Will want to write to logs for daemon process
logger = ezr.get_logger('task_runner')
app = Rocketry()





# @app.task('every 6 hours')
# def snap_runner():
#     """
#     Define a task to run in deamon mode
#     """
#     snap()


@app.task('every 3 hours')
def run():
    module_list = [
        'tasks.snapshot_sfdc',
        'tasks.snapshot_predictor',
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
    if daemon:
        logger.info('Starting daemon mode')
        app.run()
    else:
        logger.info('Running single snapshot')
        run()


if __name__ == '__main__':
    main()
