#! /usr/bin/env python

import click
import easier as ezr
from rocketry import Rocketry
logger = ezr.get_logger('snapper')

app = Rocketry()


def snap():
    logger.info('Runnin my snap')


@app.task('every 6 seconds')
def snap_runner():
    snap()

@click.command()
@click.option('-d', '--daemon', default=False, is_flag=True, help='Run in daemon mode with scheduled downloads')
def main(daemon):
    if daemon:
        app.run()
    else:
        snap()


# def run_daemon():
#     app = Rocketry()

#     def 



if __name__ == '__main__':
    main()





# @app.task('every 6 hours')
