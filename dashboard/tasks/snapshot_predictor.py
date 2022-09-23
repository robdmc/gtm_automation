import easier as ezr
import gtmarket as gtm
import os

# Will want to write to logs for daemon process
logger = ezr.get_logger('snap_forecast')


def run():
    mp = gtm.ModelParams()
    mp.fit()
    mp_hist = gtm.ModelParamsHist()
    mp_hist.store(mp)
    logger.info('success for fitting/updating model params')

    translator = gtm.GTMTranslator(local_file=os.path.expanduser('~/data/gtm_databasae.sqlite'))
    translator.postgres_to_sqlite()
    logger.info('success backing up gtm database')
