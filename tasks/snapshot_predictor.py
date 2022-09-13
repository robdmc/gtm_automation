import easier as ezr
import gtmarket as gtm

# Will want to write to logs for daemon process
logger = ezr.get_logger('snap_forecast')


def run():
    logger.info('Fitting model params')
    mp = gtm.ModelParams()
    mp.fit()
    mp_hist = gtm.ModelParamsHist()
    mp_hist.store(mp)
