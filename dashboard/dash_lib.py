from dateutil.relativedelta import relativedelta
from gtmarket.predictor import spread_values
from holoviews import opts
import datetime
import easier as ezr
import fleming
import gtmarket as gtm
import holoviews as hv
import io
import itertools
import locale
import numpy as np
import pandas as pd
import streamlit as st

locale.setlocale(locale.LC_ALL, 'en_US')
hv.extension('bokeh')
opts.defaults(opts.Area(width=800, height=400), tools=[])
opts.defaults(opts.Curve(width=800, height=400, tools=['hover']))

logger = ezr.get_logger('dash_data_runner')

USE_PG = True


def get_when():
    return fleming.floor(datetime.datetime.now(), minute=15)


def display(hv_obj):
    st.write(hv.render(hv_obj, backend='bokeh'))


def float_to_dollars(val):
    return locale.currency(val, grouping=True).split('.')[0]


def to_dollars(ser):
    return [float_to_dollars(x) for x in ser]


def to_percent(ser):
    return ['-' if x == '-' else f'{x}%' for x in ser]


def convert_dataframe(df):
    return df.to_csv().encode('utf-8')


def plot_frame(df, alpha=1, use_label=True, units='', include_total=True, ylabel='ACV'):  # pragma: no cover

    # import pdb; pdb.set_trace()

    import holoviews as hv
    colorcycle = itertools.cycle(ezr.cc.codes)
    c_list = []
    base = 0 * df.iloc[:, -1]
    for col in df.columns:
        if use_label:
            final = df[col].iloc[-1]
            label = col
            label = label.split('_')[0].title()
            label = f'{label} {final:0.1f}{units.upper()}'
            if include_total and (col == df.columns[-1]):
                label = label + f'  Total={df.iloc[-1].sum():0.1f}{units.upper()}'
        else:
            label = ''
        y = df[col]
        c = hv.Area(
            (df.index, y + base, base),
            kdims=['Date'],
            vdims=[f'{ylabel} ${units.upper()}', 'base'],
            label=label
        ).options(alpha=alpha, color=next(colorcycle), show_grid=True)
        c_list.append(c)
        c_list.append(hv.Curve((df.index, y + base)).options(color='black', alpha=.01))
        base += y
    return hv.Overlay(c_list).options(legend_position='top')


class BlobPrinter():
    def __init__(self):
        self.ps = gtm.PipeStats()

    @ezr.cached_container
    def _blob(self):

        mph = gtm.ModelParamsHist(use_pg=USE_PG)
        pm = mph.get_latest()
        blob = pm.to_blob()
        return blob

    @ezr.cached_container
    def blob(self):
        blob = self._blob
        for key in [
                'existing_pipe_model_with_expansion',
                'existing_pipe_model']:
            blob.pop(key)
        return blob

    def get_frames(self):
        ps = self.ps
        # return self.ps.get_opp_timeseries('num_sals', interval_days=30)
        ser_sal2sql = (100 * ps.get_conversion_timeseries(
            'sal2sql_opps', interval_days=90, bake_days=30)).round(1).iloc[-1]
        ser_sql2win = (100 * ps.get_conversion_timeseries(
            'sql2won_opps', interval_days=365, bake_days=90)).round(1).iloc[-1]
        ser_sal2win = (100 * ps.get_conversion_timeseries(
            'sal2won_opps', interval_days=365, bake_days=90)).round(1).iloc[-1]

        ser = ps.get_opp_timeseries('num_sals', interval_days=30).iloc[-1, :]
        ser['total'] = ser.sum()
        dfs = pd.DataFrame({'SALS / month': ser}).round().astype(int)

        dfd = (pd.DataFrame({'ACV': ps.get_mean_deal_size_timeseries().iloc[-1, :]})).round(1)
        dfd = dfd.loc[['enterprise', 'commercial', 'velocity'], :]

        dfwr = pd.DataFrame({
            'SAL⮕SQL': ser_sal2sql,
            'SQL⮕WON': ser_sql2win,
            'SAL⮕WON': ser_sal2win,
        })

        ser = (100 * ps.get_stage_win_rates_timeseries(interval_days=365, bake_days=90).iloc[-1, :]).round(1)
        ser = ser[['SAL', 'Discovery', 'Demo', 'Proposal', 'Negotiation']]
        dfswr = pd.DataFrame({'Win rate by stage': ser})

        today = fleming.floor(datetime.datetime.now(), day=1)
        dfo = ps.op.df_orders
        dfo.loc[:, 'market_segment'] = [ezr.slugify(m) for m in dfo.market_segment]

        dfo = dfo[(dfo.order_start_date <= today) & (dfo.order_ends > today)]
        ser = (12 * dfo.groupby(by='market_segment')[['mrr']].sum()).round(2).mrr
        ser = ser[['commercial', 'enterprise', 'velocity']]
        ser['combined'] = ser.sum()
        dfr = pd.DataFrame({'Current ARR': ser})

        dfn = ps.op.get_ndr_metrics()
        dfn = dfn[dfn.variable.isin(['ndr', 'renewed_pct', 'expanded_pct'])].set_index(
            ['market_segment', 'variable'])[['value']].unstack('variable')
        dfn.columns = dfn.columns.get_level_values(1)
        dfn = dfn.rename(columns={
            'ndr': '12-month NDR',
            'renewed_pct': '12-month Gross Retention',
            'expanded_pct': '12-month Expansion', })
        dfn = dfn.loc[
            ['commercial', 'enterprise', 'velocity', 'combined'],
            ['12-month Gross Retention', '12-month Expansion', '12-month NDR']
        ].round(1)

        dft = ps.get_conversion_timeseries(
            'sal2won_time', interval_days=365, bake_days=90).iloc[[-1], :].round().astype(int).T
        dft.columns = ['Days to Win']
        dft.index.name = None

        df_sales = dfs
        df_sales = df_sales.join(dfwr).drop('total')
        df_sales = df_sales.join(dft)
        df_sales = df_sales.join(dfd)
        sal_val = (.01 * df_sales['SAL⮕WON'] * df_sales['ACV']).round().astype(int)
        sql_val = (.01 * df_sales['SQL⮕WON'] * df_sales['ACV']).round().astype(int)
        value_rate = sal_val * df_sales['SALS / month']

        df_sales['SAL Val'] = to_dollars(sal_val)
        df_sales['SQL Val'] = to_dollars(sql_val)
        df_sales.loc[:, 'ACV'] = to_dollars(df_sales.ACV)
        df_sales.loc[:, 'SAL Value / Month'] = to_dollars(value_rate)

        for col in dfwr.columns:
            df_sales.loc[:, col] = to_percent(df_sales.loc[:, col])

        df_arr = dfr
        df_arr = df_arr.join(dfn)
        df_arr = df_arr.fillna('-')
        monthly_rate = (.01 * df_arr['12-month NDR']) ** (1 / 12) - 1
        # display(df_arr)
        df_arr['Value / Month'] = df_arr['Current ARR'] * monthly_rate

        df_arr.loc[:, 'Current ARR'] = to_dollars(df_arr['Current ARR'])
        df_arr.loc[:, '12-month Gross Retention'] = to_percent(df_arr.loc[:, '12-month Gross Retention'])
        df_arr.loc[:, '12-month Expansion'] = to_percent(df_arr.loc[:, '12-month Expansion'])
        df_arr.loc[:, '12-month NDR'] = to_percent(df_arr.loc[:, '12-month NDR'])
        df_arr.loc[:, 'Value / Month'] = to_dollars(df_arr['Value / Month'])
        df_arr.index.name = None

        dfswr.loc[:, 'Win rate by stage'] = to_percent(dfswr.loc[:, 'Win rate by stage'])

        return df_sales, dfswr, df_arr


class PredictorGetter:
    def __init__(self, pipe_stats_obj=None, include_sales_expansion=True):
        if pipe_stats_obj is None:
            pipe_stats_obj = gtm.PipeStats()
        self.ps = pipe_stats_obj
        self.include_sales_expansion = include_sales_expansion
        self.mph = gtm.ModelParamsHist(use_pg=USE_PG)

    def get_predicted_revenue(self, starting=None, ending_exclusive=None):
        if starting is None:
            starting = datetime.datetime.now()

        starting = pd.Timestamp(starting)
        starting = fleming.floor(starting, day=1)
        if ending_exclusive is None:
            ending_exclusive = starting + relativedelta(years=1)
        ending_exclusive = pd.Timestamp(ending_exclusive)
        deals = gtm.Deals(
            starting=starting,
            ending_exclusive=ending_exclusive,
            include_sales_expansion=self.include_sales_expansion,
            use_pg=USE_PG,
            model_params_hist=self.mph,

        )
        return deals.df_predicted

    @ezr.cached_container
    def _df_won(self):
        ps = gtm.PipeStats(pilots_are_new_biz=True, sales_expansion_are_new_biz=self.include_sales_expansion)
        df = ps.get_opp_timeseries(value='deal_acv', cumulative_since='12/31/2020')
        return df

    def get_won_revenue(self, starting=None, ending_inclusive=None):
        today = fleming.floor(datetime.datetime.now(), day=1)
        if ending_inclusive is None:
            ending_inclusive = today
        if starting is None:
            starting = fleming.floor(today, year=1)
        starting = pd.Timestamp(starting)
        ending_inclusive = pd.Timestamp(ending_inclusive)
        if starting < pd.Timestamp('1/1/2020'):
            raise ValueError('Can only get revenue since 1/1/2020')

        df = self._df_won
        df = df.loc[starting - relativedelta(days=1):ending_inclusive, :].sort_index()
        df = df - df.iloc[0, :]
        df = df.loc[starting:ending_inclusive, :]
        ind = pd.date_range(starting, ending_inclusive)
        df = df.reindex(index=ind)
        df = df.fillna(method='ffill')
        return df

    def get_forecast(self, since=None, today=None, ending_exclusive=None, separate_past_future=False):
        if today is None:
            today = fleming.floor(datetime.datetime.now(), day=1)
        if since is None:
            since = fleming.floor(today, year=1)
        if ending_exclusive is None:
            ending_exclusive = today + relativedelta(years=1)

        since, today, ending_exclusive = map(pd.Timestamp, [since, today, ending_exclusive])
        tomorrow = today + relativedelta(days=1)

        dfw = self.get_won_revenue(starting=since, ending_inclusive=today)
        dff = self.get_predicted_revenue(starting=tomorrow, ending_exclusive=ending_exclusive)

        dfw = dfw.loc[since:ending_exclusive, :]
        dff = dff.loc[since:ending_exclusive, :]

        if separate_past_future:
            return dfw, dff
        else:
            df = pd.concat([dfw, dff], axis=0)
            return df

    @ezr.cached_container
    def df_model_param_history(self):
        mph = gtm.ModelParamsHist(use_pg=USE_PG)
        df = mph.get_history()
        return df

    def get_prediction_history(self, since=None, ending_exclusive=None, units='m'):
        df = self.df_model_param_history
        min_time, max_time = [fleming.floor(d, day=1) for d in [df.time.min(), df.time.max()]]

        dates = pd.date_range(min_time, max_time)
        predictions = []
        for today in dates:
            dfw, dff = self.get_plot_frames_for_span(
                since=since, today=today, ending_exclusive=ending_exclusive, units=units)
            dff = dff + dfw.iloc[-1]
            predictions.append(dff.iloc[-1, :].sum())

        dfp = pd.DataFrame({'acv': predictions}, index=dates)
        ind = pd.date_range(dfp.index[0], ending_exclusive, inclusive='left')
        dfp = dfp.reindex(ind).fillna(method='ffill')
        return dfp

    def get_plot_frames_for_span(self, since=None, today=None, ending_exclusive=None, units='m'):
        units = units.lower()

        units_lookup = {
            'k': 1000,
            'm': 1e6,
            'u': 1,
        }

        scale = units_lookup[units]

        dfw, dff = self.get_forecast(
            since=since, today=today, ending_exclusive=ending_exclusive, separate_past_future=True)
        if not dff.empty:
            dfft = dff.T
            dfft.loc[:, dfw.index[-1]] = dfw.iloc[-1, :]
            dff = dfft.T.sort_index()

        dff = dff / scale
        dfw = dfw / scale

        # dfh = self.get_prediction_history(since=since, ending_exclusive=ending_exclusive, units=units)
        return dfw, dff  # , dfh

    def get_plot_frames(self, since=None, today=None, ending_exclusive=None, units='m'):
        units_lookup = {
            'k': 1000,
            'm': 1e6,
            'u': 1,
        }
        scale = units_lookup[units]
        dfw, dff = self.get_forecast(
            since=since, today=today, ending_exclusive=ending_exclusive, separate_past_future=True)
        dff = dff / scale
        dfw = dfw / scale
        dfh = self.get_prediction_history(since=since, ending_exclusive=ending_exclusive, units=units)
        return dfw, dff, dfh


class NDRGetter:
    def __init__(self):
        self.op = gtm.OrderProducts()
        self.ps = gtm.PipeStats()

    def _get_metrics(self, now, months=12):
        """
        Returns comparison metrics for the state of orders "now" compared to "months" ago
        """
        df = self.op.get_ndr_metrics(months=months, now=now)
        df.insert(0, 'date', now)

        return df

    @ezr.cached_container
    def df_metrics(self):
        dates = pd.date_range('1/1/2021', datetime.datetime.now())

        df_list = []
        for date in dates:
            df_list.append(self._get_metrics(date, months=12))

        df = pd.concat(df_list, axis=0, ignore_index=True, sort=False)
        return df


@st.cache
def get_ndr_metrics(when):
    return NDRGetter().df_metrics


class CSExpansion(ezr.pickle_cache_mixin):
    pkc = ezr.pickle_cache_state('reset')

    def __init__(self, today=None, ending_exclusive=None):
        self.ps = gtm.PipeStats()
        if today is None:
            today = fleming.floor(datetime.datetime.now(), day=1)
        self.today = pd.Timestamp(today)
        if ending_exclusive is None:
            ending_exclusive = self.today + relativedelta(years=1)

        self.ending = pd.Timestamp(ending_exclusive) - relativedelta(days=1)

    @ezr.cached_container
    def expansion_rate(self):
        # This gets the average expansion rate for existing contracted mrr
        # dfx = get_ndr_metrics(get_when())
        dfx = NDRGetter().df_metrics
        dfx = dfx[dfx.variable == 'expanded_pct']
        dfx = dfx.pivot(index='date', columns='market_segment', values='value')
        then = fleming.floor(datetime.datetime.now(), day=1) - relativedelta(years=1)
        dfx = dfx.loc[then:, :].drop('combined', axis=1)
        exp_ser = .01 * dfx.mean()
        return exp_ser

    @ezr.cached_container
    def df_cs_expansion(self):
        df = self.ps.loader.df_all
        df = df[df.created_date >= '1/1/2021']
        df = df[df.type == 'CS Expansion']
        return df

    @ezr.cached_container
    def df_cs_expansion_open(self):
        df = self.df_cs_expansion
        df = df[df.status_code == 0]
        df = df[df.acv.notnull()]

        # Ignore stuff with close dates in the past
        last_week = self.today - relativedelta(weeks=1)
        df = df[df.close_date >= last_week]

        # If the opp was opened after "today", I shouldn't know about it
        df = df[df.created_date <= self.today]

        # 90% of all won expansion opps closed within 90 days, so ignore opps set to close a long time from opening
        df['expected_days_open'] = (df.close_date - df.created_date).dt.days
        df = df[df.expected_days_open <= 90]

        return df

    @ezr.cached_container
    def win_rate(self):
        # Get the win rate of cs expansion opps
        df = self.df_cs_expansion
        df = df[df.status_code != 0]
        df = df.groupby(by=['market_segment', 'status_code'])[['opportunity_id']].count().unstack('status_code')
        df.columns = df.columns.get_level_values(1)
        df['win_rate'] = df.loc[:, 2] / df.sum(axis=1)
        ser_win_rate = df.win_rate
        return ser_win_rate

    @ezr.cached_container
    def expanding_account_set(self):
        df = self.df_cs_expansion_open
        accounts = df.account_id.drop_duplicates()
        return set(accounts)

    @ezr.cached_container
    def df_cs_expansion_expected_from_pipe(self):
        df = self.df_cs_expansion_open
        df = df[['opportunity_name', 'market_segment', 'close_date', 'acv', 'expected_days_open', 'stage']]
        df['discounted_acv'] = df.market_segment.map(self.win_rate) * df.acv

        df = df.groupby(
            by=['close_date', 'market_segment'])[['discounted_acv']].sum().unstack('market_segment').fillna(0)
        df.columns = df.columns.get_level_values(1)
        return df

    @ezr.cached_container
    def df_cs_expansion_from_current_orders(self):
        dfo = self.ps.op.df_orders
        dfo = dfo[['account_id', 'order_start_date', 'market_segment', 'mrr']]

        # Ignore contributions for accounts with open axpansion opps
        dfo = dfo[~dfo.account_id.isin(self.expanding_account_set)]

        dfo['acv'] = 12 * dfo.mrr

        dfo = dfo.groupby(by=['order_start_date', 'market_segment'])[['acv']].sum().unstack('market_segment').fillna(0)
        dfo.columns = [ezr.slugify(c) for c in dfo.columns.get_level_values(1)]

        starting, ending = dfo.index[0], self.ending
        dates = pd.date_range(starting, ending, name='date')

        # Note that in spreading the values of a year, I'm only including first year sales expansion
        # estimates.  Everything else gets accounted for in NDR
        dfo = dfo.reindex(dates).fillna(0)
        for col in dfo.columns:
            dfo.loc[:, col] = spread_values(dfo.loc[:, col] * self.expansion_rate[col], 1 * 365)

        dfo = dfo.loc[self.today:, :].cumsum()
        return dfo

    @ezr.pickle_cached_container()
    def df_cs_expansion_forecast(self):
        dfo = self.df_cs_expansion_from_current_orders
        dfp = self.df_cs_expansion_expected_from_pipe
        dfp = dfp.reindex(dfo.columns, axis=1).fillna(0)
        dfp = dfp.reindex(dfo.index).fillna(0)
        dfp = dfp.cumsum()

        dfo = dfo + dfp
        return dfo


class ARRGetter(ezr.pickle_cache_mixin):

    pkc = ezr.pickle_cache_state('reset')

    def __init__(self, starting=None, ending_exclusive=None):
        # Get a reference for today
        self.today = fleming.floor(datetime.datetime.now(), day=1)
        self.ps = gtm.PipeStats()

        if starting is None:
            starting = fleming.floor(self.today, year=1)
        self.starting = starting

        if ending_exclusive is None:
            ending_exclusive = self.starting + relativedelta(years=1)
        self.ending_exclusive = pd.Timestamp(ending_exclusive)
        self.ending_inclusive = self.ending_exclusive - relativedelta(days=1)
        self.mph = gtm.ModelParamsHist(use_pg=USE_PG)

    def _get_new_biz_frame(self, today, ending_exclusive):
        deals = gtm.Deals(
            starting=today,
            ending_exclusive=self.ending_exclusive,
            include_sales_expansion=True,
            use_pg=USE_PG,
            model_params_hist=self.mph
        )
        return deals.df_predicted

    @ezr.cached_container
    def df_new_biz(self):
        return self._get_new_biz_frame(today=self.today, ending_exclusive=self.ending_exclusive)

    def get_arr_history_frame(self, today=None):
        if today is None:
            today = self.today
        today = pd.Timestamp(today)

        # Use the cs plotter to get average of last 30 days of NDR
        dfm = NDRGetter().df_metrics

        # Based on yearly NDR, compute an exponential time constant for each segment
        dfm = dfm[dfm.variable == 'ndr']
        dfm = .01 * dfm.pivot(index='date', columns='market_segment', values='value')
        tau = dfm.iloc[-30:, :].mean()
        tau = -365 / np.log(tau)

        # Get the orderproducts frame
        # dfo = self.cs_plotter.op.df_orders
        dfo = self.ps.op.df_orders
        dfo.loc[:, 'market_segment'] = [ezr.slugify(m) for m in dfo.market_segment]

        # Make a range of dates from starting till today
        dates = pd.date_range(self.starting, today)

        # Populate each day with the amount of ARR on that day
        rec_list = []
        for date in dates:
            rec = {'date': date}
            batch = dfo[(dfo.order_start_date <= date) & (dfo.order_ends > date)]
            rec.update((12 * batch.groupby(by='market_segment').mrr.sum()).to_dict())
            rec_list.append(rec)

        # Make a frame out of the ARR
        df = pd.DataFrame(rec_list)
        df = df.set_index('date')

        # Extend the frame one year out into the future
        df = df.reindex(
            pd.date_range(self.starting, self.ending_inclusive, name='date')).fillna(method='ffill').reset_index()

        # Compute how many days into the future each day corresponds to
        df['days'] = np.maximum(0, (df.date - today).dt.days)
        segments = [c for c in df.columns if c not in ['days', 'date']]

        # Discount the current ARR into the future using the NDR for each segment
        for segment in segments:
            df.loc[:, segment] = df.loc[:, segment] * np.exp(-df.days / tau[segment])

        # Get rid of cols I don't want
        df = df.drop('days', axis=1)
        df = df.set_index('date')
        return df

    @ezr.cached_container
    def df_arr_history(self):
        return self.get_arr_history_frame()

    def get_prediction_history_frame(self):
        #TODO: THIS METHOD MIGHT NOT BE USED.

        rec_list = []
        for date in pd.date_range('6/2/2022', self.today):
            dfc = self.get_arr_history_frame(today=date)
            deals = gtm.Deals(
                starting=date,
                ending_exclusive=self.ending_exclusive,
                include_sales_expansion=True,
                use_pg=USE_PG,
            )
            dfp = deals.df_predicted
            # It's important you create a fresh csx here, so that it's anchored to date
            csx = CSExpansion(self.ps, self.cs_plotter, date, ending_exclusive=self.ending_exclusive)
            dfx = csx.df_cs_expansion_forecast

            current_arr = dfc.loc[self.ending_inclusive, :]
            rec = current_arr

            rec = rec + dfp.loc[self.ending_inclusive, :]
            rec = rec + dfx.loc[self.ending_inclusive, :]
            rec['date'] = date

            rec_list.append(rec)

        dfh = pd.DataFrame(rec_list).set_index('date').sort_index()
        dfh = pd.DataFrame({'arr': dfh.sum(axis=1)})
        return dfh

    @ezr.pickle_cached_container()
    def df(self):
        dfc = self.df_arr_history
        dfn = self.df_new_biz
        dfn = dfn.reindex(dfc.index).fillna(0)

        dfx = CSExpansion(today=self.today, ending_exclusive=self.ending_exclusive).df_cs_expansion_forecast
        dfx = dfx.loc[dfx.index[0]:dfx.index[-1]]
        dfx = dfx.reindex(dfc.index).fillna(0)

        df = dfn + dfc + dfx
        return df


@st.cache
def get_arr_timeseries(when, starting=None, ending_exclusive=None):
    return ARRGetter(starting=starting, ending_exclusive=ending_exclusive).df


class SALGetter:
    def __init__(self, pipe_stats_obj=None):
        if pipe_stats_obj is None:
            self.ps = gtm.PipeStats()
        else:
            self.ps = pipe_stats_obj

    @ezr.pickle_cached_container()
    def df_daily_actuals(self):
        starting, ending = pd.Timestamp('1/1/2021'), fleming.floor(datetime.datetime.now(), day=1)

        ps = self.ps
        dfa = ps.df_new_biz
        dfa = dfa[['created_date', 'market_segment']]
        dfa['num_opps'] = 1
        dfa = dfa[dfa.created_date >= '1/1/2021']
        dfa = dfa.pivot_table(index='created_date', columns='market_segment', values='num_opps', aggfunc=np.sum)
        dfa = dfa.drop('unknown', axis=1, errors='ignore').fillna(0)
        ind = pd.date_range(starting, ending, freq='D', name='created_date')
        dfa = dfa.reindex(index=ind, fill_value=0)
        return dfa

    def get_rolling_created(self, rolling_days=30, smoothing_degree=15):
        df = self.df_daily_actuals
        cols = list(df.columns)
        df['weekday'] = df.index.weekday
        df = df[df.weekday < 5]
        df['days'] = np.arange(len(df))
        df = df.rolling(rolling_days).sum().dropna()
        fitter = ezr.BernsteinFitter(monotonic=False, match_left=False, match_right=False, non_negative=True)
        for col in cols:
            df.loc[:, col + '_fit'] = fitter.fit_predict(df.loc[:, 'days'], df.loc[:, col].values, smoothing_degree)
        df = df.drop(['days', 'weekday'], axis=1)
        return df


class RateGetter:
    def __init__(self, pipe_stats_obj=None):
        if pipe_stats_obj is None:
            pipe_stats_obj = gtm.PipeStats()
        self.ps = pipe_stats_obj

    def _get_single_conversion(self, rate_name, bake_days=30, interval_days=90):
        ps = self.ps
        dfc = ps.get_conversion_timeseries(rate_name, interval_days, bake_days=bake_days) * 100
        dfc = dfc.loc['1/1/2021':, :]
        return dfc

    @ezr.cached_container
    def df_sal2sql(self):
        return self._get_single_conversion('sal2sql_opps', bake_days=30, interval_days=90)

    @ezr.cached_container
    def df_sql2won(self):
        return self._get_single_conversion('sql2won_opps', bake_days=90, interval_days=365)

    @ezr.cached_container
    def df_sal2won(self):
        return self._get_single_conversion('sal2won_opps', bake_days=90, interval_days=365)


class CSGetter(ezr.pickle_cache_mixin):

    pkc = ezr.pickle_cache_state('reset')

    def __init__(self, months=12):
        self.op = gtm.OrderProducts()
        self.months = months

    def _get_metrics(self, now, months=12):
        """
        Returns comparison metrics for the state of orders "now" compared to "months" ago
        """
        # Now is the date for which you want to compute metrics
        now = pd.Timestamp(now)

        # You will be comparing ARR between now and this many months ago to compute metrics
        then = now - relativedelta(months=months)

        # Get all orders and standardize them
        df = self.op.df_orders[['account_id', 'order_start_date', 'order_ends', 'mrr', 'market_segment']]
        df['market_segment'] = [ezr.slugify(s) for s in df.market_segment]

        # Create two frames.  One for "now" and one for "then"
        df_now = df[(df.order_start_date <= now) & (df.order_ends >= now)]
        df_then = df[(df.order_start_date <= then) & (df.order_ends >= then)]

        # Combine all revenue for a given account into a single record
        def agg_by_account(df):
            df = df.groupby(by=['account_id', 'market_segment'])[['mrr']].sum().reset_index()
            return df

        df_now = agg_by_account(df_now)
        df_then = agg_by_account(df_then)

        # Join the accounts that existed "then" with what exists "now".  Fill revenue with 0 if they don't exist "now"
        dfj = pd.merge(
            df_then, df_now, on=['account_id', 'market_segment'], how='left', suffixes=['_ref', '_ret']).fillna(0)

        # This is the base from which we will compute all metrics
        dfj['base'] = dfj.mrr_ref

        # Exppanded revenue is any revenue over and above the base an org had "back then"
        dfj['expanded'] = np.maximum(0, dfj.mrr_ret - dfj.mrr_ref)

        # Reduction revenue is any deficit below the base that an org had "back then"
        dfj['gross_churn'] = np.maximum(0, dfj.mrr_ref - dfj.mrr_ret)

        # Renewed is the amount of revenue we have "now" that we also had "back then"
        dfj['renewed'] = np.minimum(dfj.mrr_ref, dfj.mrr_ret)

        # If there is no revenue "now", that means the or churned
        dfj['churned'] = [ref if int(round(ret)) == 0 else 0 for (ref, ret) in zip(dfj.mrr_ref, dfj.mrr_ret)]

        # Reduced included churned revenue in the way computed it.  So remove that churned revenue
        dfj.loc[:, 'reduced'] = dfj.gross_churn - dfj.churned

        # Sum all revenues by market segment and convert MRR to ARR
        dfg = dfj.drop(['account_id'], axis=1).groupby(by=['market_segment']).sum()
        dfg = dfg * 12

        # Add a fake new market segment of "all"
        dfg = dfg.T
        dfg['all'] = dfg.sum(axis=1)
        dfg = dfg.T

        # Net dollar is looking at the sum of all companies we had under contract "then" and
        # looking at the percent increase/decrease to everything they are paying us now
        dfg['ndr'] = 100 * dfg.mrr_ret / dfg.mrr_ref

        # I'd also like to know more granular percentages with respect to the base
        for metric in ['expanded', 'reduced', 'renewed', 'churned', 'gross_churn']:
            dfg[f'{metric}_pct'] = 100 * dfg[metric] / dfg.base

        # Clean up some variables I don't care about
        dfg = dfg.drop(['mrr_ref', 'mrr_ret'], axis=1).reset_index()

        # Transform the data into a melted format and insert date
        dfg = pd.melt(dfg, id_vars=['market_segment'])
        dfg.insert(0, 'date', now)
        return dfg

    @ezr.pickle_cached_container()
    def df_metrics(self):
        dates = pd.date_range('1/1/2021', datetime.datetime.now())

        df_list = []
        for date in dates:
            df_list.append(self._get_metrics(date, months=12))

        df = pd.concat(df_list, axis=0, ignore_index=True, sort=False)
        df = df.set_index(['date', 'market_segment', 'variable'])
        return df

    @ezr.pickle_cached_container()
    def df_contract_months(self):

        def dollar_weighted_duration(op, now):
            # Now is the date for which you want to compute metrics
            now = pd.Timestamp(now)

            # Get all orders and standardize them
            df = op.df_orders
            df = df[(df.order_start_date <= now) & (df.order_ends >= now)]

            df = df[['mrr', 'market_segment', 'months']]
            df.loc[:, 'market_segment'] = [ezr.slugify(s) for s in df.market_segment]
            df['weight_and_value'] = df.months * df.mrr
            df['weight'] = df.mrr

            dfg = df.groupby(by='market_segment')[['weight_and_value', 'weight']].sum()
            dfg['duration_months'] = dfg.weight_and_value / dfg.weight
            rec = dfg.duration_months.to_dict()
            rec['date'] = now
            return rec

        dates = pd.date_range('1/1/2021', datetime.datetime.now())

        rec_list = []
        for date in dates:
            rec_list.append(dollar_weighted_duration(self.op, date))
        df = pd.DataFrame(rec_list).set_index('date')

        return df

    @ezr.pickle_cached_container()
    def df_contract_acv(self):

        def mean_contract_acv(now):
            # Now is the date for which you want to compute metrics
            now = pd.Timestamp(now)

            # Get all orders and standardize them
            df = self.op.df_orders
            df = df[(df.order_start_date <= now) & (df.order_ends >= now)]

            df = df[['account_id', 'mrr', 'market_segment']]
            df.loc[:, 'market_segment'] = [ezr.slugify(s) for s in df.market_segment]
            df = df.groupby(by=['account_id', 'market_segment'])[['mrr']].sum().reset_index()

            rec = (12 * df.groupby(by='market_segment').mrr.mean()).to_dict()
            rec['date'] = now
            return rec

        dates = pd.date_range('1/1/2021', datetime.datetime.now())

        rec_list = []
        for date in dates:
            rec_list.append(mean_contract_acv(date))
        df = pd.DataFrame(rec_list).set_index('date').round()

        return df

    def _compute_logo_churn_timeseries(df):
        df = df.pivot_table(index='date', columns='outcome', values='num').fillna(0)
        df = df.resample('D').asfreq().fillna(0)
        df = df.rolling(365).sum().dropna()
        df['total'] = df.sum(axis=1)
        df['logo_retention'] = df.retained / df.total
        df = df.loc[:datetime.datetime.now(), :]
        return df

    @ezr.cached_container
    def df_logo_changes(self):
        df = self.op.df_changes
        df = df[['date', 'outcome', 'market_segment']].copy()
        df.loc[:, 'outcome'] = [(v if v == 'churned' else 'retained') for v in df.outcome]
        df['num'] = 1
        return df

    @ezr.cached_container
    def segments(self):
        return sorted(self.df_logo_changes.market_segment.unique())

    @ezr.cached_container
    def df_logo_renewals(self):
        df_list = []
        for market_segment, batch in self.df_logo_changes.groupby('market_segment'):
            batch = batch.sort_values('date')
            df = self._compute_logo_churn_timeseries_for_segment(batch)
            df.insert(0, 'market_segment', market_segment)
            df_list.append(df)

        df = pd.concat(df_list, axis=0, ignore_index=True, sort=False)
        df = df.sort_values(by=['market_segment', 'date'])
        return df

    def _compute_logo_churn_timeseries_for_segment(self, df):
        df = df.pivot_table(index='date', columns=['outcome'], values='num').fillna(0)
        df = df.resample('D').asfreq().fillna(0)
        df = df.rolling(365).sum().dropna()
        df = df.loc[:datetime.datetime.now(), :]
        df.columns.name = None
        return df.reset_index()


class DashData:
    def __init__(self, use_pg=USE_PG):
        sqlite_file = '/tmp/dash_play.sqlite'
        if use_pg:
            self.mm = ezr.MiniModelPG(overwrite=False, read_only=False)
        else:
            1 / 0
            self.mm = ezr.MiniModelSqlite(file_name=sqlite_file, overwrite=False, read_only=False)

        self.methods = [
            'process_arr_time_series',
            'process_sales_progress',
            'process_sales_timeseries',
            'process_process_stats',
            'process_sal_creation_rate',
            'process_conversion_rates',
            'process_cs_metrics'
        ]

    def _enable_caches(self):
        logger.info('enabling caches')
        gtm.PipeStats.enable_pickle_cache()
        CSExpansion.enable_pickle_cache()
        ARRGetter.enable_pickle_cache()
        CSGetter.enable_pickle_cache()

    def _disable_caches(self):
        logger.info('disabling caches')
        gtm.PipeStats.disable_pickle_cache()
        CSExpansion.disable_pickle_cache()
        ARRGetter.disable_pickle_cache()
        CSGetter.disable_pickle_cache()

    def _clear_caches(self):
        logger.info('clearing caches')
        gtm.PipeStats.clear_all_default_pickle_cashes()

    def run(self):
        try:
            self._clear_caches()
            self._enable_caches()
            for method in self.methods:
                logger.info(f'running: {method}')
                getattr(self, method)()
        except:
            raise
        finally:
            self._disable_caches()
            self._clear_caches()

    def _save_frame(self, name, df, save_index=True):
        if save_index:
            df = df.reset_index(drop=False)
        data = df.to_csv(index=False)
        date = fleming.floor(datetime.datetime.now(), day=1)
        dfs = pd.DataFrame([{'date': date, 'data': data}])
        self.mm.upsert(name, ['date'], dfs)

    def process_arr_time_series(self):
        df = ARRGetter(starting=None, ending_exclusive=None).df
        self._save_frame('dash_arr_time_series', df)

    def process_sales_progress(self):
        today = fleming.floor(datetime.datetime.now(), day=1)
        ending_exclusive = '1/1/2024'
        since = '1/1/2023'

        pg = PredictorGetter()

        dfw, dff, _ = pg.get_plot_frames(since=since, today=today, ending_exclusive=ending_exclusive, units='u')
        dfx = pd.DataFrame({
            'won': dfw.iloc[-1],
            'remaining': dff.iloc[-1],
        }).T
        dfx['total'] = dfx.sum(axis=1)
        dfx = dfx.T
        dfx['total'] = dfx.sum(axis=1)
        dfx.columns.name = '2022 sales'

        dfx = dfx.round()

        self._save_frame('dash_sales_progress', dfx)

    def process_sales_timeseries(self):
        pg = PredictorGetter()
        df_won, df_forecast, df_pred_hist = pg.get_plot_frames(since='1/1/2022', ending_exclusive='1/1/2023', units='u')
        self._save_frame('dash_sales_won_timeseries', df_won)
        self._save_frame('dash_sales_forecast_timeseries', df_forecast)
        self._save_frame('dash_sales_prediction_history', df_pred_hist)

    def process_process_stats(self):
        bp = BlobPrinter()
        df_sales, df_stage_win_rate, df_arr = bp.get_frames()

        self._save_frame('dash_sales_stats', df_sales)
        self._save_frame('dash_sales_stats_stage_win_rate', df_stage_win_rate)
        self._save_frame('dash_sales_stats_arr', df_arr)

    def process_sal_creation_rate(self):
        getter = SALGetter()
        df = getter.get_rolling_created(rolling_days=30, smoothing_degree=60)
        df.index.name = 'date'
        self._save_frame('dash_sal_creation_rate', df)

    def process_conversion_rates(self):
        getter = RateGetter()
        self._save_frame('dash_sal2sql', getter.df_sal2sql)
        self._save_frame('dash_sql2won', getter.df_sql2won)
        self._save_frame('dash_sal2won', getter.df_sal2won)

    def process_cs_metrics(self):
        getter = CSGetter()
        self._save_frame('dash_cs_metrics', getter.df_metrics)
        self._save_frame('dash_contract_acv', getter.df_contract_acv)
        self._save_frame('dash_contract_months', getter.df_contract_months)
        self._save_frame('dash_logo_renewals', getter.df_logo_renewals, save_index=False)

    def get_latest(self, name):
        if name not in self.mm.table_names:
            raise ValueError(f'{name} not in {self.mm.table_names}')

        df = self.mm.query(f"""
        SELECT
            data
        FROM
            {name}
        ORDER BY
            date DESC
        LIMIT 1
        """)
        data = df.data.iloc[0]
        dfo = pd.read_csv(io.StringIO(data))
        return dfo

    def get_latest_time(self):
        name = 'dash_sales_won_timeseries'
        df = self.mm.query(f"""
        SELECT
            date
        FROM
            {name}
        ORDER BY
            date DESC
        LIMIT 1
        """)
        date = pd.Timestamp(df.date.iloc[0]).date()
        return date
