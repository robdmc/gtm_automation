from holoviews import opts
from scipy import stats
import easier as ezr
import holoviews as hv
import locale
import numpy as np
import streamlit as st

from dash_lib import (
    DashData,
    display,
    get_when,
)
locale.setlocale(locale.LC_ALL, 'en_US')

hv.extension('bokeh')
opts.defaults(opts.Area(width=800, height=400), tools=[])
opts.defaults(opts.Curve(width=800, height=400, tools=['hover']))
from hvplot import pandas


@st.cache
def get_latest_date(when):
    return DashData().get_latest_time()


@st.cache
def get_frames(when):
    dd = DashData()
    df_metrics = dd.get_latest('dash_cs_metrics')
    df_acv = dd.get_latest('dash_contract_acv')
    df_months = dd.get_latest('dash_contract_months')
    df_logo_retention = dd.get_latest('dash_logo_renewals')

    def transform_to_date(df, col):
        df.loc[:, col] = df.loc[:, col].astype(np.datetime64)
        return df

    df_metrics = transform_to_date(df_metrics, 'date')
    df_acv = transform_to_date(df_acv, 'date')
    df_months = transform_to_date(df_months, 'date')
    df_logo_retention = transform_to_date(df_logo_retention, 'date')

    return df_metrics, df_acv, df_months, df_logo_retention


class CSPlotter:
    def __init__(self, df_metrics, df_acv, df_months, df_logo_rentention):
        self.df_metrics = df_metrics
        self.df_acv = df_acv
        self.df_months = df_months
        self.df_logo_retention = df_logo_rentention

    def _plot_time_series(self, df, col, label, final_values, units='', multiplier=1):
        name_mapper = {k: f'{k} {v:0.1f}{units}'.strip() for (k, v) in final_values.items()}
        dfx = df[df.variable == col].pivot_table(index='date', columns='market_segment', values='value', aggfunc=np.sum)
        dfx = multiplier * dfx.rename(columns=name_mapper)
        # st.write(dfx)
        # st.write(dfx.reset_index().dtypes.astype(str))
        c = dfx.hvplot(ylabel=label).options(legend_position='top', show_grid=True)
        return c

    def _get_final_values(self, df, col, multiplier=1):
        dfx = df[df.variable == col].pivot_table(index='date', columns='market_segment', values='value', aggfunc=np.sum)
        out = (multiplier * dfx.iloc[-1, :]).to_dict()
        return out

    def get_retention_plots(self):

        df = self.df_metrics

        col = 'ndr'
        label = 'Rolling 12-month Net-Dollar-Retention (%)'
        units = '%'
        final_values = self._get_final_values(df, col)
        c_ndr = self._plot_time_series(df, col, label, final_values, units=units)

        col = 'expanded_pct'
        label = 'Rolling 12-month Expansion APR (%)'
        units = '%'
        final_values = self._get_final_values(df, col)
        c_ex = self._plot_time_series(df, col, label, final_values, units=units)

        col = 'renewed_pct'
        label = 'Rolling 12-month Dollar Retention Rate (%)'
        units = '%'
        final_values = self._get_final_values(df, col)
        c_ren = self._plot_time_series(df, col, label, final_values, units=units)

        col = 'reduced_pct'
        label = 'Rolling 12-month Reduction Rate (%)'
        units = '%'
        final_values = self._get_final_values(df, col, multiplier=-1)
        c_red = self._plot_time_series(df, col, label, final_values, units=units, multiplier=-1)

        col = 'churned_pct'
        label = 'Rolling 12-month Churn Rate (%)'
        units = '%'
        final_values = self._get_final_values(df, col, multiplier=-1)
        c_churn = self._plot_time_series(df, col, label, final_values, units=units, multiplier=-1)

        col = 'gross_churn_pct'
        label = 'Rolling 12-month Gross Churn (%)'
        units = '%'
        final_values = self._get_final_values(df, col, multiplier=-1)
        c_gross_churn = self._plot_time_series(df, col, label, final_values, units=units, multiplier=-1)


        return c_ndr, c_ex, c_ren, c_red, c_churn, c_gross_churn

    def _plot_logo_retention_for_segment(self, df, segment, color):
        df = df[df.market_segment == segment].copy()
        df.loc[:, 'date'] = df.date.astype(np.datetime64)
        df = df.set_index('date')
        dist = stats.beta(a=df.retained.values + 1, b=df.churned.values + 1)
        df['lower'], df['best'], df['upper'] = dist.ppf(.1), dist.ppf(.5), dist.ppf(.9)
        df.loc[:, ['lower', 'best', 'upper']] = 100 * df.loc[:, ['lower', 'best', 'upper']]

        c_list = []
        c = hv.Area((df.index, df.lower, df.upper), kdims='Date', vdims=['Logo Retention Rate (%)', 'b']).options(
            alpha=.1, color=color)
        c_list.append(c)
        c = hv.Curve((df.index, df.best), label=segment).options(color=color, show_grid=True)
        c_list.append(c)
        return hv.Overlay(c_list).options(legend_position='top')

    def get_logo_rention_plot(self):
        df = self.df_logo_retention
        c_list = []
        segments = sorted(df.market_segment.unique())
        for color, seg in zip(ezr.cc, segments):
            c_list.append(self._plot_logo_retention_for_segment(df, seg, color))

        return hv.Overlay(c_list).options(legend_position='top')

    def plot(self):
        display(self.plot_ndr_time_series())
        display(self.plot_duration_time_series())
        display(self.plot_contracted_acv())


as_of = get_latest_date(get_when())
df_metrics, df_acv, df_months, df_logo_retention = get_frames(get_when())

st.title('CS Trends')
st.markdown(f'### as of {as_of}')
plotter = CSPlotter(df_metrics, df_acv, df_months, df_logo_retention)


c_ndr, c_ex, c_ren, c_red, c_churn, c_gross_churn = plotter.get_retention_plots()
c_logo_ret = plotter.get_logo_rention_plot()

st.markdown('---')
st.markdown('## Net Dollar Retention')
with st.expander("See explanation"):
    st.markdown("""
            Net Dollar Retention is defined with the following statement: Look at how much money
            our customers were paying us one year ago.  Compare that number with the amount those
            same customers are paying us today.  The ratio of those two numbers is the 12-month
            Net Dollar Retention, (NDR).  Here are the steps taken to make this graph.

            * For each day:
            * Define `now` as today
            * Define `then` as exactly 12 months ago
            * Find all orders with `order_start_date` and `order_ends` that bracket the `then` date.
            Sum up their MRR and call it `mrr_then`.  Also store their account_ids.
            * Now find all orders with `order_start_date` and `order_ends` that bracket the `now` date.
            * Limit the `now` orders to conly contain records that match the account_ids from the `then` set.
            * Sum up the revenue of the `now` set and call it `mrr_now.`
            * This will be the amount of money that accounts active `then` are paying us `now`.
            * Compute NDR as the ratio `mrr_now` / `mrr_then`.
            * Plot a point on the graph for the `now` date with this NDR value.
        """)
display(c_ndr)


st.markdown('---')
st.markdown('## Expansion')
display(c_ex)

st.markdown('---')
st.markdown('## Gross Churn')
display(c_gross_churn)


st.markdown('---')
st.markdown('## Revenue Fraction lost to Churned Accounts')
with st.expander("See explanation"):
    st.markdown("""
         This is the percentage of our revenue that we lost to actually churned customers.
         These numbers ignore reduced contract values.
    """)
display(c_churn)

st.markdown('---')
st.markdown('## Revenue Fraction lost to Reduced Accounts')
with st.expander("See explanation"):
    st.markdown("""
        This is the percentage of our revenue that we lost to customers downsizing.
    """)
display(c_red)


st.markdown('---')
st.markdown('## Logo Retention Rate')

with st.expander("See explanation"):
    st.markdown("""
        This graph estimates our logo retention rate based on the previous 365 days.


        There are so many different scenarios in managing our active customer base, that computing
        a logo retention rate is pretty involved.  A crude sketch of the logic is this:
        * For every order, create two events.  An "order_created" event, and an "order_expired" event.
        * For each logo
            * Sort the events by date
            * Collapse all events that happen within a grace period (currently 15 days) into a single event.
              (What this means is that if an order expires, and nine days later a new order is created, we will
              fake the event associated with the new order created to match the date of the previous order expiration).
              This is only done in the volatile computation of events and the result is not stored anywhere.
            * Compare the pre-post MRR for each event to determine if the logo got expanded, reduced, or stayed the
              same (renewed).
            * Run additional checks to mark reduced orders as churned if the order was marked as churned.
        * Then, using these collapsed dates
            * Loop over all dates.
            * Count how many churned and non-churned events happened in the past 365 days
            * Use these counts to determine the beta distribution defined by the "win" and "lost" counts.
            * Use quantiles `[0.1, 0.5, 0.9]` to determine the best estimate and uncertaines of the logo renewal rate.
    """)

display(c_logo_ret)
