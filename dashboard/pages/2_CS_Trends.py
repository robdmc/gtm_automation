import streamlit as st
# from streamlit import caching
import gtmarket as gtm
import pandas as pd
import numpy as np
import fleming
import datetime
import copy
import locale
locale.setlocale(locale.LC_ALL, 'en_US')
import easier as ezr
from dateutil.relativedelta import relativedelta
import holoviews as hv
import datetime
import itertools
from holoviews import opts
from hvplot import pandas

import dash_lib as dl
from dash_lib import (
    DashData,
    convert_dataframe,
    float_to_dollars,
    to_dollars,
    display,
    plot_frame,
    get_when,
)
hv.extension('bokeh')
opts.defaults(opts.Area(width=800, height=400), tools=[])
opts.defaults(opts.Curve(width=800, height=400, tools=['hover']))


@st.cache
def get_latest_date(when):
    return DashData().get_latest_time()

@st.cache
def get_frames(when):
    dd = DashData()
    df_metrics = dd.get_latest('dash_cs_metrics')
    df_acv = dd.get_latest('dash_contract_acv')
    df_months = dd.get_latest('dash_contract_months')

    def transform_to_date(df, col):
        df.loc[:, col] = df.loc[:, col].astype(np.datetime64)
        return df

    df_metrics = transform_to_date(df_metrics, 'date')
    df_acv = transform_to_date(df_acv, 'date')
    df_months = transform_to_date(df_months, 'date')

    return df_metrics, df_acv, df_months





class CSPlotter:
    def __init__(self, df_metrics, df_acv, df_months):
        self.df_metrics = df_metrics
        self.df_acv = df_acv
        self.df_months = df_months
        
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
        return c_ndr, c_ex, c_ren, c_red, c_churn

        
        # return hv.Layout(c_list)#.cols(1).options(shared_axes=False)
    
    def plot_duration_time_series(self):
    
        # def dollar_weighted_duration(op, now):
        #     # Now is the date for which you want to compute metrics
        #     now = pd.Timestamp(now)


        #     # Get all orders and standardize them
        #     df = op.df_orders
        #     df = df[(df.order_start_date <= now) & (df.order_ends >= now)]

        #     df = df[['mrr', 'market_segment', 'months']]
        #     df.loc[:, 'market_segment'] = [ezr.slugify(s) for s in df.market_segment]
        #     df['weight_and_value'] = df.months * df.mrr
        #     df['weight'] = df.mrr


        #     dfg = df.groupby(by='market_segment')[['weight_and_value', 'weight']].sum()
        #     dfg['duration_months'] = dfg.weight_and_value / dfg.weight
        #     rec = dfg.duration_months.to_dict()
        #     rec['date'] = now
        #     return rec


        # dates = pd.date_range('1/1/2021', datetime.datetime.now())

        # rec_list = []
        # for date in tqdm(dates):
        #     rec_list.append(dollar_weighted_duration(self.op, date))
        # df = pd.DataFrame(rec_list).set_index('date')
        
        df = self.df_months
        renamer = {k: f'{k} {v:0.1f}' for (k, v) in df.iloc[-1, :].items()}
        df = df.rename(columns=renamer)

        return df.hvplot(ylabel='Dollar Weighted Contract Duration (months)').options(legend_position='top', show_grid=True)


    def plot_contracted_acv(self):
    
        def mean_contract_acv(op, now):
            # Now is the date for which you want to compute metrics
            now = pd.Timestamp(now)


            # Get all orders and standardize them
            df = op.df_orders
            df = df[(df.order_start_date <= now) & (df.order_ends >= now)]

            df = df[['account_id', 'mrr', 'market_segment']]
            df.loc[:, 'market_segment'] = [ezr.slugify(s) for s in df.market_segment]
            df = df.groupby(by=['account_id', 'market_segment'])[['mrr']].sum().reset_index()

            rec = (12 * df.groupby(by='market_segment').mrr.mean() / 1000).to_dict()
            rec['date'] = now
            return rec



        dates = pd.date_range('1/1/2021', datetime.datetime.now())

        rec_list = []
        for date in tqdm(dates):
            rec_list.append(mean_contract_acv(op, date))
        df_ = pd.DataFrame(rec_list).set_index('date')

        df = df_.copy()
        renamer = {k: f'{k} {v:0.1f}K' for (k, v) in df.iloc[-1, :].items()}
        df = df.rename(columns=renamer)

        c_abs = df.hvplot(ylabel='ACV per Deal $K').options(legend_position='top', show_grid=True, width=450)
        
        df= df_.copy()
        df = 100 * (df / df.iloc[0, :] - 1)
        renamer = {k: f'{k} {v:0.1f}%' for (k, v) in df.iloc[-1, :].items()}
        df = df.rename(columns=renamer)
        c_rel = df.hvplot(ylabel='Percent Change in ACV per Deal').options(legend_position='top', show_grid=True, width=450)
        return hv.Layout([c_abs, c_rel]).cols(2).options(shared_axes=False)  
    
    def plot(self):
        display(self.plot_ndr_time_series())
        display(self.plot_duration_time_series())
        display(self.plot_contracted_acv())






as_of = get_latest_date(get_when())
df_metrics, df_acv, df_months = get_frames(get_when())


st.title('CS Trends')
st.markdown(f'### as of {as_of}')
# st.write(df_metrics)
# st.write(df_acv)
# st.write(df_months)
plotter = CSPlotter(df_metrics, df_acv, df_months)
# plotter.plot_ndr_time_series()


c_ndr, c_ex, c_ren, c_red, c_churn = plotter.get_retention_plots()

st.markdown('---')
st.markdown(f'## Net Dollar Retention')
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
st.markdown(f'## Expansion')
with st.expander("See explanation"):
     st.markdown("""
         This is ...
     """)
display(c_ex)

st.markdown('---')
st.markdown(f'## Gross Revenue Churn')
with st.expander("See explanation"):
     st.markdown("""
         This is ...
     """)
display(c_churn)


st.markdown('---')
st.markdown(f'## Revenue Fraction lost to Churned Accounts')
with st.expander("See explanation"):
     st.markdown("""
         This is the percentage of our revenue that we lost to actually churned customers.
         These numbers ignore reduced contract values.
     """)
display(c_churn)

st.markdown('---')
st.markdown(f'## Revenue Fraction lost to Reduced Accounts')
with st.expander("See explanation"):
     st.markdown("""
         This is the percentage of our revenue that we lost to customers downsizing.
     """)
display(c_red)


