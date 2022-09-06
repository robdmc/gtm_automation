import streamlit as st
import gtmarket as gtm
from dateutil.relativedelta import relativedelta
import fleming
import pandas as pd
import numpy as np
import datetime
import itertools
import holoviews as hv
from holoviews import opts
import copy
from dash_lib import DashData, PredictorGetter
import easier as ezr
import locale
from dash_lib import (
    display,
    plot_frame,
    get_when,
    to_dollars,
    DashData,
    SALGetter,
    convert_dataframe
)
hv.extension('bokeh')


opts.defaults(opts.Area(width=800, height=400), tools=[])
opts.defaults(opts.Curve(width=800, height=400, tools=['hover']))


class PredictorPlotter:
    def __init__(self, dfw, dff, dfh, units='m'):
        units_lookup = {
            'k': 1000,
            'm': 1e6,
            'u': 1,
        }
        self.units = units.lower()
        scale = units_lookup[self.units]

        self.dfw = dfw / scale
        self.dff = dff / scale
        self.dfh = dfh / scale

        self.dff = self.dff + self.dfw.iloc[-1]
        
    
    def plot_latest_prediction(self):
        
        c1 = plot_frame(self.dfw, units=self.units, use_label=False)
        c2 = plot_frame(self.dff, units=self.units, alpha=.5, use_label=True, include_total=False)
        # display(dff.sum(axis=1))
        
        return (c1 * c2).options(legend_position='top')
    
    def plot_prediction_history(self):
        final_val = self.dfh.acv.iloc[-1]
        c = hv.Curve((self.dfh.index, self.dfh.acv), label=f'Pacing to {final_val:0.1f}{self.units.upper()}').options(color='gray')
        return c
    
    def plot(self):
        ol = self.plot_latest_prediction()
        c = self.plot_prediction_history()
        return hv.Overlay([ol, c]).options(legend_position='top')




@st.cache
def get_plotting_frames(when):
    dd = DashData()
    dfw = dd.get_latest('dash_sales_won_timeseries')
    dff = dd.get_latest('dash_sales_forecast_timeseries')
    dfh = dd.get_latest('dash_sales_prediction_history')
    
        

    def clean_it(df):
        df = df.rename(columns={'index': 'date'})
        df.loc[:, 'date'] = df.loc[:, 'date'].astype(np.datetime64)
        df = df.set_index('date')
        return df

    dfw = clean_it(dfw)
    dff = clean_it(dff)
    dfh = clean_it(dfh)

    return dfw, dff, dfh


@st.cache
def get_sal_frame(when):
    dd = DashData()
    df = dd.get_latest('dash_sal_creation_rate')
    df.loc[:, 'date'] = df.date.astype(np.datetime64)
    df = df.set_index('date')
    return df


@st.cache
def get_conversion_frames(when):
    dd = DashData()
    df_sal2sql = dd.get_latest('dash_sal2sql')
    df_sql2won = dd.get_latest('dash_sql2won')
    df_sal2won = dd.get_latest('dash_sal2won')


    def clean_it(df):
        df.loc[:, 'date'] = df.loc[:, 'date'].astype(np.datetime64)
        df = df.set_index('date')
        return df

    df_sal2sql = clean_it(df_sal2sql)
    df_sql2won = clean_it(df_sql2won)
    df_sal2won = clean_it(df_sal2won)

    return df_sal2sql, df_sql2won, df_sal2won



def plot_prediction(dfw, dff, dfh):
    pp = PredictorPlotter(dfw, dff, dfh)
    c = pp.plot()

    display(c)


def plot_rolling_sal_creation(df, rolling_days=30):
    cols = sorted(c for c in df.columns if 'fit' not in c)
    df = df.reset_index()
    colors = ezr.cc.codes[:len(cols)]

    c_list = []
    for col, color in zip(cols, colors):
        fit_col = col + '_fit'
        x = df['date']
        y = df[col]
        yf = df[fit_col]
        c = hv.Curve((x, yf), 'Date', f'Rolling {rolling_days}-day SALS', label=col).options(show_grid=True)
        c_list.append(c)

        c = hv.Scatter((x, y), label='')#.options(size=5)
        c_list.append(c)
        
    ol = hv.Overlay(c_list).options(legend_position='top')
    return ol    



def plot_conversion(dfc, ylabel):
    dfc = dfc.loc['1/1/2021':, :]
    
    
    c_list = []
    latest_vals = dfc.iloc[-1]
    colorcycle = iter(list(ezr.cc.codes))
    for col, val in latest_vals.items():
        color = next(colorcycle)
        
        actual = np.round(dfc[col].iloc[-1], 1)
        c = hv.Curve((dfc.index, dfc[col]), 'Date', ylabel, label=f'{col.title()} {actual}%').options(color=color, show_grid=True)
        c_list.append(c)
        
    return hv.Overlay(c_list).options(legend_position='top')


@st.cache
def get_latest_date(when):
    return DashData().get_latest_time()

as_of = get_latest_date(get_when())
as_of = as_of.strftime("%B %d, %Y")



dfw, dff, dfh = get_plotting_frames(get_when())

dfj = pd.concat([dfw, dff]).sort_index()
dfj['total'] = dfj.sum(axis=1)
dfj.index.name = 'date'

dfj = dfj.round().astype(int)
dfjd = dfj.copy().reset_index()
for col in dfj.columns:
    dfj.loc[:, col] = to_dollars(dfj.loc[:, col])
dfj.index = [str(t.date()) for t in dfj.index]
dfj.index.name = 'date'
dfj = dfj.reset_index()

df_sal = get_sal_frame(get_when())
df_sal2sql, df_sql2won, df_sal2won = get_conversion_frames(get_when())




st.title('Sales Trends')
st.markdown(f'### as of {as_of}')
st.markdown('---')
st.markdown(f'## 12/31/22 Sales Forecast')

with st.expander("See explanation"):
     st.markdown("""
         The shaded areas represent the amount of revenue for each segment.  The dark colored
         areas are revenue that has already been won, whereas the light shaded areas represent
         forcasted revenue.  The grey line represents a history of what the forecast has been for
         each dat in the past.  This forecast is generated as follows:

         * Take all "New Business" opportunities without an ACV specified and impute the average ACV of deals
           signed in their market segment.
         * Multiply the ACV of each opportunity by the win-rate corresponding to the stage the
           opportunity is in. Do this without regard to market segment.
         * The expected revenue for each opportunity is just its ACV discounted by its win-rate.
           Assume that we win this expected value at the close-date of each opportunity.
         * Start a forecast by just looking at the expected revenue from existing opportunities
         * We know the average time-to-win for each market segment.  We also know the win-rate for
           each market segment.  We combine this knowledge with the rate at which we are generating
           SALS.  We can put all this together with the average ACV by market-segment to arrive at
           revenue expected to be won by opportunities not yet created.
         * Add the revenue from these not-yet-created opportunities to the forecast
         * Take all open "Sales Expansion" opportunities, multiply the Sales Expansion win rate.
         * Add "Sales Expansion" opportunities to the forecast.
         * Assume that each year, every company under contract expands their contract with us by 10% 
           (this is a very rough estimate as of Aug. 2022).  Prorate this expansion to determine a daily
           expansion rate and add it to the forecast.


        

     """)



plot_prediction(dfw, dff, dfh)

with st.expander("See Table"):
    st.download_button(
        label='Download CSV',
        data=convert_dataframe(dfjd),
        file_name='sales_forecast.csv',
        mime='text/csv',
    )
    st.table(dfj)




st.markdown('---')
st.markdown(f'## 30-day Rolling SAL Creation Rate')

display(plot_rolling_sal_creation(df_sal))
df_sal_pretty = df_sal.copy()
df_sal_pretty.index = pd.Index([str(d.date()) for d in df_sal_pretty.index], name='date')
df_sal_pretty = df_sal_pretty[[c for c in df_sal_pretty.columns if not '_fit' in c]]
df_sal_pretty = df_sal_pretty.astype(int)

with st.expander("See Table"):
    st.download_button(
        label='Download CSV',
        data=convert_dataframe(df_sal_pretty),
        file_name='sal_creation_30day_rolling.csv',
        mime='text/csv',
    )
    st.table(df_sal_pretty)


st.markdown('---')
st.markdown(f'## Pipeline Conversion Rates')

st.markdown(f'### SAL to SQL')
display(plot_conversion(df_sal2sql, 'SAL to SQL'))
with st.expander("See Table"):
    st.download_button(
        label='Download CSV',
        data=convert_dataframe(df_sal2sql.reset_index()),
        file_name='sal2sql.csv',
        mime='text/csv',
    )
    st.table(df_sal2sql)


st.markdown(f'### SQL to Won')
display(plot_conversion(df_sql2won, 'SQL to Won'))
with st.expander("See Table"):
    st.download_button(
        label='Download CSV',
        data=convert_dataframe(df_sql2won.reset_index()),
        file_name='sql2won.csv',
        mime='text/csv',
    )
    st.table(df_sql2won)

st.markdown(f'### SAL to Won')
display(plot_conversion(df_sal2won, 'SAL to Won'))
with st.expander("See Table"):
    st.download_button(
        label='Download CSV',
        data=convert_dataframe(df_sal2won.reset_index()),
        file_name='sal2won.csv',
        mime='text/csv',
    )
    st.table(df_sal2won)


