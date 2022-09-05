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

import dash_lib as dl
from dash_lib import (
    convert_dataframe,
    float_to_dollars,
    to_dollars,
    display,
    plot_frame,
)



@st.cache
def get_arr_timeseries():
    df = dl.DashData().get_latest('arr_time_series')
    df.loc[:, 'date'] = df.date.astype(np.datetime64)
    df = df.set_index('date')
    return df


@st.cache
def get_sales_progress_frames():
    df = dl.DashData().get_latest('sales_progress')
    df = df.set_index('index')
    df.index.name = 'Market Segment'
    df_pretty = df.copy()
    for col in df_pretty.columns:
        df_pretty.loc[:, col] = to_dollars(df_pretty.loc[:, col])
    return df, df_pretty



@st.cache
def get_process_stats_frames():
    dd = dl.DashData()
    df_sales = dd.get_latest('sales_stats')
    df_stage_wr = dd.get_latest('sales_stats_stage_win_rate')
    df_arr = dd.get_latest('sales_stats_arr')

    df_sales = df_sales.set_index('index')
    df_stage_wr = df_stage_wr.set_index('index')
    df_arr = df_arr.set_index('index')
    return df_sales, df_stage_wr, df_arr




def plot_arr_timeseries():
    today = fleming.floor(datetime.datetime.now())
    ending_exclusive = pd.Timestamp('1/1/2023')
    ending_inclusive = ending_exclusive - relativedelta(days=1)

    df = get_arr_timeseries() / 1e6

    df_past = df.loc[:today]
    df_future = df.loc[today:]

    c1 = plot_frame(df_past, alpha=1, use_label=False, units='m', include_total=False, ylabel='ARR')
    c2 = plot_frame(df_future, alpha=.5, use_label=True, units='m', include_total=True, ylabel='ARR')
    # c3 = hv.Curve((dfh.index, dfh.arr)).options(color='black')
    return hv.Overlay([c1, c2]).options(legend_position='top')
    # return hv.Overlay([c1, c2, c3]).options(legend_position='top')


    
dfprog_download, dfprog =  get_sales_progress_frames()        

# # The "when" argument dictates when the cache automatically resets
df_sales, df_stage_wr, df_arr = get_process_stats_frames()
dfarr = get_arr_timeseries()


expected_arr = dfarr.iloc[-1].sum().round()
expected_arr = float_to_dollars(expected_arr) 

st.title('GTM Summary Statistics')

st.markdown('---')
st.markdown(f'## 12/31/22 ARR forecast: {expected_arr}')

display(plot_arr_timeseries())

with st.expander("See Table"):
    dfarr_disp = dfarr.copy()
    dfarr_disp['total'] = dfarr_disp.sum(axis=1)
    dfarr_disp.index = [str(d.date()) for d in dfarr_disp.index]
    for col in dfarr_disp.columns:
        dfarr_disp.loc[:, col] = to_dollars(dfarr_disp.loc[:, col])

    st.download_button(
        label='Download CSV',
        data=convert_dataframe(dfarr),
        file_name='arr_forecast_timeseries.csv',
        mime='text/csv',
    )

    st.table(dfarr_disp)

st.markdown('---')
st.markdown('## YTD Sales Progress')
with st.expander("See explanation"):
     st.write("""
         This is how progress on our sales numbers breaks down for the year.  For each market segment
         you can see how much we've won YTD, how much we have forecasted to win for the remainder of the
         year, and what that totals to.  Finally, you have the total revenue won and remaining (forecast).
     """)
st.table(dfprog)

st.download_button(
    label='Download CSV',
    data=convert_dataframe(dfprog_download),
    file_name='sales_progress.csv',
    mime='text/csv',
)

st.markdown('---')
st.markdown('### SalesProcess Metrics')
with st.expander("See explanation"):
     st.write("""
        The statistics in this table represent the most recent estimates of their corresponding metrics.
        The end goal is to assign a dollar value the total SDR activity in any given month.  Below are
        details about how the different columns are computed
        * **SALS / month:** The number of SALS created in the last 30 days
        * **SAL⮕SQL:** The rolling-90 day average of SAL to SQL conversion as of 30 days ago. (It takes 30 days to "bake".)
        * **SQL⮕Won:** The rolling-365 day average of SQL to Closed-Won as of 90 days ago. (It takes 90 days to "bake".)
        * **SAL⮕Won:** The rolling-365 day average of SAL to Closed-Won as of 90 days ago. (It takes 90 days to "bake".)
        This is the end-to-end conversion rate of our sales funnel.
        * **Days to Win:** This is the average number of days it takes for an opportunity to move from SAL to Closed-Won.
        This statistic also comes with a baking time.  We look at all opps closed in the 365 days ending 90 days ago.
        We then average the number of days those opps stayed open.
        * **ACV:** This is the average ACV computed over all currently active customers.
        * **SAL Val:** This is the value of a single SAL.  It is computed by multiplying the average deal size by the
                    expected SAL win rate.
        * **SQL Val:** This is the value of a single SAL.  It is computed by multiplying the average deal size by the
                    expected SQL win rate.
        * **SAL Value / Month:** This is the average value being generated by the SDR team per month.  It is just the 
                                number of SALS we have generated over the past thirty days multiplied by their average ACV.
     """)
st.table(df_sales)
st.download_button(
    label='Download CSV',
    data=convert_dataframe(df_sales),
    file_name='sales_summary.csv',
    mime='text/csv',
)

st.markdown('### By Stage')
with st.expander("See explanation"):
     st.write("""
        These are estimates of our win rate by stage.  Baking time is an imporant consideration for these.
        For each market segment the win rate was computed by looking at all opps closed (won or lost) in the
        365 days ending 90 days ago.  We search for all closed opps that passed through a given stage.  Once
        identified, we compute the fraction of these that were won, and use that as our estimate.
     """)
st.table(df_stage_wr)
st.download_button(
    label='Download CSV',
    data=convert_dataframe(df_stage_wr),
    file_name='stage_win_rates.csv',
    mime='text/csv',
)


    

st.markdown('---')
st.markdown('## CS Summary')

with st.expander("See explanation"):
     st.markdown("""
        This table presents statistics about our current customer base. 
        Below are descriptions of each column. For each customer, compute two numbers,
        `arr_then` (the amount they were paying us a year ago) and `arr_now` the amount they 
        are paying us today. When you see the `sum` function below, understand that it is the
        sum over all customers.
        * **Current ARR:** `sum(arr_now)` The total ARR currently under contract.  This sums up the MRR
                           for all orders starting today or before and ending in the future.
                           The sum is multiplied by 12 to convert to ARR.
        * **12-month Gross Retention:** `sum(min(arr_then, arr_now)) / sum(arr_then)`. This is the fraction 
            of revenue that customers continue to pay us one year later (ignoring expansion).
        * **12-month Expansion:** `sum(max(arr_now - arr_then, 0)) / sum(arr_then)`. This is the fraction
            by which customers expanded their revenue with us.  It is a number that is greater than or equal to zero,
            never negative.
        * **12-month NDR:** `sum(arr_now) / sum(arr_then)`.   We define NDR as the all-inclusive ratio of what
            customers are paying us now with respect to what they were paying us a year ago. Let's make some definitions.
        * **Value / Month** This is am estimate of the monthly change in ARR of the average customer in market segment. 
            Here is how we compute it.
            * $f_a$ is the fraction of revenue compared to a year ago.  It's just annual NDR.
            * $f_m$ is the fraction of revenue compared to a month ago.  It's just monthly NDR.
            * Mathematically, we know that $f_a = f_m^{12}$ or $f_m=f_a^\\frac{1}{12}$
            * Defining $R$ as the current revenue, the estimated change of revenue for 
              any month should be $\Delta = R(f_m - 1)= R \\left(f_a^\\frac{1}{12} - 1\\right)$
            * But, since $f_a$ is just the annual NDR, and $R$ is ARR, we come to the final formula for value per month
                * $\\textbf{Value / Month} = \\textbf{ARR}\\left(\\textbf{NDR}^\\frac{1}{12} - 1\\right)$

     """)


st.table(df_arr)
st.download_button(
    label='Download CSV',
    data=convert_dataframe(df_arr),
    file_name='cs_summary.csv',
    mime='text/csv',
)
