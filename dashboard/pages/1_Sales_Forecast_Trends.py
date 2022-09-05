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
    dfw = dd.get_latest('sales_won_timeseries')
    dff = dd.get_latest('sales_forecast_timeseries')
    dfh = dd.get_latest('sales_prediction_history')
    
        

    def clean_it(df):
        df = df.rename(columns={'index': 'date'})
        df.loc[:, 'date'] = df.loc[:, 'date'].astype(np.datetime64)
        df = df.set_index('date')
        return df

    dfw = clean_it(dfw)
    dff = clean_it(dff)
    dfh = clean_it(dfh)

    # dfw = dfw.rename(columnns={'index': 'date'})
    # dff = dff.rename(columnns={'index': 'date'})
    # dfh = dfh.rename(columnns={'index': 'date'})

    # dfw.loc[:, 'date'] = dfw.loc[:, 'date'].astype(np.datetime64)
    # # st.write(dff.head())
    # # st.write(dfh.head())

    # # dfw = dfw.set_index('index')
    # st.write(dfw.head())
    # dff = dff.set_index('date')
    # dfh = dfh.set_index('index')
    return dfw, dff, dfh


def plot_prediction(dfw, dff, dfh):
    pp = PredictorPlotter(dfw, dff, dfh)
    c = pp.plot()

    display(c)


@st.cache
def convert_dataframe(df):
    return df.to_csv().encode('utf-8')







st.title('Sales Forecast Trend')
@st.cache
def get_latest_date(when):
    return DashData().get_latest_time()

as_of = get_latest_date(get_when())
as_of = as_of.strftime("%B %d, %Y")
st.markdown(f'### as of {as_of}')


# dfh = get_prediction_history(get_when())
# st.write(dfh)


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


when = fleming.floor(datetime.datetime.now(), hour=1)
dfw, dff, dfh = get_plotting_frames(when)

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

plot_prediction(dfw, dff, dfh)

with st.expander("See Table"):
    st.download_button(
        label='Download CSV',
        data=convert_dataframe(dfjd),
        file_name='sales_forecast.csv',
        mime='text/csv',
    )
    st.table(dfj)

























# import streamlit as st
# import gtmarket as gtm
# from dateutil.relativedelta import relativedelta
# import fleming
# import pandas as pd
# import datetime
# import itertools
# import holoviews as hv
# from holoviews import opts
# import copy
# hv.extension('bokeh')

# # opts.defaults(opts.Curve(width=800, height=400, tools=['hover']))
# # opts.defaults(opts.Overlay(width=800, height=400, tools=['hover']))

# # import itertools
# # import pandas as pd
# # import numpy as np
# # import locale
# # locale.setlocale(locale.LC_ALL, 'en_US')
# import easier as ezr
# # import fleming 
# # from simbiz import api_live as smb
# # import datetime
# # from holoviews import opts

# class PredictorPlotter:
#     def __init__(self, pipe_stats_obj=None, include_sales_expansion=True):
#         if pipe_stats_obj is None:
#             pipe_stats_obj = gtm.PipeStats()
#         self.ps = pipe_stats_obj
#         self.include_sales_expansion = include_sales_expansion
        
#     def get_predicted_revenue(self, starting=None, ending_exclusive=None):
#         if starting is None:
#             starting = datetime.datetime.now()
            
#         starting = pd.Timestamp(starting)
#         starting = fleming.floor(starting, day=1)        
#         if ending_exclusive is None:
#             ending_exclusive = starting + relativedelta(years=1)
#         ending_exclusive = pd.Timestamp(ending_exclusive)
#         deals = gtm.Deals(starting=starting, ending_exclusive=ending_exclusive, include_sales_expansion=self.include_sales_expansion)
#         return deals.df_predicted
    
#     @ezr.cached_container
#     def _df_won(self):
#         ps = gtm.PipeStats(pilots_are_new_biz=True, sales_expansion_are_new_biz=self.include_sales_expansion)
#         df = ps.get_opp_timeseries(value='deal_acv', cumulative_since='12/31/2020')
#         return df
        
#     def get_won_revenue(self, starting=None, ending_inclusive=None):
#         today = fleming.floor(datetime.datetime.now(), day=1)
#         if ending_inclusive is None:
#             ending_inclusive = today
#         if starting is None:
#             starting = fleming.floor(today, year=1)
#         starting = pd.Timestamp(starting)
#         ending_inclusive = pd.Timestamp(ending_inclusive)
#         if starting < pd.Timestamp('1/1/2020'):
#             raise ValueError('Can only get revenue since 1/1/2020')
        
#         df = self._df_won
#         df = df.loc[starting - relativedelta(days=1):ending_inclusive, :].sort_index()
#         df = df - df.iloc[0, :]
#         df = df.loc[starting:ending_inclusive, :]
#         ind = pd.date_range(starting, ending_inclusive)
#         df = df.reindex(index=ind)
#         df = df.fillna(method='ffill')
#         return df
    
#     def get_forecast(self, since=None, today=None, ending_exclusive=None, separate_past_future=False):
#         if today is None:
#             today = fleming.floor(datetime.datetime.now(), day=1)
#         if since is None:
#             since = fleming.floor(today, year=1)
#         if ending_exclusive is None:
#             ending_exclusive = today + relativedelta(years=1)
            
#         since, today, ending_exclusive = map(pd.Timestamp, [since, today, ending_exclusive])
#         tomorrow = today + relativedelta(days=1)
            
        
#         dfw = self.get_won_revenue(starting=since, ending_inclusive=today)
#         dff = self.get_predicted_revenue(starting=tomorrow, ending_exclusive=ending_exclusive)
#         dff = dff + dfw.iloc[-1, :]
        
#         dfw = dfw.loc[since:ending_exclusive, :]
#         dff = dff.loc[since:ending_exclusive, :]
        
#         if separate_past_future:
#             return dfw, dff
#         else:
#             df = pd.concat([dfw, dff], axis=0)
#             return df
        
#     def _get_plot_frames(self, since=None, today=None, ending_exclusive=None, units='m'):
#         units = units.lower()
            
#         units_lookup = {
#             'k': 1000,
#             'm': 1e6
#         }
        
#         scale = units_lookup[units]
        
#         dfw, dff = self.get_forecast(since=since, today=today, ending_exclusive=ending_exclusive, separate_past_future=True)
#         if not dff.empty:
#             dfft = dff.T
#             dfft.loc[:, dfw.index[-1]] = dfw.iloc[-1, :]
#             dff = dfft.T.sort_index()
        
#         dff = dff / scale
#         dfw = dfw /  scale
#         return dfw, dff
        
    
    
#     def plot_latest_prediction(self, since=None, today=None, ending_exclusive=None, units='m'):
#         dfw, dff = self._get_plot_frames(since=since, today=today, ending_exclusive=ending_exclusive, units=units)
        
#         c1 = plot_frame(dfw, units=units, use_label=False)
#         c2 = plot_frame(dff, units=units, alpha=.5, use_label=True, include_total=False)
#         # display(dff.sum(axis=1))
        
#         return (c1 * c2).options(legend_position='top')
    
#     def plot_prediction_history(self, since=None, ending_exclusive=None,  units='m'):
#         mph = gtm.ModelParamsHist()
#         df = mph.get_history()
#         min_time, max_time = [fleming.floor(d, day=1) for d in [df.time.min(), df.time.max()]]
        
#         dates = pd.date_range(min_time, max_time)
#         predictions = []
#         for today in dates:
#             dfw, dff = self._get_plot_frames(since=since, today=today, ending_exclusive=ending_exclusive, units=units)
#             predictions.append(dff.iloc[-1, :].sum())
            
#         dfp = pd.DataFrame({'acv': predictions}, index=dates)
#         ind = pd.date_range(dfp.index[0], ending_exclusive, inclusive='left')
#         dfp = dfp.reindex(ind).fillna(method='ffill')
#         # display(dfp)
#         final_val = dfp.acv.iloc[-1]
#         c = hv.Curve((dfp.index, dfp.acv), label=f'Pacing to {final_val:0.1f}{units.upper()}').options(color='gray')
#         return c
    
#     def plot(self, since=None, today=None, ending_exclusive=None, units='m'):
#         ol = self.plot_latest_prediction(since=since, today=today, ending_exclusive=ending_exclusive, units=units)
#         return ol
#         c = self.plot_prediction_history(since=since, ending_exclusive=ending_exclusive, units=units)
#         return hv.Overlay([ol, c]).options(legend_position='top')
                                         

# def plot_frame(df, alpha=1, use_label=True, units='', include_total=True, ylabel='ACV'):  # pragma: no cover
#     import holoviews as hv
#     colorcycle = itertools.cycle(ezr.cc.codes)
#     c_list = []
#     base = 0 * df.iloc[:, -1]
#     for col in df.columns:
#         if use_label:
#             final = df[col].iloc[-1]
#             label = col
#             label = label.split('_')[0].title()
#             label = f'{label} {final:0.1f}{units.upper()}'
#             if include_total and (col == df.columns[-1]):
#                 label = label + f'  Total={df.iloc[-1].sum():0.1f}{units.upper()}'
#         else:
#             label = ''
#         y = df[col]
#         c = hv.Area(
#             (df.index, y + base, base),
#             kdims=['Date'],
#             vdims=[f'{ylabel} ${units.upper()}', 'base'],
#             label=label
#         ).options(alpha=alpha, color=next(colorcycle), show_grid=True)
#         c_list.append(c)
#         c_list.append(hv.Curve((df.index, y + base)).options(color='black', alpha=.01))
#         base += y
#     return hv.Overlay(c_list).options(legend_position='top')


# def display(hv_obj):
#     st.write(hv.render(hv_obj, backend='bokeh'))


# # @st.cache(allow_output_mutation=True)
# def get_predict_plot(when):
#     since='1/1/2022'
#     ending_exclusive = '1/1/2023'
#     ps = gtm.PipeStats()
#     pp = PredictorPlotter(ps, include_sales_expansion=True)
#     return pp.plot(since=since, ending_exclusive=ending_exclusive, units='m')



# st.title('Sales Funnel Trends')

# st.markdown('---')
# st.markdown('## Sales Forecast Trend')
# when = fleming.floor(datetime.datetime.now(), hour=1)
# display(copy.deepcopy(get_predict_plot(when)))

# st.markdown('---')
# st.markdown('## SAL Creation Rate')

# st.markdown('---')
# st.markdown('## SAL to SQL Conversion')

# st.markdown('---')
# st.markdown('## SQL to Won Conversion')

# st.markdown('---')
# st.markdown('## SAL to Won Conversion')

# st.markdown('---')
# st.markdown('## Days to Win')

# st.markdown('---')
# st.markdown('## Average Closed-Won ACV')

# # @st.cache
# # def get_frame():
# #     from simbiz import api_live as smb
# #     op = smb.OppLoader()
# #     op.enable_pickle_cache()
# #     return op.df_new_biz



# # def to_dollars(ser):
# #     return [locale.currency(x, grouping=True).split('.')[0] for x in ser]

# # def to_percent(ser):
# #     return ['-' if x == '-' else f'{x}%' for x in ser]

# # def print_blob():
# #     bp = BlobPrinter()
# #     ps = smb.PipeStats()
# #     ps.enable_pickle_cache()
# #     bp.display_win_rates(ps)



# # class BlobPrinter():
    
# #     @ezr.cached_container
# #     def _blob(self):
        
# #         pmh = smb.ModelParamsHist()
# #         pm = pmh.get_latest()
# #         blob = pm.to_blob()
# #         return blob
    
# #     @ezr.cached_container
# #     def blob(self):
# #         blob = self._blob
# #         for key in [
# #             'existing_pipe_model_with_expansion',            
# #             'existing_pipe_model']:
# #             blob.pop(key)
# #         return blob
    
# #     def display_win_rates(self, pipe_stats_obj):
# #         ps = pipe_stats_obj
# #         ser_sal2sql = (100 * ps.get_conversion_timeseries('sal2sql_opps', interval_days=90, bake_days=30)).round(1).iloc[-1]
# #         ser_sql2win = (100 * ps.get_conversion_timeseries('sql2won_opps', interval_days=365, bake_days=90)).round(1).iloc[-1]
# #         ser_sal2win = (100 * ps.get_conversion_timeseries('sal2won_opps', interval_days=365, bake_days=90)).round(1).iloc[-1]
        
# #         ser = ps.get_opp_timeseries('num_sals', interval_days=30).iloc[-1, :]
# #         ser['total'] = ser.sum()
# #         dfs = pd.DataFrame({'SALS / month': ser}).round().astype(int)
        
        
# #         dfd = (pd.DataFrame({'ACV': ps.get_mean_deal_size_timeseries().iloc[-1, :]})).round(1)
# #         dfd = dfd.loc[['enterprise', 'commercial', 'velocity'], :]
        
        
# #         dfwr = pd.DataFrame({
# #             'SAL⮕SQL': ser_sal2sql,
# #             'SQL⮕WON': ser_sql2win,
# #             'SAL⮕WON': ser_sal2win,
# #         })
        
# #         ser = (100 * ps.get_stage_win_rates_timeseries(interval_days=365, bake_days=90).iloc[-1, :]).round(1)
# #         ser = ser[['SAL', 'Discovery', 'Demo', 'Proposal', 'Negotiation']]
# #         dfswr = pd.DataFrame({'Win rate by stage': ser})        
        
# #         today = fleming.floor(datetime.datetime.now(), day=1)
# #         dfo = ps.op.df_orders
# #         dfo.loc[:, 'market_segment'] = [ezr.slugify(m) for m in dfo.market_segment]

# #         dfo = dfo[(dfo.order_start_date <= today) & (dfo.order_ends > today)]
# #         ser = (12 * dfo.groupby(by='market_segment')[['mrr']].sum()).round(2).mrr
# #         ser = ser[['commercial', 'enterprise', 'velocity']]
# #         ser['combined'] = ser.sum()
# #         # dfr = pd.DataFrame({'Current ARR ($M)': ser})
# #         dfr = pd.DataFrame({'Current ARR': ser})
        
# #         dfn = pipe_stats_obj.op.get_ndr_metrics()
# #         dfn = dfn[dfn.variable == 'ndr'].set_index('market_segment')[['value']].round(1).rename(columns={'value': '12-month NDR'})
        
# #         dft = ps.get_conversion_timeseries('sal2won_time', interval_days=365, bake_days=90).iloc[[-1], :].round().astype(int).T
# #         dft.columns = ['Days to Win']
# #         dft.index.name = None      
        
# #         df_sales = dfs
# #         df_sales = df_sales.join(dfwr).drop('total')
# #         df_sales = df_sales.join(dft)
# #         df_sales = df_sales.join(dfd)
# #         sal_val = (.01 * df_sales['SAL⮕WON'] * df_sales['ACV']).round().astype(int)
# #         sql_val = (.01 * df_sales['SQL⮕WON'] * df_sales['ACV']).round().astype(int)
# #         value_rate = sal_val * df_sales['SALS / month']
        
        
# #         df_sales['SAL Val'] = to_dollars(sal_val)
# #         df_sales['SQL Val'] = to_dollars(sql_val) #[locale.currency(x, grouping=True).split('.')[0] for x in sql_val]
# #         df_sales.loc[:, 'ACV'] = to_dollars(df_sales.ACV) #[locale.currency(x, grouping=True).split('.')[0] for x in df_sales.ACV]
# #         df_sales.loc[:, 'SAL Value / Month'] = to_dollars(value_rate) #[locale.currency(x, grouping=True).split('.')[0] for x in value_rate]
        
# #         for col in dfwr.columns:
# #             df_sales.loc[:, col] = to_percent(df_sales.loc[:, col])
        
        
# #         df_arr = dfr
# #         df_arr = df_arr.join(dfn)
# #         df_arr = df_arr.fillna('-')
# #         monthly_rate = (.01 * df_arr['12-month NDR']) ** (1 / 12) - 1
# #         # display(df_arr)
# #         df_arr['Value / Month'] = df_arr['Current ARR'] * monthly_rate
        
# #         df_arr.loc[:, 'Current ARR'] = to_dollars(df_arr['Current ARR'])
# #         df_arr.loc[:, '12-month NDR'] = to_percent(df_arr.loc[:, '12-month NDR'])
# #         df_arr.loc[:, 'Value / Month'] = to_dollars(df_arr['Value / Month'])
# #         df_arr.index.name = None
# #         st.dataframe(df_sales, width=1800)
# #         st.dataframe(df_arr, width=800)


# # def plot_frame(df, alpha=1, use_label=True, units='', include_total=True, ylabel='ACV'):  # pragma: no cover
# #     import holoviews as hv
# #     colorcycle = itertools.cycle(ezr.cc.codes)
# #     c_list = []
# #     base = 0 * df.iloc[:, -1]
# #     for col in df.columns:
# #         if use_label:
# #             final = df[col].iloc[-1]
# #             label = col
# #             label = label.split('_')[0].title()
# #             label = f'{label} {final:0.1f}{units.upper()}'
# #             if include_total and (col == df.columns[-1]):
# #                 label = label + f'  Total={df.iloc[-1].sum():0.1f}{units.upper()}'
# #         else:
# #             label = ''
# #         y = df[col]
# #         c = hv.Area(
# #             (df.index, y + base, base),
# #             kdims=['Date'],
# #             vdims=[f'{ylabel} ${units.upper()}', 'base'],
# #             label=label
# #         ).options(alpha=alpha, color=next(colorcycle), show_grid=True)
# #         c_list.append(c)
# #         c_list.append(hv.Curve((df.index, y + base)).options(color='black', alpha=.01))
# #         base += y
# #     return hv.Overlay(c_list).options(legend_position='top')


# # class PredictorPlotter:
# #     def __init__(self, pipe_stats_obj=None, include_sales_expansion=True):
# #         if pipe_stats_obj is None:
# #             pipe_stats_obj = smb.PipeStats()
# #         self.ps = pipe_stats_obj
# #         self.include_sales_expansion = include_sales_expansion
        
# #     def get_predicted_revenue(self, starting=None, ending_exclusive=None):
# #         if starting is None:
# #             starting = datetime.datetime.now()
            
# #         starting = pd.Timestamp(starting)
# #         starting = fleming.floor(starting, day=1)        
# #         if ending_exclusive is None:
# #             ending_exclusive = starting + relativedelta(years=1)
# #         ending_exclusive = pd.Timestamp(ending_exclusive)
# #         deals = smb.Deals(starting=starting, ending_exclusive=ending_exclusive, include_sales_expansion=self.include_sales_expansion)
# #         return deals.df_predicted
    
# #     @ezr.cached_container
# #     def _df_won(self):
# #         ps = smb.PipeStats(pilots_are_new_biz=True, sales_expansion_are_new_biz=self.include_sales_expansion)
# #         df = ps.get_opp_timeseries(value='deal_acv', cumulative_since='12/31/2020')
# #         return df
        
# #     def get_won_revenue(self, starting=None, ending_inclusive=None):
# #         today = fleming.floor(datetime.datetime.now(), day=1)
# #         if ending_inclusive is None:
# #             ending_inclusive = today
# #         if starting is None:
# #             starting = fleming.floor(today, year=1)
# #         starting = pd.Timestamp(starting)
# #         ending_inclusive = pd.Timestamp(ending_inclusive)
# #         if starting < pd.Timestamp('1/1/2020'):
# #             raise ValueError('Can only get revenue since 1/1/2020')
        
# #         df = self._df_won
# #         df = df.loc[starting - relativedelta(days=1):ending_inclusive, :].sort_index()
# #         df = df - df.iloc[0, :]
# #         df = df.loc[starting:ending_inclusive, :]
# #         ind = pd.date_range(starting, ending_inclusive)
# #         df = df.reindex(index=ind)
# #         df = df.fillna(method='ffill')
# #         return df
    
# #     def get_forecast(self, since=None, today=None, ending_exclusive=None, separate_past_future=False):
# #         if today is None:
# #             today = fleming.floor(datetime.datetime.now(), day=1)
# #         if since is None:
# #             since = fleming.floor(today, year=1)
# #         if ending_exclusive is None:
# #             ending_exclusive = today + relativedelta(years=1)
            
# #         since, today, ending_exclusive = map(pd.Timestamp, [since, today, ending_exclusive])
# #         tomorrow = today + relativedelta(days=1)
            
        
# #         dfw = self.get_won_revenue(starting=since, ending_inclusive=today)
# #         dff = self.get_predicted_revenue(starting=tomorrow, ending_exclusive=ending_exclusive)
# #         dff = dff + dfw.iloc[-1, :]
        
# #         dfw = dfw.loc[since:ending_exclusive, :]
# #         dff = dff.loc[since:ending_exclusive, :]
        
# #         if separate_past_future:
# #             return dfw, dff
# #         else:
# #             df = pd.concat([dfw, dff], axis=0)
# #             return df
        
# #     def _get_plot_frames(self, since=None, today=None, ending_exclusive=None, units='m'):
# #         units = units.lower()
            
# #         units_lookup = {
# #             'k': 1000,
# #             'm': 1e6
# #         }
        
# #         scale = units_lookup[units]
        
# #         dfw, dff = self.get_forecast(since=since, today=today, ending_exclusive=ending_exclusive, separate_past_future=True)
# #         if not dff.empty:
# #             dfft = dff.T
# #             dfft.loc[:, dfw.index[-1]] = dfw.iloc[-1, :]
# #             dff = dfft.T.sort_index()
        
# #         dff = dff / scale
# #         dfw = dfw /  scale
# #         return dfw, dff
        
    
    
# #     def plot_latest_prediction(self, since=None, today=None, ending_exclusive=None, units='m'):
# #         dfw, dff = self._get_plot_frames(since=since, today=today, ending_exclusive=ending_exclusive, units=units)
        
# #         c1 = plot_frame(dfw, units=units, use_label=False)
# #         c2 = plot_frame(dff, units=units, alpha=.5, use_label=True, include_total=False)
# #         # display(dff.sum(axis=1))
        
# #         return (c1 * c2).options(legend_position='top')
    
# #     def plot_prediction_history(self, since=None, ending_exclusive=None,  units='m'):
# #         mph = smb.ModelParamsHist()
# #         df = mph.get_history()
# #         min_time, max_time = [fleming.floor(d, day=1) for d in [df.time.min(), df.time.max()]]
        
# #         dates = pd.date_range(min_time, max_time)
# #         predictions = []
# #         for today in dates:
# #             dfw, dff = self._get_plot_frames(since=since, today=today, ending_exclusive=ending_exclusive, units=units)
# #             predictions.append(dff.iloc[-1, :].sum())
            
# #         dfp = pd.DataFrame({'acv': predictions}, index=dates)
# #         ind = pd.date_range(dfp.index[0], ending_exclusive, inclusive='left')
# #         dfp = dfp.reindex(ind).fillna(method='ffill')
# #         # display(dfp)
# #         final_val = dfp.acv.iloc[-1]
# #         c = hv.Curve((dfp.index, dfp.acv), label=f'Pacing to {final_val:0.1f}{units.upper()}').options(color='gray')
# #         return c
    
# #     def plot(self, since=None, today=None, ending_exclusive=None, units='m'):
# #         ol = self.plot_latest_prediction(since=since, today=today, ending_exclusive=ending_exclusive, units=units)
# #         c = self.plot_prediction_history(since=since, ending_exclusive=ending_exclusive, units=units)
# #         return hv.Overlay([ol, c]).options(legend_position='top')
                                         

# # def display(hv_obj):
# #     st.write(hv.render(hv_obj, backend='bokeh'))


# # def plot_predict(ending_exclusive):        
# #     since='1/1/2022'
# #     ps = smb.PipeStats()
# #     ps.enable_pickle_cache()
# #     pp = PredictorPlotter(ps, include_sales_expansion=True)
# #     display(pp.plot(since=since, ending_exclusive=ending_exclusive, units='m'))




# # opts.defaults(opts.Curve(width=800, height=400, tools=['hover']))
        
        

# # print_blob()


# # ending_exclusive = st.text_input('The prediction end date', value='1/1/2023')
# # plot_predict(ending_exclusive)
