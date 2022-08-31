import streamlit as st
# import itertools
# import pandas as pd
# import numpy as np
# import locale
# locale.setlocale(locale.LC_ALL, 'en_US')
# import easier as ezr
# import fleming 
# from simbiz import api_live as smb
# import datetime
# from dateutil.relativedelta import relativedelta
# import holoviews as hv
# from holoviews import opts
# hv.extension('bokeh')




st.title('this is my title')

# @st.cache
# def get_frame():
#     from simbiz import api_live as smb
#     op = smb.OppLoader()
#     op.enable_pickle_cache()
#     return op.df_new_biz



# def to_dollars(ser):
#     return [locale.currency(x, grouping=True).split('.')[0] for x in ser]

# def to_percent(ser):
#     return ['-' if x == '-' else f'{x}%' for x in ser]

# def print_blob():
#     bp = BlobPrinter()
#     ps = smb.PipeStats()
#     ps.enable_pickle_cache()
#     bp.display_win_rates(ps)



# class BlobPrinter():
    
#     @ezr.cached_container
#     def _blob(self):
        
#         pmh = smb.ModelParamsHist()
#         pm = pmh.get_latest()
#         blob = pm.to_blob()
#         return blob
    
#     @ezr.cached_container
#     def blob(self):
#         blob = self._blob
#         for key in [
#             'existing_pipe_model_with_expansion',            
#             'existing_pipe_model']:
#             blob.pop(key)
#         return blob
    
#     def display_win_rates(self, pipe_stats_obj):
#         ps = pipe_stats_obj
#         ser_sal2sql = (100 * ps.get_conversion_timeseries('sal2sql_opps', interval_days=90, bake_days=30)).round(1).iloc[-1]
#         ser_sql2win = (100 * ps.get_conversion_timeseries('sql2won_opps', interval_days=365, bake_days=90)).round(1).iloc[-1]
#         ser_sal2win = (100 * ps.get_conversion_timeseries('sal2won_opps', interval_days=365, bake_days=90)).round(1).iloc[-1]
        
#         ser = ps.get_opp_timeseries('num_sals', interval_days=30).iloc[-1, :]
#         ser['total'] = ser.sum()
#         dfs = pd.DataFrame({'SALS / month': ser}).round().astype(int)
        
        
#         dfd = (pd.DataFrame({'ACV': ps.get_mean_deal_size_timeseries().iloc[-1, :]})).round(1)
#         dfd = dfd.loc[['enterprise', 'commercial', 'velocity'], :]
        
        
#         dfwr = pd.DataFrame({
#             'SAL⮕SQL': ser_sal2sql,
#             'SQL⮕WON': ser_sql2win,
#             'SAL⮕WON': ser_sal2win,
#         })
        
#         ser = (100 * ps.get_stage_win_rates_timeseries(interval_days=365, bake_days=90).iloc[-1, :]).round(1)
#         ser = ser[['SAL', 'Discovery', 'Demo', 'Proposal', 'Negotiation']]
#         dfswr = pd.DataFrame({'Win rate by stage': ser})        
        
#         today = fleming.floor(datetime.datetime.now(), day=1)
#         dfo = ps.op.df_orders
#         dfo.loc[:, 'market_segment'] = [ezr.slugify(m) for m in dfo.market_segment]

#         dfo = dfo[(dfo.order_start_date <= today) & (dfo.order_ends > today)]
#         ser = (12 * dfo.groupby(by='market_segment')[['mrr']].sum()).round(2).mrr
#         ser = ser[['commercial', 'enterprise', 'velocity']]
#         ser['combined'] = ser.sum()
#         # dfr = pd.DataFrame({'Current ARR ($M)': ser})
#         dfr = pd.DataFrame({'Current ARR': ser})
        
#         dfn = pipe_stats_obj.op.get_ndr_metrics()
#         dfn = dfn[dfn.variable == 'ndr'].set_index('market_segment')[['value']].round(1).rename(columns={'value': '12-month NDR'})
        
#         dft = ps.get_conversion_timeseries('sal2won_time', interval_days=365, bake_days=90).iloc[[-1], :].round().astype(int).T
#         dft.columns = ['Days to Win']
#         dft.index.name = None      
        
#         df_sales = dfs
#         df_sales = df_sales.join(dfwr).drop('total')
#         df_sales = df_sales.join(dft)
#         df_sales = df_sales.join(dfd)
#         sal_val = (.01 * df_sales['SAL⮕WON'] * df_sales['ACV']).round().astype(int)
#         sql_val = (.01 * df_sales['SQL⮕WON'] * df_sales['ACV']).round().astype(int)
#         value_rate = sal_val * df_sales['SALS / month']
        
        
#         df_sales['SAL Val'] = to_dollars(sal_val)
#         df_sales['SQL Val'] = to_dollars(sql_val) #[locale.currency(x, grouping=True).split('.')[0] for x in sql_val]
#         df_sales.loc[:, 'ACV'] = to_dollars(df_sales.ACV) #[locale.currency(x, grouping=True).split('.')[0] for x in df_sales.ACV]
#         df_sales.loc[:, 'SAL Value / Month'] = to_dollars(value_rate) #[locale.currency(x, grouping=True).split('.')[0] for x in value_rate]
        
#         for col in dfwr.columns:
#             df_sales.loc[:, col] = to_percent(df_sales.loc[:, col])
        
        
#         df_arr = dfr
#         df_arr = df_arr.join(dfn)
#         df_arr = df_arr.fillna('-')
#         monthly_rate = (.01 * df_arr['12-month NDR']) ** (1 / 12) - 1
#         # display(df_arr)
#         df_arr['Value / Month'] = df_arr['Current ARR'] * monthly_rate
        
#         df_arr.loc[:, 'Current ARR'] = to_dollars(df_arr['Current ARR'])
#         df_arr.loc[:, '12-month NDR'] = to_percent(df_arr.loc[:, '12-month NDR'])
#         df_arr.loc[:, 'Value / Month'] = to_dollars(df_arr['Value / Month'])
#         df_arr.index.name = None
#         st.dataframe(df_sales, width=1800)
#         st.dataframe(df_arr, width=800)


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


# class PredictorPlotter:
#     def __init__(self, pipe_stats_obj=None, include_sales_expansion=True):
#         if pipe_stats_obj is None:
#             pipe_stats_obj = smb.PipeStats()
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
#         deals = smb.Deals(starting=starting, ending_exclusive=ending_exclusive, include_sales_expansion=self.include_sales_expansion)
#         return deals.df_predicted
    
#     @ezr.cached_container
#     def _df_won(self):
#         ps = smb.PipeStats(pilots_are_new_biz=True, sales_expansion_are_new_biz=self.include_sales_expansion)
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
#         mph = smb.ModelParamsHist()
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
#         c = self.plot_prediction_history(since=since, ending_exclusive=ending_exclusive, units=units)
#         return hv.Overlay([ol, c]).options(legend_position='top')
                                         

# def display(hv_obj):
#     st.write(hv.render(hv_obj, backend='bokeh'))


# def plot_predict(ending_exclusive):        
#     since='1/1/2022'
#     ps = smb.PipeStats()
#     ps.enable_pickle_cache()
#     pp = PredictorPlotter(ps, include_sales_expansion=True)
#     display(pp.plot(since=since, ending_exclusive=ending_exclusive, units='m'))




# opts.defaults(opts.Curve(width=800, height=400, tools=['hover']))
        
        

# print_blob()


# ending_exclusive = st.text_input('The prediction end date', value='1/1/2023')
# plot_predict(ending_exclusive)
