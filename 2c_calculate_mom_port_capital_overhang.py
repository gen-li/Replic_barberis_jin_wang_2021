# ======================================================================================================================
#                           Replicate Barberis, Jin, and Wang (2021)
#                    Part 2c: calculate momemtum portfolio capital overhang
#
#                                       Author: Gen Li
#                                         03/14/2021
#
#   Note: I run Part 1 on Google Datalab from Google Cloud Platform (GCP). It takes around 1.5h to finish part 1 with
#         32-core CPU and parallel computation.
#
#
# ======================================================================================================================
import pandas as pd
import os
import wrds
import numpy as np
from fuzzywuzzy import fuzz
import sqlite3
import glob
from pandas.tseries.offsets import *
from scipy import stats
from time import process_time
# import modin.pandas as pd
# from distributed import Client
# client = Client()
import statsmodels.api as sm
from multiprocessing import Pool, cpu_count


# Directory set up
project_dir = "/Users/genli/Dropbox/UBC/Course/2020 Term2/COMM 673/COMM673_paper_replica"   # Change to your project directory
data_folder = project_dir + "/data"
os.chdir(project_dir + "/_temp")



#%%
# ======================================================================================================================
#   Part 1: Calculate purchase price R_i for each stock-month
# ======================================================================================================================
def applyParallel(dfGrouped, func):
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])

    return pd.concat(ret_list, ignore_index=True)


def calculate_R(df):
    print("I am calculating capital overhang for " + str(df.permno.values[0]) + " on " + str(df.date_new.values[0]))
    cri = (crsp_m_sub['permno'] == df.permno.values[0]) & ((crsp_m_sub['date_new'] <= df.date_new.values[0]))
    df_temp = crsp_m_sub[cri].copy()
    df_temp = df_temp.sort_values("date")

    # calculate cum product part
    df_temp['cum_prod'] = df_temp['1_minus_V_t'].cumprod()
    df_temp['cum_prod_last'] = df_temp['cum_prod'].iloc[-1]
    df_temp['cum_prod'] = df_temp['cum_prod_last'] / df_temp['cum_prod']

    # calculate purchase price weights
    df_temp['V_t_cum_prod'] = df_temp['V_t'] * df_temp['cum_prod']
    df_temp['V_t_cum_prod'] = df_temp['V_t_cum_prod'] / df_temp['V_t_cum_prod'].sum()  # normalize weights such that sum = 1

    # calculate R
    df_temp['R'] = df_temp['V_t_cum_prod'] * df_temp['prc'].abs()

    # calculate number of months used
    df_temp['num_obs'] = df_temp['R'].notnull().sum()

    #   df_temp['R_sum'] = df_temp['R'].sum()
    df_result = pd.DataFrame({'permno': df_temp.permno.values[-1], 'date_new': df_temp.date_new.values[-1],
                              'R': df_temp['R'].sum(), 'num_obs': df_temp['num_obs'].values[-1]},
                             index=df_temp.index.values)

    return df_result


if __name__ == '__main__':
    crsp_m = pd.read_pickle(data_folder + "/CRSP_m_19620101_20151231.pkl")

    # Convert date in CRSP and check if crsp date is end date of month
    crsp_m['date_new'] = pd.to_datetime(crsp_m.date)
    crsp_m['date_new'] = crsp_m['date_new'] + MonthEnd(0)
    (crsp_m['date_new'].dt.month != (crsp_m['date_new'] + MonthEnd(0)).dt.month).sum()

    # Convert vol back to one unit share
    crsp_m['vol'] = crsp_m['vol'] * 100
    crsp_m['shrout'] = crsp_m['shrout'] * 1000

    # V_t-n
    crsp_m_sub = crsp_m.copy()
    crsp_m_sub['V_t'] = crsp_m_sub.vol / crsp_m_sub.shrout
    crsp_m_sub['1_minus_V_t'] = 1 - crsp_m_sub.V_t

    # cum prod 1 - V_t-n+_\tau
    crsp_m_sub = crsp_m_sub.sort_values(['permno', 'date_new'])

    crsp_m_grouped = crsp_m_sub.iloc[:100,:].groupby(['permno', 'date_new'])

    # Parallel computing on Google datalab
    result = applyParallel(crsp_m_grouped, calculate_R)
    result = result.drop_duplicates()
    result.to_pickle('capital_overhang_R_parallel.pkl')
    print("I am done.")

    # !gsutil cp 'capital_overhang_R_parallel.pkl' gs://comm673/data/capital_overhang_R_parallel.pkl



#%%
# ======================================================================================================================
#   Part 2: Calculate capital overhang for each stock-month
# ======================================================================================================================
# cap_overhang = pd.read_pickle(data_folder + '/capital_overhang_R_parallel.pkl')
cap_overhang1 = pd.read_pickle(data_folder + '/capital_overhang_R_parallel_first_half.pkl')
cap_overhang2 = pd.read_pickle(data_folder + '/capital_overhang_R_parallel_second_half.pkl')
cap_overhang = cap_overhang1.append(cap_overhang2,ignore_index=True)
crsp_m = pd.read_pickle(data_folder + "/CRSP_m_19620101_20151231.pkl")
umd = pd.read_pickle("momentum_10_portfolio.pkl")

# Convert date in CRSP and check if crsp date is end date of month
crsp_m['date_new'] = pd.to_datetime(crsp_m.date)
crsp_m['date_new'] = crsp_m['date_new'] + MonthEnd(0)
(crsp_m['date_new'].dt.month != (crsp_m['date_new'] + MonthEnd(0)).dt.month).sum()

# Add R purchase price
crsp_m = crsp_m.merge(cap_overhang, how='left', on=['permno','date_new'])

# R should be applied to next price and return obs.
crsp_m = crsp_m.sort_values(['permno', 'date'])
crsp_m = crsp_m.set_index('date')
crsp_m['L1_R'] = crsp_m.groupby(['permno'])['R'].shift(1)
crsp_m = crsp_m.reset_index()
crsp_m['date'] = pd.to_datetime(crsp_m['date'])

# Calculate capital overhang
crsp_m.loc[crsp_m.R == 0, 'R'] = np.nan #######!!!!!!!
crsp_m.loc[crsp_m.L1_R == 0, 'L1_R'] = np.nan #######!!!!!!!
crsp_m['cap_overhang'] = (crsp_m.prc.abs() - crsp_m.L1_R) / crsp_m.L1_R
# crsp_m['cap_overhang'] = (crsp_m.prc.abs() - crsp_m.R) / crsp_m.R

# Merge capital overhang back to umd portfolio data
crsp_m_sub = crsp_m[['date', 'permno', 'prc', 'R','L1_R', 'cap_overhang']].copy()
umd = umd.merge(crsp_m_sub, how='left', on=['permno', 'date'])

# Get mom portfolio mean capital overhang in each month
momr_cap_overhang = umd.groupby(['momr','date']).cap_overhang.mean().to_frame('avg_cap_overhang').reset_index()
temp = umd.groupby(['momr','date']).cap_overhang.count().to_frame('num_stock').reset_index()
momr_cap_overhang = momr_cap_overhang.merge(temp, on=['momr','date'])

momr_bp_cap_overhang = umd.groupby(['momr_bp','date']).cap_overhang.mean().to_frame('avg_cap_overhang_bp').reset_index()
temp = umd.groupby(['momr_bp','date']).cap_overhang.count().to_frame('num_stock_bp').reset_index()
momr_bp_cap_overhang = momr_bp_cap_overhang.merge(temp, on=['momr_bp','date'])

momr_cap_overhang_all = momr_cap_overhang.merge(momr_bp_cap_overhang,how='left', left_on=['momr','date'], right_on=['momr_bp','date'])

# Export
momr_cap_overhang_all.to_pickle(data_folder + '/umd_cap_overhang.pkl')



#%%
#
# #%% test
# momr_cap_overhang_all_sub = momr_cap_overhang_all[momr_cap_overhang_all.date <= pd.to_datetime("2014-12-31")]
# test = momr_cap_overhang_all.groupby('momr').avg_cap_overhang.mean().to_frame('avg_cap_overhang').reset_index()
# test = momr_cap_overhang_all.groupby('momr_bp').avg_cap_overhang_bp.mean().to_frame('avg_cap_overhang_bp').reset_index()
#
#
# # test = umd[['date']].drop_duplicates()
# # test['year'] = test.date.dt.year
# # test['month'] = test.date.dt.month
# # test.duplicated(subset=['year','month']).sum()
#
# #%% CHECK WHERE I DID WRONG
# crsp_m = pd.read_pickle(data_folder + "/CRSP_m_19620101_20151231.pkl")
#
# # Convert date in CRSP and check if crsp date is end date of month
# crsp_m['date_new'] = pd.to_datetime(crsp_m.date)
# crsp_m['date_new'] = crsp_m['date_new'] + MonthEnd(0)
# (crsp_m['date_new'].dt.month != (crsp_m['date_new'] + MonthEnd(0)).dt.month).sum()
#
# # Convert vol back to one unit share
# crsp_m['vol'] = crsp_m['vol']*100
# crsp_m['shrout'] = crsp_m['shrout']*1000
#
# # V_t-n
# crsp_m_sub = crsp_m.copy()
# crsp_m_sub['V_t'] = crsp_m_sub.vol/ crsp_m_sub.shrout
# crsp_m_sub['1_minus_V_t'] = 1 - crsp_m_sub.V_t
#
# # cum prod 1 - V_t-n+_\tau
# crsp_m_sub = crsp_m_sub.sort_values(['permno', 'date_new'])
# crsp_m_grouped = crsp_m_sub.iloc[:100,:].groupby(['permno', 'date_new'])
#
# test = [group for name, group in crsp_m_grouped]
# aa = calculate_R(test[0])
# # # test
# # cri = (crsp_m_sub['permno'] == 10066) & ((crsp_m_sub['date_new'] <= pd.to_datetime('2001-10-31')))
# # df_temp = crsp_m_sub[cri].copy()
# # df_temp = df_temp.sort_values("date")
# #
# # # calculate cum product part
# # df_temp['cum_prod'] = df_temp['1_minus_V_t'].cumprod()
# # df_temp['cum_prod_last'] = df_temp['cum_prod'].iloc[-1]
# # df_temp['cum_prod'] = df_temp['cum_prod_last'] / df_temp['cum_prod']
# #
# # # calculate purchase price weights
# # df_temp['V_t_cum_prod'] = df_temp['V_t'] * df_temp['cum_prod']
# # df_temp['V_t_cum_prod'] = df_temp['V_t_cum_prod']/df_temp['V_t_cum_prod'].sum() # normalize weights such that sum = 1
# #
# # # calculate R
# # df_temp['R'] = df_temp['V_t_cum_prod'] * df_temp['prc'].abs()
# #
# # # calculate number of months used
# # df_temp['num_obs'] = df_temp['R'].notnull().sum()
#
#
#
# def calculate_R(df):
#     print("I am calculating capital overhang for " + str(df.permno.values[0]) + " on " + str(df.date_new.values[0]))
#     cri = (crsp_m_sub['permno'] == df.permno.values[0]) & ((crsp_m_sub['date_new'] <= df.date_new.values[0]))
#     df_temp = crsp_m_sub[cri].copy()
#     df_temp = df_temp.sort_values("date")
#
#     # calculate cum product part
#     df_temp['cum_prod'] = df_temp['1_minus_V_t'].cumprod()
#     df_temp['cum_prod_last'] = df_temp['cum_prod'].iloc[-1]
#     df_temp['cum_prod'] = df_temp['cum_prod_last'] / df_temp['cum_prod']
#
#     # calculate purchase price weights
#     df_temp['V_t_cum_prod'] = df_temp['V_t'] * df_temp['cum_prod']
#     df_temp['V_t_cum_prod'] = df_temp['V_t_cum_prod'] / df_temp[
#         'V_t_cum_prod'].sum()  # normalize weights such that sum = 1
#
#     # calculate R
#     df_temp['R'] = df_temp['V_t_cum_prod'] * df_temp['prc'].abs()
#
#     # calculate number of months used
#     df_temp['num_obs'] = df_temp['R'].notnull().sum()
#
#     #   df_temp['R_sum'] = df_temp['R'].sum()
#     df_result = pd.DataFrame({'permno': df_temp.permno.values[-1], 'date_new': df_temp.date_new.values[-1],
#                               'R': df_temp['R'].sum()},index=df_temp.index.values)
#
#     return df_result
#
#
# # Parallel computing on Google datalab
# result = applyParallel(crsp_m_grouped, calculate_R)
# result = result.drop_duplicates()