# ======================================================================================================================
#                           Replicate Barberis, Jin, and Wang (2021)
#                    Part 2d: calculate momemtum portfolio value-weighted returns
#
#                                       Author: Gen Li
#                                         03/14/2021
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
umd = pd.read_pickle("momentum_10_portfolio.pkl")
crsp_m = pd.read_pickle(data_folder + "/CRSP_m_19620101_20151231.pkl")

# Convert date in CRSP and check if crsp date is end date of month
crsp_m['date_new'] = pd.to_datetime(crsp_m.date)
crsp_m['date_new'] = crsp_m['date_new'] + MonthEnd(0)
(crsp_m['date_new'].dt.month != (crsp_m['date_new'] + MonthEnd(0)).dt.month).sum()

# Convert date in umd
umd['date_new'] = umd.date + MonthEnd(0)
(umd['date'].dt.month != (umd['date_new'] + MonthEnd(0)).dt.month).sum()

# market value
crsp_m['p']=crsp_m['prc'].abs()/crsp_m['cfacpr'] # price adjusted
crsp_m['tso']=crsp_m['shrout']*crsp_m['cfacshr']*1e3 # total shares out adjusted
crsp_m['me'] = crsp_m['p']*crsp_m['tso']/1e6 # market cap in $mil

# last market value
crsp_m = crsp_m.sort_values(['permno','date_new'])
crsp_m = crsp_m.set_index('date_new')
crsp_m['L1_me'] = crsp_m.groupby('permno').me.shift(1)
crsp_m = crsp_m.reset_index()

# add last me
crsp_m_sub = crsp_m[['permno','date_new','L1_me']].copy()
umd = umd.merge(crsp_m_sub, how='left', on=['permno','date_new'])

# Fill missing last me
umd['L1_me'] = umd['L1_me'].fillna(0)

# calculate value weighted portfolio return
momr_vw_ret = umd.groupby(['momr','date']).apply(lambda x: np.average(x.ret, weights=x.L1_me)).to_frame('vw_ret').reset_index()
momr_bp_vw_ret = umd.groupby(['momr_bp','date']).apply(lambda x: np.average(x.ret, weights=x.L1_me)).to_frame('vw_ret_bp').reset_index()
momr_vw_ret_all = momr_vw_ret.merge(momr_bp_vw_ret,how='left', left_on=['momr','date'], right_on=['momr_bp','date'])

# Export
# momr_vw_ret_all.to_pickle(data_folder + '/umd_vw_ret.pkl')

#%% Generate expected return output for mom 1- 10
momr_vw_ret_all = pd.read_pickle(data_folder + '/umd_vw_ret.pkl')

test = momr_vw_ret_all.groupby('momr').vw_ret.mean().to_frame('mean_vw_ret').reset_index()
test['annualized_mean_vw_ret'] = test['mean_vw_ret']*12

test.to_csv(data_folder + "/momr_avg_expected_returns_all.csv")


#%%
std_skew = pd.read_pickle(data_folder + '/umd_1Y_std_skew_all.pkl')
test = std_skew.groupby('momr').momr_ret_std.mean().to_frame('mean_std').reset_index()
test['mean_std'] = test['mean_std']

test = std_skew.groupby('momr').momr_ret_skew.mean().to_frame('mean_skew').reset_index()
test['mean_std'] = test['mean_std']

import matplotlib.pyplot as plt

test2 = std_skew.groupby('momr').momr_ret_skew.mean().to_frame('mean_skew').reset_index()
test = test.merge(test2, on='momr')

# # Fixing random state for reproducibility
# np.random.seed(19680801)
#
#
# x = np.arange(0.0, 50.0, 2.0)
# y = x ** 1.3 + np.random.rand(*x.shape) * 30.0
# s = np.random.rand(*x.shape) * 800 + 500

# plt.scatter(std_skew.momr_ret_std, std_skew.momr_ret_skew, c="g", alpha=0.5, marker=r'$\clubsuit$',
#             label="mom")
plt.scatter(test.mean_std, test.mean_skew, c="g", alpha=0.5, marker=r'$\clubsuit$',
            label="mom")
plt.xlabel("standard deviation")
plt.ylabel("Skewnewss")
plt.legend(loc='upper left')
plt.show()

