# ======================================================================================================================
#                           Replicate Barberis, Jin, and Wang (2021)
#                    Part 2d: calculate momemtum portfolio \theta mi stocks' market weights
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
# ======================================================================================================================
#   Part 1: Calculate XXXXX
# ======================================================================================================================
umd = pd.read_pickle("momentum_10_portfolio.pkl")
crsp_m = pd.read_pickle(data_folder + "/CRSP_m_19620101_20151231.pkl")

# Convert date in CRSP and check if crsp date is end date of month
crsp_m['date_new'] = pd.to_datetime(crsp_m.date)
crsp_m['date_new'] = crsp_m['date_new'] + MonthEnd(0)
(crsp_m['date_new'].dt.month != (crsp_m['date_new'] + MonthEnd(0)).dt.month).sum()

# Convert date in umd
umd['date_new'] = umd.date + MonthEnd(0)
(umd['date'].dt.month != (umd['date_new'] + MonthEnd(0)).dt.month).sum()

# market value (me)
crsp_m['p']=crsp_m['prc'].abs()/crsp_m['cfacpr'] # price adjusted
crsp_m['tso']=crsp_m['shrout']*crsp_m['cfacshr']*1e3 # total shares out adjusted
crsp_m['me'] = crsp_m['p']*crsp_m['tso'] # market cap in $mil

# add me
crsp_m_sub = crsp_m[['permno','date_new','me']].copy()
umd = umd.merge(crsp_m_sub, how='left', on=['permno','date_new'])

# Calculate total market value of all stocks in a month
total_me_m = umd.groupby(['date']).me.sum().to_frame('momth_total_me').reset_index()

# Calculate total market value of all stocks for each decile portfolio in a month
decile_me_m = umd.groupby(['date','momr']).me.sum().to_frame('momth_momr_me').reset_index()
decile_me_m_bp = umd.groupby(['date','momr_bp']).me.sum().to_frame('momth_momr_me_bp').reset_index()
# Note: I use mean() instead of sum() because we need to divide the total market value by the stock number in the decile portfolio

# Merge total me
# umd = umd.merge(decile_me_m, how='left', on=['date','momr'])
decile_me_m_all = decile_me_m.merge(decile_me_m_bp, how='left', left_on=['momr','date'], right_on=['momr_bp','date'])
decile_me_m_all = decile_me_m_all.merge(total_me_m, how='left', on=['date'])

# calculate market weights
decile_me_m_all['theta_mi'] = decile_me_m_all.momth_momr_me / decile_me_m_all.momth_total_me
decile_me_m_all['theta_mi_bp'] = decile_me_m_all.momth_momr_me_bp / decile_me_m_all.momth_total_me

# average theta_mi
momr_avg_theta = decile_me_m_all.groupby(['momr']).theta_mi.mean().to_frame('avg_theta_mi').reset_index()
momr_avg_theta_bp = decile_me_m_all.groupby(['momr_bp']).theta_mi_bp.mean().to_frame('avg_theta_mi_bp').reset_index()
momr_avg_theta_all = momr_avg_theta.merge(momr_avg_theta_bp, how='left', left_on=['momr'], right_on=['momr_bp'])

# export
momr_avg_theta_all.to_pickle(data_folder + '/momr_avg_theta_all.pkl')
