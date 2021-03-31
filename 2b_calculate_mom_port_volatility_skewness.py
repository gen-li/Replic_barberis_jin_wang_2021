# ======================================================================================================================
#                           Replicate Barberis, Jin, and Wang (2021)
#                    Part 2b: calculate momemtum portfolio volatility and skewness
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
from distributed import Client
client = Client()
import statsmodels.api as sm

# Directory set up
project_dir = "/Users/genli/Dropbox/UBC/Course/2020 Term2/COMM 673/COMM673_paper_replica"   # Change to your project directory
data_folder = project_dir + "/data"
os.chdir(project_dir + "/_temp")


#%%
# ======================================================================================================================
#   Part 1: Merge portfolio and CRSP monthly return data
# ======================================================================================================================
umd = pd.read_pickle("momentum_10_portfolio.pkl")
crsp_m = pd.read_pickle(data_folder + "/CRSP_m_19620101_20151231.pkl")
factor = pd.read_csv(data_folder + "/F-F_Research_Data_Factors.csv", skiprows=3)

# Clean factor data
factor = factor.iloc[:1135, :]
factor.columns = ['date', 'Mkt_RF', 'SMB', 'HML', 'RF']
factor['date'] = pd.to_datetime(factor['date'], format='%Y%m')
for c in ['Mkt_RF', 'SMB', 'HML', 'RF']:
    factor[c] = pd.to_numeric(factor[c])
    factor[c] = factor[c] / 100

# Convert factor date to end of month
factor['date_new']=factor['date']+MonthEnd(0)
(factor['date_new'].dt.month != factor['date'].dt.month).sum()
factor = factor.drop(columns=['date'])

# Convert date in CRSP and check if crsp date is end date of month
crsp_m['date_new'] = pd.to_datetime(crsp_m.date)
crsp_m['date_new'] = crsp_m['date_new'] + MonthEnd(0)
(crsp_m['date_new'].dt.month != (crsp_m['date_new'] + MonthEnd(0)).dt.month).sum()


# Convert permno data format
umd['permno'] = umd['permno'].astype(int)
umd['permno'].isnull().sum()
crsp_m['permno'] = crsp_m['permno'].astype(int)
crsp_m['permno'].isnull().sum()

# Add factor data to crsp
crsp_m = crsp_m.merge(factor, how='left', on=['date_new'])
crsp_m['ret_rf'] = crsp_m.ret - crsp_m.RF

# Create umd one year window variable
umd['one_year_start'] = umd.hdate1
umd['one_year_end'] = umd['one_year_start'] + pd.Timedelta("365D")
umd['one_year_end'] = umd['one_year_end'] + MonthBegin(0)
(umd['one_year_end'].dt.month != umd['one_year_start'].dt.month).sum()

# umd['one_year_start_date'] = umd['one_year_start'].dt.date
# umd['one_year_end_date'] = umd['one_year_end'].dt.date

# Create group number for each combination of permno and date
umd['group_num'] = umd.groupby(['permno', 'date']).ngroup()


# Get news through SQL
conn = sqlite3.connect(':memory:')
umd.to_sql('umd', conn, index=False, if_exists="replace")
crsp_m.to_sql('crsp_m', conn, index=False, if_exists="replace")

qry = '''
    select  
        a.group_num, a.permno, a.date, a.momr, a.momr_bp, a.hdate1, a.hdate2, a.one_year_start, a.one_year_end, b.date_new, b.ret, b.ret_rf, b.Mkt_RF
    from
        umd a, crsp_m b
    where
        a.permno = b.permno and (b.date_new between a.one_year_start and a.one_year_end)
    '''
umd_1Y_monthly_ret = pd.read_sql_query(qry, conn)
umd[~umd.group_num.isin(umd_1Y_monthly_ret.group_num)].shape[0] # Check if any missing umd groupnum
# umd_1Y_monthly_ret.to_pickle("umd_1Y_monthly_ret.pkl")


#%%
# ======================================================================================================================
#   Part 2: Calculate annual returns for each portfolio and cross-sectional volatility and skewness
# ======================================================================================================================
umd_1Y_monthly_ret = pd.read_pickle("umd_1Y_monthly_ret.pkl")

# Calculate annual returns
umd_1Y_monthly_ret['ret'] = umd_1Y_monthly_ret.ret.fillna(0)
umd_1Y_monthly_ret['log_ret'] = np.log(1 + umd_1Y_monthly_ret.ret)
temp = umd_1Y_monthly_ret.groupby('group_num').log_ret.sum().reset_index()
temp['annu_ret'] = np.exp(temp.log_ret) - 1
temp = temp.drop(columns=['log_ret'])

umd_1Y_monthly_ret = umd_1Y_monthly_ret.merge(temp, how='left', on=['group_num'])
umd_1Y_monthly_ret.annu_ret.isnull().sum()

# Add number of monthly stock returns for each group_num
temp = umd_1Y_monthly_ret.groupby('group_num').ret.size().to_frame('num_month_ret').reset_index()
umd_1Y_monthly_ret = umd_1Y_monthly_ret.merge(temp, on='group_num', how='left')

umd_1Y_monthly_ret = umd_1Y_monthly_ret.sort_values(['group_num', 'date_new']).reset_index(drop=True)

# Collapse monthly portfolio return to annual level
umd_1Y_annual_ret = umd_1Y_monthly_ret[['group_num', 'permno', 'date', 'momr', 'momr_bp', 'hdate1',
                                        'hdate2', 'annu_ret', 'num_month_ret']].drop_duplicates()

umd_1Y_annual_ret.group_num.nunique()
umd_1Y_annual_ret.num_month_ret.describe()


# ######################################################################################################################
# Calculate cross-sectional volatility and skewness for each MOM portfolio-month
# ######################################################################################################################
umd_1Y_annual_ret['date'] = pd.to_datetime(umd_1Y_annual_ret['date'])
umd_1Y_annual_ret = umd_1Y_annual_ret.sort_values(['momr', 'date'])
umd_1Y_annual_ret['date_year'] = umd_1Y_annual_ret.date.dt.year
umd_1Y_annual_ret['date_month'] = umd_1Y_annual_ret.date.dt.month

umd_1Y_annual_ret[['date_year', 'date_month']].drop_duplicates().shape[0] # Check if date is unique date month
umd_1Y_annual_ret.date.nunique()

# replace those annu_ret with NaN if num_month_ret < 12
umd_1Y_annual_ret.loc[umd_1Y_annual_ret.num_month_ret < 12, 'annu_ret'] = np.nan

# Calculate momr volatility and skewness
momr_1Y_std_skew = pd.DataFrame()
# number of stocks in each momr-month portfolio
momr_1Y_std_skew = umd_1Y_annual_ret.groupby(['momr', 'date', 'date_year', 'date_month'])['annu_ret'].count()\
    .to_frame('momr_ret_num').reset_index()
# std of each momr-month portfolio
temp = umd_1Y_annual_ret.groupby(['momr', 'date', 'date_year', 'date_month'])['annu_ret'].std()\
    .to_frame('momr_ret_std').reset_index()
momr_1Y_std_skew = momr_1Y_std_skew.merge(temp, on=['momr', 'date', 'date_year', 'date_month'], how='left')

# skewness of each momr-month portfolio
temp = umd_1Y_annual_ret.groupby(['momr', 'date', 'date_year', 'date_month'])['annu_ret'].skew()\
    .to_frame('momr_ret_skew').reset_index()
momr_1Y_std_skew = momr_1Y_std_skew.merge(temp, on=['momr', 'date', 'date_year', 'date_month'], how='left')


# Calculate momr_bp volatility and skewness
momr_bp_1Y_std_skew = pd.DataFrame()
# number of stocks in each momr_bp-month portfolio
momr_bp_1Y_std_skew = umd_1Y_annual_ret.groupby(['momr_bp', 'date', 'date_year', 'date_month'])['annu_ret'].count()\
    .to_frame('momr_bp_ret_num').reset_index()
# std of each momr_bp-month portfolio
temp = umd_1Y_annual_ret.groupby(['momr_bp', 'date', 'date_year', 'date_month'])['annu_ret'].std()\
    .to_frame('momr_bp_ret_std').reset_index()
momr_bp_1Y_std_skew = momr_bp_1Y_std_skew.merge(temp, on=['momr_bp', 'date', 'date_year', 'date_month'], how='left')

# skewness of each momr_bp-month portfolio
temp = umd_1Y_annual_ret.groupby(['momr_bp', 'date', 'date_year', 'date_month'])['annu_ret'].skew()\
    .to_frame('momr_bp_ret_skew').reset_index()
momr_bp_1Y_std_skew = momr_bp_1Y_std_skew.merge(temp, on=['momr_bp', 'date', 'date_year', 'date_month'], how='left')


# Put together the results of two method of mom ranking 
umd_1Y_std_skew_all = momr_1Y_std_skew.merge(momr_bp_1Y_std_skew, how='left',
                                             left_on=['momr', 'date', 'date_year', 'date_month'],
                                             right_on=['momr_bp', 'date', 'date_year', 'date_month'])

# Export 
umd_1Y_std_skew_all.to_pickle(data_folder + '/umd_1Y_std_skew_all.pkl')
