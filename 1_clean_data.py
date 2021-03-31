# ======================================================================================================================
#                           Replicate Barberis, Jin, and Wang (2021)
#                                   Part 1: data cleaning
#
#                                       Author: Gen Li
#                                         03/14/2021
#
# ======================================================================================================================
import pandas as pd
import os
import wrds
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import sqlite3
import glob
from pandas.tseries.offsets import *
from scipy import stats
from time import process_time
import modin.pandas as pdm
from distributed import Client
client = Client()

# Directory set up
project_dir = "/Users/genli/Dropbox/UBC/Course/2020 Term2/COMM 673/COMM673_paper_replica"   # Change to your project directory
data_folder = project_dir + "/data"
os.chdir(project_dir + "/_temp")


#%%
# ======================================================================================================================
#   Part 1: Connect WRDS and Download data
# ======================================================================================================================
conn=wrds.Connection()


###################
# CRSP Block      #
###################
# sql similar to crspmerge macro
# added exchcd=-2,-1,0 to address the issue that stocks temp stopped trading
# without exchcd=-2,-1, 0 the non-trading months will be tossed out in the output
# leading to wrong cumret calculation in momentum step
# Code	Definition
# -2	Halted by the NYSE or AMEX
# -1	Suspended by the NYSE, AMEX, or NASDAQ
# 0	Not Trading on NYSE, AMEX, or NASDAQ
# 1	New York Stock Exchange
# 2	American Stock Exchange
# 3 The Nasdaq Stock Market(SM)

begdate = '07/01/1962'
enddate = '12/31/2015'


# # Doanload daily data
# crsp_d = conn.raw_sql("""
#                       select a.permno, a.permco, b.ncusip, a.date,
#                       b.shrcd, b.exchcd, b.siccd,
#                       a.ret, a.vol, a.shrout, a.prc, a.cfacpr, a.cfacshr
#                       from crsp.dsf as a
#                       left join crsp.dsenames as b
#                       on a.permno=b.permno
#                       and b.namedt<=a.date
#                       and a.date<=b.nameendt
#                       where a.date between '{begdate}' and '{enddate}'
#                       and b.exchcd between -2 and 3
#                       and b.shrcd between 10 and 11
#                       """)
#
# crsp_d.to_pickle(data_folder + "/CRSP_d_19620701_20151231.pkl")
# start_t = process_time()
# crsp_d = pd.read_pickle(data_folder + "/CRSP_d_19620701_20151231.pkl")
# end_t = process_time()
# print("Elapsed time:", end_t, start_t)


# Doanload monthly data
# crsp_m = conn.raw_sql("""
#                       select a.permno, a.permco, b.ncusip, a.date,
#                       b.shrcd, b.exchcd, b.siccd,
#                       a.ret, a.vol, a.shrout, a.prc, a.cfacpr, a.cfacshr
#                       from crsp.msf as a
#                       left join crsp.msenames as b
#                       on a.permno=b.permno
#                       and b.namedt<=a.date
#                       and a.date<=b.nameendt
#                       where a.date between '{begdate}' and '{enddate}'
#                       and b.exchcd between -2 and 3
#                       and b.shrcd between 10 and 11
#                       """)

# crsp_m.to_pickle(data_folder + "/CRSP_m_19620101_20151231.pkl")


#%%
# ======================================================================================================================
#   Part 2: Clean data
# ======================================================================================================================
# Read downloaded data
crsp_m = pd.read_pickle(data_folder + "/CRSP_m_19620101_20151231.pkl")

# Change variable format to int
crsp_m[['permco','permno','shrcd','exchcd']] = crsp_m[['permco','permno','shrcd','exchcd']].astype(int)

# Line up date to be end of month
crsp_m['date']=pd.to_datetime(crsp_m['date'])

#######################################################
# Create Momentum Portfolio                           #
# Measures Based on Past (J) Month Compounded Returns #
#######################################################

J = 11 # Formation Period Length: J can be between 3 to 12 months
K = 1 # Holding Period Length: K can be between 3 to 12 months

_tmp_crsp = crsp_m[['permno','date','ret']].sort_values(['permno','date']).set_index('date')

# Replace missing return with 0
_tmp_crsp['ret']=_tmp_crsp['ret'].fillna(0)

# Calculate rolling cumulative return
# by summing log(1+ret) over the formation period
_tmp_crsp['logret']=np.log(1+_tmp_crsp['ret'])
# for l in range(2,13):
#     _tmp_crsp['L' + str(l) + '_logret'] = _tmp_crsp.groupby(['permno'])['logret'].shift(l)
#
# _tmp_crsp['L2_12_logret'] = _tmp_crsp[['L2_logret', 'L3_logret', 'L4_logret',
#        'L5_logret', 'L6_logret', 'L7_logret', 'L8_logret', 'L9_logret',
#        'L10_logret', 'L11_logret', 'L12_logret']].sum(axis=1, min_count=11)
#
# _tmp_crsp['cumret_1'] = np.exp(_tmp_crsp.L2_12_logret) - 1

umd = _tmp_crsp.groupby(['permno'])['logret'].rolling(J, min_periods=J).sum()
umd = umd.reset_index()
umd['cumret']=np.exp(umd['logret'])-1
umd = umd.sort_values(['permno','date']).set_index('date')
umd["cumret_2"] = umd.groupby(['permno'])['cumret'].shift(2)
umd = umd.reset_index().drop(columns=['logret','cumret']).rename(columns={"cumret_2": "L2_L12_cumret"})

_tmp_crsp = _tmp_crsp.merge(umd, on=['permno','date'])
_tmp_crsp = _tmp_crsp.drop(columns=["logret"])


########################################
# Formation of 10 Momentum Portfolios  #
########################################
# For each date: assign ranking 1-10 based on cumret
# 1=lowest 10=highest cumret
umd = _tmp_crsp.copy()
umd=umd.dropna(axis=0, subset=['L2_L12_cumret'])
umd['momr']=umd.groupby('date')['L2_L12_cumret'].transform(lambda x: pd.qcut(x, 10, labels=False))

umd.momr=umd.momr.astype(int)
umd['momr'] = umd['momr']+1

umd['form_date'] = umd['date']
# umd['medate'] = umd['date']+MonthEnd(0)
umd['hdate1']=umd['form_date']+MonthBegin(-1)
umd['hdate2']=umd['form_date']+MonthEnd(K-1)

# umd.to_pickle("momentum_10_portfolio.pkl")


#################################################
# Add momentum breakpoint from French data lib  #
#################################################
umd = pd.read_pickle("momentum_10_portfolio.pkl")
mom_breakp = pd.read_csv(data_folder + "/Prior_2-12_Breakpoints.csv", skiprows=3, header=None)
col_name = ['date_month', 'n'] + ['p_' + str(x) for x in range(5, 105, 5) ]
mom_breakp.columns = col_name

# Convert date month
mom_breakp['date_day'] = pd.to_datetime(mom_breakp.date_month, format="%Y%m")
mom_breakp = mom_breakp[['date_day'] + col_name]

# Convert percentage return to unit level
p_col = ['p_' + str(x) for x in range(5, 105, 5) ]
for c in p_col:
    mom_breakp[c] = mom_breakp[c]/100

# Keep sub data and merge
col_new = ['date_day'] + ['p_' + str(x) for x in range(10, 110, 10) ]
mom_breakp_sub = mom_breakp[col_new].copy()

# Merge French breakpoint data and assign french mom rank
umd = umd.merge(mom_breakp_sub, how='left', left_on=['hdate1'], right_on=['date_day'])

umd['momr_bp'] = np.nan
cri = umd.L2_L12_cumret < umd.p_10
umd.loc[cri, 'momr_bp'] = 1

for n in range(1, 10):
    cri = (umd.L2_L12_cumret >= umd['p_' + str(n) + '0']) & (umd.L2_L12_cumret < umd['p_' + str(n+1) + '0'])
    umd.loc[cri, 'momr_bp'] = n+1

cri = umd.L2_L12_cumret >= umd.p_100
umd.loc[cri, 'momr_bp'] = 10

# Clean variables
umd['momr_bp'] = umd.momr_bp.astype(int)
umd = umd[['permno', 'date', 'ret', 'L2_L12_cumret', 'momr', 'momr_bp', 'form_date', 'hdate1', 'hdate2']].copy()

# Export
# umd.to_pickle("momentum_10_portfolio.pkl")


