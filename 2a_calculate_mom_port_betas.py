# ======================================================================================================================
#                           Replicate Barberis, Jin, and Wang (2021)
#                           Part 2a: calculate momemtum portfolio betas
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
#   Part 1: Merge portfolio and CRSP daily return data
# ======================================================================================================================
umd = pd.read_pickle("momentum_10_portfolio.pkl")
crsp_d = pd.read_pickle(data_folder + "/CRSP_d_19620701_20151231.pkl")
factor = pd.read_csv(data_folder + "/F-F_Research_Data_Factors_daily.csv", skiprows=3)

# Clean factor data
factor = factor.iloc[:-1, :]
factor.columns = ['date_new', 'Mkt_RF', 'SMB', 'HML', 'RF']
factor['date_new'] = pd.to_datetime(factor['date_new'], format='%Y%m%d')
for c in ['Mkt_RF', 'SMB', 'HML', 'RF']:
    factor[c] = pd.to_numeric(factor[c])
    factor[c] = factor[c] / 100


# Convert permno data format
umd['permno'] = umd['permno'].astype(int)
umd['permno'].isnull().sum()
crsp_d['permno'] = crsp_d['permno'].astype(int)
crsp_d['permno'].isnull().sum()


# Add factor data to crsp
crsp_d['date_new'] = pd.to_datetime(crsp_d.date)
crsp_d = crsp_d.merge(factor, how='left', on=['date_new'])
crsp_d['ret_rf'] = crsp_d.ret - crsp_d.RF


# Create umd one year window variable
umd['one_year_start'] = umd.hdate1
umd['one_year_end'] = umd['one_year_start'] + pd.Timedelta("365D")
umd['one_year_start_date'] = umd['one_year_start'].dt.date
umd['one_year_end_date'] = umd['one_year_end'].dt.date

# Create group number for each combination of permno and date
umd['group_num'] = umd.groupby(['permno', 'date']).ngroup()

# # Get news through SQL
# conn = sqlite3.connect(':memory:')
# umd.to_sql('umd', conn, index=False, if_exists="replace")
# crsp_d.to_sql('crsp_d', conn, index=False, if_exists="replace")
# 
# qry = '''
#     select  
#         a.permno, a.date, a.momr, a.momr_bp, a.hdate1, a.hdate2,  b.*
#     from
#         umd a, crsp_d b
#     where
#         a.permno = b.permno and (b.date between a.one_year_start_date and a.one_year_end_date)
#     '''
# umd_1Y_daily_ret = pd.read_sql_query(qry, conn)
# umd_1Y_daily_ret.to_pickle("umd_1Y_daily_ret.pkl")


#%%
# ======================================================================================================================
#   Part 2: Calculate beta for each portfolio-month
# ======================================================================================================================
group_groups = np.int(np.floor(umd.group_num.max() / 10000) + 1)
for g in range(group_groups):
    print("==========================================")
    print("I AM REGRESSING GROUP " + str(g))
    print("==========================================")

    if g != group_groups - 1:
        start_ind = g * 10000
        end_ind = (g + 1) * 10000

        cri = (umd.group_num >= start_ind) & (umd.group_num <= end_ind)
        umd_sub = umd.loc[cri].copy()
        crsp_d_sub = crsp_d.loc[crsp_d.permno.isin(umd_sub.permno)].copy()

    else:
        start_ind = g * 10000

        cri = (umd.group_num >= start_ind)
        umd_sub = umd.loc[cri].copy()
        crsp_d_sub = crsp_d.loc[crsp_d.permno.isin(umd_sub.permno)].copy()


    # Get news through SQL
    conn = sqlite3.connect(':memory:')
    umd_sub.to_sql('umd', conn, index=False, if_exists="replace")
    crsp_d_sub.to_sql('crsp_d', conn, index=False, if_exists="replace")

    qry = '''
        select  
            a.group_num, a.permno, a.date, a.momr, a.momr_bp, a.hdate1, a.hdate2, a.one_year_start_date, a.one_year_end_date, b.ret, b.ret_rf, b.Mkt_RF
        from
            umd a, crsp_d b
        where
            a.permno = b.permno and (b.date between a.one_year_start_date and a.one_year_end_date)
        '''
    umd_1Y_daily_ret = pd.read_sql_query(qry, conn)
    # umd_1Y_daily_ret.to_pickle("umd_1Y_daily_ret.pkl")

    def get_beta_alpha(df):
        try:
            X = sm.add_constant(df['Mkt_RF'])
            Y = df['ret_rf']

            result = sm.OLS(Y, X, missing='drop').fit()
            result.params
            output = pd.Series({'group_num': df.group_num.iloc[1]})
            output = output.append(result.params)
        except:
            output = pd.Series({'group_num': df.group_num.iloc[1], 'const':np.NaN, 'Mkt_RF':np.NaN})

        # return result.params.get('const'), result.params.get('Mkt_RF')
        return output


    beta_alpha = umd_1Y_daily_ret.groupby('group_num').apply(get_beta_alpha)
    # beta_alpha.to_pickle('beta_alpha_group_' + str(g) + '.pkl')



#%%
# ======================================================================================================================
#   Part 3: Aggregate beta for all portfolio-month
# ======================================================================================================================
beta_files = glob.glob('beta_alpha_group_*')

# Aggregate all beta output files
beta_alpha_all = pd.DataFrame()
for f in beta_files:
    temp = pd.read_pickle(f)
    beta_alpha_all = beta_alpha_all.append(temp, ignore_index=True)

beta_alpha_all = beta_alpha_all.sort_values('group_num')
beta_alpha_all = beta_alpha_all.reset_index(drop=True)


# Merge with UMD dataset
beta_alpha_all = beta_alpha_all.merge(umd, how='right', on=['group_num'])
beta_alpha_all = beta_alpha_all.drop_duplicates()


# Export
beta_alpha_all.to_pickle(data_folder + '/beta_alpha_all.pkl')


#%%
# ======================================================================================================================
#   Part 4: Calculate average beta by mom group and month
# ======================================================================================================================
beta_alpha_all = pd.read_pickle(data_folder + '/beta_alpha_all.pkl')

# Average beta of each momr portfolio-month
avg_beta_momr = beta_alpha_all.groupby(['momr','date']).Mkt_RF.mean().to_frame('avg_beta').reset_index()
temp = beta_alpha_all.groupby(['momr','date']).Mkt_RF.count().to_frame('num_stock').reset_index()
avg_beta_momr = avg_beta_momr.merge(temp, how='left', on=['momr','date'])

# Average beta of each momr_bp port.-month
avg_beta_momr_bp = beta_alpha_all.groupby(['momr_bp','date']).Mkt_RF.mean().to_frame('avg_beta_bp').reset_index()
temp = beta_alpha_all.groupby(['momr_bp','date']).Mkt_RF.count().to_frame('num_stock_bp').reset_index()
avg_beta_momr_bp = avg_beta_momr_bp.merge(temp, how='left', on=['momr_bp','date'])

# Aggregate two results
avg_beta_momr_all = avg_beta_momr.merge(avg_beta_momr_bp, how='left', left_on=['momr','date'], right_on=['momr_bp','date'])

# Export
avg_beta_momr_all.to_pickle(data_folder + '/avg_beta_mom_port.pkl')


#%% Parallel processing
# group_groups = np.int(np.floor(umd.group_num.max() / 10000) + 1)
#
# umd_split = np.array_split(umd, group_groups)
#
# def single_work(g, umd, crsp_d):
#     # crsp_d_sub = crsp_d.loc[crsp_d.permno.isin(umd_sub.permno)].copy()
#
#     if g != group_groups - 1:
#         start_ind = g * 10000
#         end_ind = (g + 1) * 10000
#
#         cri = (umd.group_num >= start_ind) & (umd.group_num <= end_ind)
#         umd_sub = umd.loc[cri].copy()
#         crsp_d_sub = crsp_d.loc[crsp_d.permno.isin(umd_sub.permno)].copy()
#
#     else:
#         start_ind = g * 10000
#
#         cri = (umd.group_num >= start_ind)
#         umd_sub = umd.loc[cri].copy()
#         crsp_d_sub = crsp_d.loc[crsp_d.permno.isin(umd_sub.permno)].copy()
#
#     # Get news through SQL
#     conn = sqlite3.connect(':memory:')
#     umd_sub.to_sql('umd', conn, index=False, if_exists="replace")
#     crsp_d_sub.to_sql('crsp_d', conn, index=False, if_exists="replace")
#
#     qry = '''
#         select
#             a.group_num, a.permno, a.date, a.momr, a.momr_bp, a.hdate1, a.hdate2, a.one_year_start_date, a.one_year_end_date, b.ret, b.ret_rf, b.Mkt_RF
#         from
#             umd a, crsp_d b
#         where
#             a.permno = b.permno and (b.date between a.one_year_start_date and a.one_year_end_date)
#         '''
#     umd_1Y_daily_ret = pd.read_sql_query(qry, conn)
#     # umd_1Y_daily_ret.to_pickle("umd_1Y_daily_ret.pkl")
#
#     def get_beta_alpha(df):
#         try:
#             X = sm.add_constant(df['Mkt_RF'])
#             Y = df['ret_rf']
#
#             result = sm.OLS(Y, X, missing='drop').fit()
#             result.params
#             output = pd.Series({'group_num': df.group_num.iloc[1]})
#             output = output.append(result.params)
#         except:
#             output = pd.Series({'group_num': df.group_num.iloc[1], 'const':np.NaN, 'Mkt_RF':np.NaN})
#
#         # return result.params.get('const'), result.params.get('Mkt_RF')
#         return output
#
#
#     beta_alpha = umd_1Y_daily_ret.groupby('group_num').apply(get_beta_alpha)
#     beta_alpha.to_pickle('beta_alpha_group_' + str(g) + '.pkl')
#
#     return beta_alpha
#
#
# # import multiprocessing as mp
# # from multiprocessing import Pool
# # from functools import partial
# # cores = mp.cpu_count()
# # pool = Pool(cores)
# # # for n, frame in enumerate(pool.imap(single_work, (umd_split,crsp_d)), start=1):
# # #     frame.to_pickle('{}'.format(n))
# # # pool.close()
# # # pool.join()
# #
# # if __name__ == '__main__':
# #     N= mp.cpu_count()
# #
# #     with mp.Pool(processes = N) as p:
# #         prod_x = partial(single_work, crsp_d=crsp_d)
# #         results = p.map(prod_x, umd_split[:2])
# #         results.to_pickle("beta_alpha_result_all.pkl")
#
# Pros = []
# all_reg_results = pd.DataFrame()
# from multiprocessing import Process
# import single_work
# def main():
#   for i in range(3):
#      print("Thread Started")
#      p = Process(target=single_work, args=(i, umd, crsp_d))
#      Pros.append(p)
#      p.start()
#
#   # block until all the threads finish (i.e. block until all function_x calls finish)
#   for t in Pros:
#      t.join()
#
# main()