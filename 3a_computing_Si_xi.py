# ======================================================================================================================
#                           Replicate Barberis, Jin, and Wang (2021)
#                              Part 3a: computing expected returns
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
# from sympy import symbols, Eq, solve
from sympy import solve, Poly, Eq, Function, exp, symbols, sqrt
from scipy.optimize import fsolve
import math
from math import sqrt, gamma
import timeit
from scipy.integrate import quad
from scipy.special import kv


# Directory set up
project_dir = "/Users/genli/Dropbox/UBC/Course/2020 Term2/COMM 673/COMM673_paper_replica"   # Change to your project directory
data_folder = project_dir + "/data"
os.chdir(project_dir + "/_temp")


#%% Set parameters
nu = 7.5
sigma_m = 0.25
Rf = 1

# Theta mi
momr_avg_theta_all = pd.read_pickle(data_folder + '/momr_avg_theta_all.pkl')

theta_mi_all = momr_avg_theta_all.avg_theta_mi
theta_i_minus1_all = theta_mi_all

momr_avg_theta_all.to_csv(data_folder + '/momr_avg_theta_all.csv') # Export to allow julia to read

gamma_hat, b0 = (0.6, 0.6)
alpha, delta, lamb = (0.7, 0.65, 1.5)

# ***************
# Beta
# ***************
avg_beta_momr_all = pd.read_pickle(data_folder + '/avg_beta_mom_port.pkl')
avg_beta_momr_all = avg_beta_momr_all[avg_beta_momr_all.date <= pd.to_datetime("2014-12-31")]

momr_beta = avg_beta_momr_all.groupby(['momr']).avg_beta.mean().to_frame('avg_beta').reset_index()
momr_beta.to_csv(data_folder + '/momr_avg_beta_all.csv') # Export to allow julia to read

# ***************
# Gain overhang
# ***************
momr_cap_overhang_all = pd.read_pickle(data_folder + '/umd_cap_overhang.pkl')
momr_cap_overhang_all = momr_cap_overhang_all[momr_cap_overhang_all.date <= pd.to_datetime("2014-12-31")]

momr_gi = momr_cap_overhang_all.groupby(['momr']).avg_cap_overhang.mean().to_frame('avg_gi').reset_index()
momr_gi.to_csv(data_folder + '/momr_avg_g_i_all.csv') # Export to allow julia to read


#%%
# ======================================================================================================================
# Part 1: Calculating S and \xi
# ======================================================================================================================
umd_1Y_std_skew_all = pd.read_pickle(data_folder + '/umd_1Y_std_skew_all.pkl')
umd_1Y_std_skew_all = umd_1Y_std_skew_all[umd_1Y_std_skew_all.date <= pd.to_datetime("2014-12-31")]

# Get Std(R) and Skew(R)
momr_std = umd_1Y_std_skew_all.groupby(['momr']).momr_ret_std.mean().to_frame('avg_std').reset_index()
momr_skew = umd_1Y_std_skew_all.groupby(['momr']).momr_ret_skew.mean().to_frame('avg_skew').reset_index()
momr_std_skew = momr_std.merge(momr_skew, how='left', on=['momr'])

# ***************
# Solve S and xi
# ***************
def equation_std_skew(p,*args):
    Si, xi = p
    std_R, skew_R = args
    return [(nu/(nu-2) * Si + ((2 * nu ** 2) / ((nu-2)**2 * (nu-4))) * xi**2) ** (0.5) - std_R,
            ((2 * xi * sqrt(nu * (nu - 4)) /
              (sqrt(Si) * ((2 * nu * xi ** 2) / Si + (nu - 2) * (nu - 4)) ** (3 / 2)))
             * (3 * (nu - 2) + (8 * nu * xi ** 2) / (Si * (nu - 6))))- skew_R]


def fsolve_si_xi(std_R, skew_R):
    Si, xi = fsolve(equation_std_skew,(0.1, 0.1),args=(std_R, skew_R))
    # return {'Si':Si, 'xi':xi}
    return Si, xi


start = timeit.default_timer()
momr_std_skew['Si'], momr_std_skew['xi'] = zip(*momr_std_skew.apply(lambda x: fsolve_si_xi(x.avg_std, x.avg_skew), axis=1))

stop = timeit.default_timer()
execution_time = stop - start
print("Program Executed in "+str(execution_time)) # It returns time in seconds

momr_std_skew.to_csv(data_folder + '/momr_avg_std_skew_Si_xi_all.csv') # Export to allow julia to read

