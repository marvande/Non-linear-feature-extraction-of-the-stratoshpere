from scipy.spatial import distance_matrix
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import warnings
from sklearn.neighbors import NearestNeighbors
import scipy.sparse.linalg as linalg
import scipy.signal as signal
import seaborn as sns
import matplotlib.dates as mdates
from datetime import datetime
import os
import math 

from functions_takens import *

# Percentage of nearest neighbours:
PERC_NEIGH = 10
print(f'Percentage of nearest neighbours: {PERC_NEIGH/100}')

# Number of eigenmaps to compute:
NUM_EIGENVALUES = 21
print(f'Number of eigenmaps: {NUM_EIGENVALUES}')

# Data set to consider: ('raw/anomalies')
DATA = 'anomalies'
print(f'Data set considered: {DATA}')

# Path to input data:
INPUT_DATA = '../../../data/vandermeer/input_data/'
print(f'Path to input data: {INPUT_DATA}')

# Wether to do NLSA or Laplacian Eigenmaps. Takens True for NLSA
USE_TAKENS = True
print(f'Using takens embedding: {USE_TAKENS}')

# Time-step in Takens embedding:
TAU = 4 * 30 * 2
print(f'tau = {TAU/(4*30)} months')

if DATA == 'raw':
    PATH = '../../../data/vandermeer/pickles/raw/'
elif DATA == 'anomalies': 
    PATH = '../../../data/vandermeer/pickles/anomalies/'
if not os.path.exists(PATH):
        os.makedirs(PATH)
print(f'Global path: {PATH}')

PATH1 = PATH + str(PERC_NEIGH)+'perc/'
if not os.path.exists(PATH1):
        os.makedirs(PATH1)
print(f'Precise path: {PATH1}')

# path to simple_kernel:
PATH_SIMPLE = PATH1 + 'simple_kernel/'
if not os.path.exists(PATH_SIMPLE):
        os.makedirs(PATH_SIMPLE)
print(f'Path to simple kernel: {PATH_SIMPLE}')

PATH_SIMPLE_TAKENS = PATH1 + 'simple_kernel/takens/'
if not os.path.exists(PATH_SIMPLE_TAKENS):
        os.makedirs(PATH_SIMPLE_TAKENS)
print(f'Path to simple kernel for NLSA: {PATH_SIMPLE_TAKENS}')

# path to NLSA results:
PATH_TAKENS = PATH + str(PERC_NEIGH)+'perc/takens/'
if not os.path.exists(PATH_TAKENS):
        os.makedirs(PATH_TAKENS)
print(f'Path to NLSA results: {PATH_TAKENS}')


############## NLSA:

# load distance matrix: 
#distance_matrix = pd.read_pickle(PATH1+'distance_matrix.pkl').values

D = 150
print('Using D:{}'.format(D))
p = PATH1 + 'distance_matrix_{}.pkl'.format(D)
print('Reading distance matrix from:{}'.format(p))

distance_matrix = pd.read_pickle(PATH1 + 'distance_matrix_{}.pkl'.format(D)).values
print(f'Distance matrix shape: {distance_matrix.shape}')

# input data: 
if DATA == 'raw':
    df = pd.read_csv(INPUT_DATA + 'raw_data_coefficients.csv', sep=',')
elif DATA == 'anomalies': 
    df = pd.read_csv(INPUT_DATA + 'anomalies_coefficients.csv', sep=',')
print(f'Input data shape: {df.shape}')

# temporal data: 
time = df['Date']
time = pd.to_datetime(time)
df_ = df.drop(['Unnamed: 0','Date'], axis = 1)
df_time = pd.concat([time, df_], axis=1)
df_time = df_time.set_index('Date')

years = range(1979, 2019)
years = [str(y) for y in years]

# number of measures per year: 
counts = {}
sum_ = 0
for y in years:
    sum_ += len(df_time[y])
    counts[y] = len(df_time[y])
    
# positions of last measure per year:   
cs = [counts[i] for i in years]
last_year_pos = [np.sum(cs[:i+1]) for i in range(len(cs))]

# positions of last and first measure per winter: 
last_pos_winters = give_last_post_winters(last_year_pos, years, df_time)
first_pos_winters = give_first_post_winters(last_year_pos, years, df_time)
assert(len(last_pos_winters)==len(first_pos_winters))

# indices between first and last positions of winters: 
indices_of_points = []
for i in range(len(first_pos_winters)):
    indices_of_points.append(
        range(first_pos_winters[i] + TAU, last_pos_winters[i] + 1, 1))
for i in indices_of_points:
    assert (i[-1] - i[0] > 600)
print(indices_of_points)

# total number of points:
le = np.sum([len(i) for i in indices_of_points])

# time series corresponding to those points
time_nlsa = time[indices_of_points[0]]
for i in range(1, len(last_pos_winters)):
    time_nlsa = pd.concat([time_nlsa, time[indices_of_points[i]]], axis = 0)
assert(le==len(time_nlsa))

indices_m = []
for j in range(len(indices_of_points)):
    for i in range(len(indices_of_points[j])):
        indices_m.append(indices_of_points[j][i])
assert(len(indices_m)==le)


#Number of entire winters:
num_years = 2018-1980+1
print(f'Number of years: {num_years}')

m,n  = le, le
dist_Y = np.zeros((m,n))
print(f'New distance matrix shape: {dist_Y.shape}')
dist_Y = apply_takens(dist_Y, distance_matrix, indices_m, TAU, PATH1, NAME = D)


print(f'Reading upper triangle matrix at {PATH1}:')
dist_Y = pd.read_pickle(PATH1+'dist_Y_takens_{}.pkl'.format(D)).values

# Make matrix symetric:
takens_matrix = dist_Y + dist_Y.T

# Save NLSA distance matrix matrix:
print(f'Save NLSA distance matrix at {PATH1}')
takens_df = pd.DataFrame(takens_matrix)
takens_df.to_pickle(PATH1 + 'distance_matrix_takens_{}.pkl'.format(D))