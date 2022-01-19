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


def give_last_post_winters(last_year_pos, years, df_time):
    last_pos_winters = []
    
    for i in range(len(years)):
        jan = len(df_time['X1'][years[i]+'-01'])
        feb = len(df_time['X1'][years[i]+'-02'])
        mar = len(df_time['X1'][years[i]+'-03'])
        apr = len(df_time['X1'][years[i]+'-04'])
        spring = np.sum([jan, feb, mar, apr])
        if i > 0:
            last_pos_winters.append(spring+last_year_pos[i-1])
    return last_pos_winters

def give_first_post_winters(last_year_pos, years, df_time):
    first_pos_winters = []
    for i in range(len(years)):
        oct_ = len(df_time['X1'][years[i]+'-10'])
        nov = len(df_time['X1'][years[i]+'-11'])
        dec = len(df_time['X1'][years[i]+'-12'])
        winter = np.sum([oct_, nov, dec])
        if i > 0:
            first_pos_winters.append(last_year_pos[i-1]-winter+1)
    return first_pos_winters

def apply_takens(dist_Y, distance_matrix, indices_m, TAU, PATH1, NAME):
    m, n = dist_Y.shape[0], dist_Y.shape[1]
    print(f'New distance matrix being constructed of shape {dist_Y.shape}')
    for i in tqdm(range(m)):
        # construct upper triangle as matrix is symmetric:
        for j in range(n):
            # check we're in upper triangle:
            if j > i:
                # sum to put in squared distances
                sum_ = 0
                indice_row = indices_m[i]
                indice_col = indices_m[j]
                #print(indice_row, indice_col)
                for t in range(TAU):
                    # construct form the past so sum up from indice -> indice-tau
                    sum_ += (distance_matrix[indice_row - t, indice_col - t])**2        
                    # add new distance:
                dist_Y[i,j] = math.sqrt(sum_)

    dist_Y_df = pd.DataFrame(dist_Y)
    
    print(f'Saving upper triangle matrix at {PATH1}')
    dist_Y_df.to_pickle(PATH1 + f'dist_Y_takens_{NAME}.pkl')
    return dist_Y