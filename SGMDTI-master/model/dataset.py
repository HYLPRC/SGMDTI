import argparse

import math
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter, defaultdict
from scipy.spatial.distance import euclidean
from DREAMwalk.utils import read_graph
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
from itertools import combinations
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hierarchy_file', type=str, required=True)
    parser.add_argument('--network_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True,
                       help='similarity graph output file name')
    
    
    parser.add_argument('--cut_off', type=float, default=0.5)
    parser.add_argument('--weighted', type=bool, default=True)    
    parser.add_argument('--directed', type=bool, default=False)
    parser.add_argument('--net_delimiter',type=str,default='\t',
                       help='delimiter of networks file; default = tab')
    
    args=parser.parse_args()
    args={'networkf':args.network_file,
     'hierf':args.hierarchy_file,
     'outputf':args.output_file,
     'cutoff':args.cut_off,
     'weighted':args.weighted,
     'directed':args.directed,
     'net_delimiter':args.net_delimiter}
    
    return args

def generate_sim_data(DF, PF, nodes,cutoff,directed):
    DFsim = pairwise(DF)
    # print(DFsim.shape)
    PFsim = pairwise(PF)
    # print(PFsim.shape)
    DFsim = np.insert(DFsim.values, 2, 1, axis=1)
    PFsim = np.insert(PFsim.values, 2, 2, axis=1)
    concatenated_df = np.vstack((DFsim, PFsim))
    num_rows = concatenated_df.shape[0]
    index_column = np.arange(num_rows).reshape((num_rows, 1))
    concatenated_df = np.concatenate((concatenated_df, index_column), axis=1)
    # print(concatenated_df.shape)
    # print(concatenated_df)
    return concatenated_df

def EuclideanDistance(df):
    features = df.iloc[:, 1:].values
    # sample_combinations = list(combinations(df.iloc[:, 0], 2))
    distances = euclidean_distances(features, features)
    np.fill_diagonal(distances, np.nan)
    
    row, col = np.where(~np.isnan(distances))
    # distance_results = pd.DataFrame({'Sample1': [df.iloc[i, 0] for i in row], 
    #                               'Sample2': [df.iloc[j, 0] for j in col], 
    #                               'EuclideanDistance': distances[row, col]})
    
    distance_results = pd.DataFrame({'Sample1': df.index[row], 
                                  'Sample2': df.index[col], 
                                  'EuclideanDistance': distances[row, col]})
    return distance_results 

def cosinesim(df):
    features = df.iloc[:, 1:].values
    cosine_similarities = cosine_similarity(features)
    np.fill_diagonal(cosine_similarities, np.nan)
    row, col = np.where(~np.isnan(cosine_similarities))
    distance_results = pd.DataFrame({'Sample1': df.index[row], 
                                    'Sample2': df.index[col], 
                                    'EuclideanDistance': cosine_similarities[row, col]})
    return distance_results

def pairwise(df):
    features = df.iloc[:, 1:].values
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(features)

    distances = pairwise_distances(data_standardized, metric='correlation')
    correlation_matrix = 1 - distances
    np.fill_diagonal(correlation_matrix, np.nan)
    correlation_matrix[correlation_matrix < 0.6] = np.nan
    row, col = np.where(~np.isnan(correlation_matrix))
    # distance_results = pd.DataFrame({'Sample1': df.index[row], 
    #                                 'Sample2': df.index[col], 
    #                                 'EuclideanDistance': correlation_matrix[row, col]})
    distance_results = pd.DataFrame({'Sample1': [df.iloc[i, 0] for i in row], 
                                  'Sample2': [df.iloc[j, 0] for j in col], 
                                  'EuclideanDistance': correlation_matrix[row, col]})
    return distance_results
    


def dataset(networkf:str, DF_file:str, PF_file:str, outputf:str,
         cutoff:float, weighted:bool=True, directed:bool=False, net_delimiter:str='\t'):
    G=read_graph(networkf,weighted=weighted,directed=directed,
             delimiter=net_delimiter)
    nodes=list(G.nodes())
    DF=pd.read_csv(DF_file, header=None)
    PF=pd.read_csv(PF_file, header=None)
    sim_values=generate_sim_data(DF, PF, nodes,cutoff,directed)

    np.savetxt(outputf, sim_values, delimiter='\t', fmt='%s')
    print(f'Similarity graph saved: {outputf}')

if __name__ == '__main__':
    args=parse_args()
    save_sim_data(**args)