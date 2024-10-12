import argparse
import pandas as pd
import os
import math
import random
import pickle
import networkx as nx
import numpy as np   
from scipy import stats
from tqdm import tqdm
import parmap
from collections import defaultdict, Counter
from time import time
from HSG import HSG, read_graph, set_seed
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_file', type=str, required=True)
    
    parser.add_argument('--sim_network_file', type=str, default='')
    parser.add_argument('--node_type_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default='embedding_file.pkl')
    parser.add_argument('--seed', type=float, default=42)
    parser.add_argument('--tp_factor', type=float, default=0.5)
    parser.add_argument('--weighted', type=bool, default=True)    
    parser.add_argument('--directed', type=bool, default=False)
    parser.add_argument('--num_walks', type=int, default=10)
    parser.add_argument('--walk_length', type=int, default=10)
    parser.add_argument('--dimension', type=int, default=128)    
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                       help='if default, set to all available cpu count')
    parser.add_argument('--p', type=float, default=1)
    parser.add_argument('--q', type=float, default=1)
    parser.add_argument('--em_max_iter',type=int,default=5,
                   help='maximum EM iteration for edge type transition matrix training')
    parser.add_argument('--net_delimiter',type=str,default='\t',
                       help='delimiter of networks file; default = tab')
    parser.add_argument('--train_data', type=str, default='train.pkl')

    args=parser.parse_args()
    args={
        'netf':args.network_file,
        'sim_netf':args.sim_network_file,
        'outputf':args.output_file,
        'nodetypef':args.node_type_file,
        'tp_factor':args.tp_factor,
        'seed':args.seed,
        'weighted':args.weighted,
        'directed':args.directed,
        'num_walks':args.num_walks,
        'walk_length':args.walk_length,
        'dimension':args.dimension,
        'window_size':args.window_size,
        'workers':args.workers,
        'em_max_iter':args.em_max_iter,
        'p':args.p,
        'q':args.q,
        'net_delimiter':args.net_delimiter,
        'train_data':args.train_data
    }
    return args

# Train edge type transition matrix
def train_edgetype_transition_matrix(em_max_iter,G,networkf,net_delimiter,walk_length,p,q):
    matrix_conv_rate=0.01
    matrices={0:_init_edge_transition_matrix(networkf,net_delimiter)}
    for i in range(em_max_iter): # EM iteration
        walks=_sample_edge_paths(G,matrices[i],walk_length,p,q)   # M step
        matrices[i+1] = _update_trans_matrix(walks, matrices[i])  # E step
        matrix_diff = np.nan_to_num(np.absolute((matrices[i+1]-matrices[i])/matrices[i])).mean()
        if matrix_diff < matrix_conv_rate: 
            break
    return matrices[i+1]

def _sample_edge_paths(G,trans_matrix,walk_length,p,q):
    edges=list(G.edges(data=True))
    sampled_edges=random.sample(edges,int(len(edges)*0.01))  # sample 1% of edges from the original network
    edge_walks=[]
    for edge in sampled_edges:
        edge_walks.append(_edge_transition_walk(edge, G, trans_matrix, walk_length, p, q))
    return edge_walks

def _edge_transition_walk(edge, G, matrix, walk_length, p, q): 
    edge=(edge[0],edge[1],edge[2]['type'])
    walk=[edge]
    edge_path = [edge[2]]

    while len(walk) < walk_length:
        cur_edge = walk[-1]
        prev_node = cur_edge[0]
        cur_node = cur_edge[1]
        cur_edge_type = cur_edge[2]

        nbrs = G.neighbors(cur_node)
        # calculate edge weights for all neighbors
        nbrs_list=[]
        weights_list=[]
        for nbr in nbrs:
            for nbr_edge in G[cur_node][nbr].values():
                nbr_edge_type = nbr_edge['type']
                nbr_edge_weight = nbr_edge['weight']
                trans_weight = matrix[cur_edge_type-1][nbr_edge_type-1]
                if G.has_edge(nbr,prev_node) or G.has_edge(prev_node,nbr): 
                    nbrs_list.append((nbr,nbr_edge_type))
                    weights_list.append(trans_weight*nbr_edge_weight)
                elif nbr == prev_node: # p: Return parameter
                    nbrs_list.append((nbr,nbr_edge_type))
                    weights_list.append(trans_weight*nbr_edge_weight/p)
                else: # q: In-out parameter q
                    nbrs_list.append((nbr,nbr_edge_type))
                    weights_list.append(trans_weight*nbr_edge_weight/q)
        if sum(weights_list)>0: # the direction_node has next_link_end_node
            if not nbrs_list:
                continue
            next_edge=random.choices(nbrs_list,weights=weights_list)[0]
            next_edge=(cur_node,next_edge[0],next_edge[1])
            walk.append(next_edge)
            edge_path.append(next_edge[2])
        else:
            break
    return edge_path

def _init_edge_transition_matrix(networkf,net_delimiter):
    edgetypes=set()
    with open(networkf,'r') as fin:
        lines=fin.readlines()
    for line in lines:
        edgetypes.add(line.split(net_delimiter)[2])

    type_count=len(edgetypes)    # Number of edge types
    print('type_count', type_count)
    matrix=np.ones((type_count+1,type_count+1))
    matrix=matrix/matrix.sum()
    return matrix

def _update_trans_matrix(walks, matrix):
    type_count=len(matrix)
    matrix=np.zeros(matrix.shape)
    repo = defaultdict(list)

    for walk in walks:
        walk=[i-1 for i in walk]
        edge_count=Counter(walk)
    #         if edge_id in curr
        for i in range(type_count):
            repo[i].append(edge_count[i])
    for i in range(type_count):
        for j in range(type_count):
            sim_score = pearsonr_test(repo[i],repo[j])  
            matrix[i][j] = sim_score
    return np.nan_to_num(matrix)

def pearsonr_test(v1,v2): #original metric: the larger the more similar
    result = stats.mstats.pearsonr(v1,v2)[0]
    return sigmoid(result)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

#Teleport guided random walk
def generate_DREAMwalk_paths(G, G_sim, trans_matrix, p,q,num_walks, walk_length,tp_factor, workers, m):
    tot_walks = []
    nodes = list(G.nodes())

#     print(f'# of walkers : {min(num_walks,workers)} for {num_walks} times')
    walks = parmap.map(_parmap_walks, range(num_walks), 
                        nodes, G, G_sim, trans_matrix,p,q, walk_length,tp_factor, m,
                        pm_pbar=False, pm_processes=min(num_walks,workers))
    for walk in walks:
        tot_walks += walk

    return tot_walks    
    
# parallel walks
def _parmap_walks(_, nodes,G,G_sim,trans_matrix,p,q,walk_length,tp_factor,m):
    walks = []
    random.shuffle(nodes) 
    for node in nodes:
        walks.append(_DREAMwalker(node,G,G_sim,trans_matrix,p,q,walk_length,tp_factor,m))
    # print(walks)

    # new_list = list(filter(lambda x: x is not None, walks))
    # walks = new_list
    # print(len(walks))
    # if None in walks:
    #     print("walks has None")
    # output_file = 'walks.txt'
    # with open(output_file, 'w') as f:
    #     for walk in walks:
    #         f.write(walk + '\n')
    return walks

def _DREAMwalker(start_node, G, G_sim,trans_matrix,p,q, walk_length,tp_factor,m):
    walk = [start_node]
    edge_walk = []
    cur_metapath_idx = 0
    # select first edge from any neighbors
    nbrs_list=[]
    weights_list=[]
    for nbr in sorted(G.neighbors(start_node)):    
        for cur_edge in G[start_node][nbr].values():
            # print(cur_edge['type'])
            if cur_edge['type'] is not int(m[cur_metapath_idx]):
                continue
            nbrs_list.append((nbr,cur_edge['type']))
            weights_list.append(cur_edge['weight'])
    if not nbrs_list:
        F = []
        F.append(start_node)
        return F
        # return
    next_edge=random.choices(nbrs_list,
                                  weights=weights_list)[0]
    walk.append(next_edge[0])
    edge_walk.append(next_edge[1])
    
    cur_metapath_idx += 1
    while len(walk) < walk_length:
        prev=walk[-2]
        cur=walk[-1] 
        cur_edge_type=edge_walk[-1]

        cur_nbrs = sorted(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            # perform DREAMwalk path generation
            if (cur in G_sim.nodes()) & (np.random.rand() < tp_factor):
                next_node=_teleport_operation(cur,G_sim)               
            else:
                prev=(prev,cur_edge_type)
                cur_type = m[cur_metapath_idx % len(m)]
                next_node,next_edge_type=_network_traverse(cur,prev,G,trans_matrix,p,q,cur_type)
                edge_walk.append(next_edge_type)
            if next_node is not None:
                walk.append(next_node)
        else: # dead end
            break #if start node has 0 neighbour : dead end
        cur_metapath_idx += 1
    # if not walk:
    #     print('walk is None')
    #     return
    return walk

def _network_traverse(cur,prev,G,trans_matrix,p,q,cur_type):
    prev_node=prev[0]
    cur_edge_type=prev[1]
    cur_nbrs = sorted(G.neighbors(cur))
    nbrs_list=[]
    weights_list=[]
    
    # search for reachable edges and their weights
    for nbr in cur_nbrs:
        nbr_edges = G[cur][nbr]
        for nbr_edge in nbr_edges.values():
            nbr_edge_type = nbr_edge['type']
            if nbr_edge_type is not int(cur_type):
                continue
            nbr_edge_weight = nbr_edge['weight']
            if cur_edge_type is None:
                continue
            # print('cur_edge_type:', cur_edge_type)
            # print('nbr_edge_type:', nbr_edge_type)
            trans_weight=trans_matrix[cur_edge_type-1][nbr_edge_type-1]
            if G.has_edge(nbr,prev_node) or G.has_edge(prev_node,nbr): 
                nbrs_list.append((nbr,nbr_edge_type))
                weights_list.append(trans_weight*nbr_edge_weight)
            elif nbr == prev_node: # p: Return parameter
                nbrs_list.append((nbr,nbr_edge_type))
                weights_list.append(trans_weight*nbr_edge_weight/p)
            else: # q: In-Out parameter
                nbrs_list.append((nbr,nbr_edge_type))
                weights_list.append(trans_weight*nbr_edge_weight/q)
                
    # sample next node and edge type from searched weights
    if not nbrs_list:
        return None, None
    next_edge=random.choices(nbrs_list,weights=weights_list)[0]
    # print('next_edge:', next_edge)
    # if next_edge is None:
    #     print('next_edge is None')

    return next_edge    

def _teleport_operation(cur,G_sim):
    cur_nbrs = sorted(G_sim.neighbors(cur))
    random.shuffle(cur_nbrs)
    selected_nbrs=[]
    distance_sum=0
    for nbr in cur_nbrs:
        nbr_links = G_sim[cur][nbr]
        for i in nbr_links:
            nbr_link_weight = nbr_links[i]['weight']
            distance_sum += nbr_link_weight
            selected_nbrs.append(nbr)
    if distance_sum==0:
        return False
    rand = np.random.rand() * distance_sum
    threshold = 0

    for nbr in set(selected_nbrs):
        nbr_links = G_sim[cur][nbr]
        for i in nbr_links:
            nbr_link_weight = nbr_links[i]['weight']
            threshold += nbr_link_weight
            if threshold >= rand:
                next = nbr
                break
    return next

def SGM(netf:str, sim_netf:str, outputf:str, train_data:str, nodetypef:str=None, 
                         tp_factor:float=0.5, seed:int=42,
                         directed:bool=False, weighted:bool=True, em_max_iter:int=5,
                         num_walks:int=10, walk_length:int=10, workers:int=os.cpu_count(), 
                         dimension:int=128, window_size:int=4, p:float=1, q:float=1, 
                         net_delimiter:str='\t'):
    set_seed(int(time.time()))
    print('Reading network files...')
    
    if sim_netf:
        G_sim=read_graph(sim_netf,
                        weighted=True,
                        directed=False)
    else:
        G_sim=nx.empty_graph()
        print('> No input similarity file detected!')
        tp_factor=0

    metapath_edge = ['6','1', '3', '2', '5', '6', '4']
    metapath = ['DTD','DD', 'DSD', 'DID', 'TT', 'TDT', 'TIT']
    for i in range(10):
        G=read_graph(netf,weighted=weighted,directed=directed,
                 delimiter=net_delimiter)
        df = pd.read_csv(train_data +'train_fold_'+ str(i) +'.csv')
        train = df[df['2'] == 1][['0', '1']]
        for edge in train.values:
            G.add_edge(edge[0], edge[1], type=int(6), weight=float(1.0))
        print('Training edge type transition matrix...')
        trans_matrix=train_edgetype_transition_matrix(em_max_iter,G,
                                    netf,net_delimiter,walk_length,p,q) 
        for m in range(len(metapath_edge)):
            start_time = time.time()
            print('Generating paths...')
            walks=generate_DREAMwalk_paths(G,G_sim,trans_matrix,p,q,num_walks,
                                        walk_length,tp_factor, workers, metapath_edge[m])
            # print(walks)
            X = pd.DataFrame(walks)
            csv_file = 'walks.csv'
            X.to_csv(csv_file, index=False)
            print('Generating node embeddings...')
            use_hetSG = True if nodetypef != None else False
            embeddings = HeterogeneousSG(use_hetSG, walks, set(G.nodes()), nodetypef=nodetypef,
                                        embedding_size=dimension, window_length=window_size, workers=workers)

            nodetp = pd.read_csv(nodetypef, sep='\t')
            node = nodetp["node"]
            for j in node:
                if j in embeddings:
                    continue
                else:
                    embeddings[j] = np.zeros(128)

            output_template = '{}embedding_file{}.pkl'
            with open(os.path.join(outputf, output_template.format(metapath[m],str(i+1))),'wb') as fw:
                pickle.dump(embeddings,fw)
            end_time = time.time()
            print("read_data time: ", end_time - start_time, "s")   
            print(f'Node embeddings saved: {outputf}')

if __name__ == '__main__':
    args=parse_args()
    save_embedding_files(**args)
