import os
import numpy as np
import pickle
import os
import random
import numpy as np
import networkx as nx
import tensorflow as tf
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

def _prep_hetSG_walks(walks, node2id, nodetypef):
    with open(nodetypef,'r') as fr:
        lines=fr.readlines()
    node2type={}
    for line in lines[1:]:
        line=line.strip().split('\t')
        node2type[line[0]]=line[1]

    # we need to annotate the nodes with prefixes for the HetSG to recognize the nodetype
    type2meta = {
        'drug':'d',
        'disease':'d',
        'protein':'p',
        'se':'s'
            }

    annot_walks=[]
    for walk in walks:
        annot_walk=[]
        for node in walk:
            nodeid=node2id[node]
            try : node = type2meta[node2type[node]]+nodeid
            except KeyError : node='e'+nodeid
            annot_walk.append(node)
        annot_walks.append(' '.join(annot_walk))
    return annot_walks

def _prep_SG_walks(walks, node2id):
    # does not require node type prefixes
    annot_walks=[]
    for walk in walks:
        annot_walk=[node2id[node] for node in walk]
        annot_walks.append(' '.join(annot_walk))        
    return annot_walks

def HSG(use_hetSG:bool, walks:list, nodes:set, nodetypef:str, embedding_size:int, 
                   window_length:int, workers:int):
    use_hetSG=int(use_hetSG)
    
    node2id={node:str(i) for i, node in enumerate(nodes)}
    id2node={str(i):node for i, node in enumerate(nodes)}
    
    if use_hetSG:   
        annot_walks = _prep_hetSG_walks(walks,node2id, nodetypef)
    else:
        annot_walks = _prep_SG_walks(walks,node2id)
        
    tmp_walkf='tmp_walkfile'
    with open(tmp_walkf,'w') as fw:
        fw.write('\n'.join(annot_walks))

    outputf = 'tmp_outputf'
        
    # use hetsg.cpp file for embedding vector generation -> outputs .txt file
    os.system('g++ DREAMwalk/HeterogeneousSG.cpp -o HetSG -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result')
    os.system(f'./HetSG -train {tmp_walkf} -output {outputf} -pp {use_hetSG} -min-count 0 -size {embedding_size} -iter 1 -window {window_length} -threads {workers}')
    
    with open(outputf+'.txt','r') as fr:
        lines=fr.readlines()
        
    # remove tmp files
    os.system(f'rm -f {outputf} {tmp_walkf} {outputf}.txt HetSG')
    
    embeddings={}
    for line in lines[1:]:
        line=line.strip().split(' ')
        if line[0] == '</s>':
            pass
        else:
            nodeid = line[0]
            if use_hetSG:
                embedding=np.array([float(i) for i in line[1:]], dtype=np.float32)
                embeddings[id2node[nodeid[1:]]]=embedding
            else:
                embedding=np.array([float(i) for i in line[1:]], dtype=np.float32)
                embeddings[id2node[nodeid]]=embedding

    return embeddings
                       
def read_graph(edgeList,weighted=True, directed=False,delimiter='\t'):
    '''
    Reads the input network in networkx.
    '''
    if weighted:
        G = nx.read_edgelist(edgeList, nodetype=str, 
                             data=(('type',int),('weight',float),('id',int)), 
                             create_using=nx.MultiDiGraph(),
                            delimiter=delimiter)
    else:
        G = nx.read_edgelist(edgeList, nodetype=str,data=(('type',int)), 
                             create_using=nx.MultiDiGraph(),
                            delimiter=delimiter)
        for edge in G.edges():
            edge=G[edge[0]][edge[1]]
            for i in range(len(edge)):
                edge[i]['weight'] = 1.0
    if not directed:
        G = G.to_undirected()

    return G

def trans(data):
    # fingerprint=df.values
    A=AE(6,30,30,data)
    return A

def AE(layer, epochs, batch_size, mat_contents):
    encoding_dim = 64
    layer = int(layer)
    epochs = int(epochs)
    batch_size = int(batch_size)
    L = [int(len(mat_contents[0]) // (2**i)) for i in range(layer)]

    x_train_test = []
    for row in mat_contents:  # 接口
        a = row
        x_train_test.append(a)

    x_train_test = tf.convert_to_tensor(x_train_test)
    x_train = x_train_test
    # this is our input placeholder
    input_img = tf.keras.layers.Input(shape=(L[0],))

    # encoder layers
    encoded = input_img
    for i in range(layer-1):
        encoded = tf.keras.layers.Dense(L[i+1], activation='relu')(encoded)

    encoder_output = tf.keras.layers.Dense(encoding_dim)(encoded)

    # decoder layers
    decoded = encoder_output
    for i in range(layer-1):
        decoded = tf.keras.layers.Dense(L[layer-i-1], activation='relu')(decoded)

    decoded = tf.keras.layers.Dense(L[0], activation='tanh')(decoded)
    # construct the autoencoder model
    autoencoder = tf.keras.models.Model(input_img, decoded)
    encoder = tf.keras.models.Model(input_img, encoder_output)
    # compile autoencoder
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
    # training
    autoencoder.fit(x_train, x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    verbose=0)
    # plotting
    encoded_imgs = encoder.predict(x_train)
    # to xlsx
    return encoded_imgs
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f'random seed with {seed}')
