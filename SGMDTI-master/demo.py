import os
from model.SGM import SGM
from model.train import train
from model.dataset import dataset
from model.feature import feature

if __name__ == '__main__':


    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    networkf='hetero_A/graph.txt'
    smiles_file = './hetero_A/data/drug_smiles.csv'
    pro_file = './hetero_A/data/drug_smiles.csv'
    DF_file = 'hetero_A/DF.csv'
    PF_file = 'hetero_A/PF.csv'
    simf='hetero_A/data_similarity_graph.txt'
    cutoff=0.8
    tp_f = 0.5
    nodetypef='hetero_A/nodetype.tsv'
    embeddingf='hetero_A/embedding/'
    pairf='hetero_A/DTI.tsv'
    modelf='hetero_A/clf.pkl'
    train_dataset = 'hetero_A/10_cv/'
    feature(smiles_file= smiles_file, pro_file = pro_file, DF_file, PF_file=PF_file)
    dataset(networkf=networkf,DF_file=DF_file,PF_file=PF_file,outputf=simf,cutoff=cutoff)
    SGM(netf=networkf,sim_netf=simf, outputf=embeddingf,nodetypef=nodetypef,tp_factor=tp_f)
    train(embeddingf=embeddingf, pairf=pairf, modelf=modelf, DF_file=DF_file,PF_file=PF_file, train_data=train_dataset)
    


    


