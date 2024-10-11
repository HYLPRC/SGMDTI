import argparse
import pickle
import numpy as np
from time import time
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from random import randint
import pandas as pd
from DREAMwalk.utils import set_seed, trans
from sklearn.model_selection import StratifiedKFold
import os
import warnings
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file', type=str, required=True)
    parser.add_argument('--pair_file', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--model_checkpoint', type=str, default='clf.pkl')
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    
    args = parser.parse_args()
    args = {'embeddingf':args.embedding_file,
     'pairf':args.pair_file,
     'seed':args.seed,
     'patience':args.patience,
     'modelf':args.model_checkpoint,
     'testr':args.test_ratio,
     'validr':args.validation_ratio
     }
    return args

def feature(Data, mir, dr, Fm, Fd):
    
    miR_1 = []
    dr_1 = []
    for i in mir:
        miR_1.append(i)
    for i in dr:
        dr_1.append(i)

    feature = []
    miRNA_id = miR_1
    drug_id = dr_1
    for i in Data:
        part=[]
        part.extend(Fm[miRNA_id.index(i[0])])
        part.extend(Fd[drug_id.index(i[1])])
        feature.append(part)
    feature = np.delete(feature, 0, 1)
    feature = np.delete(feature, 300, 1)
    feature = np.array(feature)
    feature = feature.astype(float)
    return feature

def split_dataset(embeddingf, seed, DF, PF, metapath, train, test, fold_i):
    
    embedding_f = '{}embedding_file.pkl'
    xs_train = None
    xs_test = None
    train = train.values
    test = test.values
    np.random.shuffle(train)
    np.random.shuffle(test)
    for m in metapath:
        with open(os.path.join(embeddingf, embedding_f.format(m)),'rb') as fin:
            embedding_dict = pickle.load(fin)
        x_train, train_label, x_test, test_label = [], [], [], []
        for i in train:
            drug = i[0]
            dis = i[1]
            label = i[2]
            x_train.append(embedding_dict[drug]-embedding_dict[dis])
            train_label.append(int(label))
        for i in test:
            drug = i[0]
            dis = i[1]
            label = i[2]
            x_test.append(embedding_dict[drug]-embedding_dict[dis])
            test_label.append(int(label))
        if xs_train is None:
            xs_train = np.array(x_train)
        else:
            xs_train = np.concatenate((xs_train, x_train), axis=1)
        if xs_test is None:
            xs_test = np.array(x_test)
        else:
            xs_test = np.concatenate((xs_test, x_test), axis=1)
    xs_train = trans(xs_train)
    xs_test = trans(xs_test)
    drug = DF.iloc[:, 0].values
    drug_feature = DF.values
    protein = PF.iloc[:, 0].values
    protein_feature = PF.values
    train_feature = feature(train[:, :2], drug, protein, drug_feature, protein_feature)
    test_feature = feature(test[:, :2], drug, protein, drug_feature, protein_feature)
    train_feature = np.concatenate((xs_train, train_feature), axis=1)
    test_feature = np.concatenate((xs_test, test_feature), axis=1)

    x,y = {},{}
    x['train'] = train_feature
    y['train'] = train_label
    x['test'] = test_feature
    y['test'] = test_label
    y['train']=[[i] for i in y['train']]
    x['train'],x['valid'],y['train'],y['valid'] = train_test_split(x['train'],y['train'],
                                                                    test_size = 0.05, random_state = 42,
                                                                    stratify = y['train'])
    return x, y


def return_scores(target_list, pred_list):
    metric_list = [
        accuracy_score, 
        roc_auc_score, 
        average_precision_score, 
        f1_score
    ] 
    
    scores = []
    for metric in metric_list:
        if metric in [roc_auc_score, average_precision_score]:
            scores.append(metric(target_list,pred_list))
        else: # accuracy_score, f1_score
            scores.append(metric(target_list, pred_list.round())) 
    return scores



def predict_dda(embeddingf:str, pairf:str, DF_file: str, PF_file: str,train_data:str, modelf:str='clf.pkl',  seed:int=18,
                validr:float=0.1, testr:float=0.1):
    DF=pd.read_csv(DF_file, header=None)
    PF=pd.read_csv(PF_file, header=None)
    set_seed(int(time()))
    for i in range(10):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        train = pd.read_csv(train_data +'train_fold_'+ str(i) +'.csv')
        test = pd.read_csv(train_data +'test_fold_'+ str(i) +'.csv')
        metapath = ['DD', 'DSD', 'DID', 'TT', 'TIT']
        x,y = split_dataset(embeddingf, seed, DF, PF, metapath, train, test, i)
        clf = XGBClassifier(base_score = 0.5, booster = 'gbtree',eval_metric ='error',objective = 'binary:logistic',
            gamma = 0,learning_rate = 0.1, max_depth = 10,n_estimators = 1000,
            tree_method = 'hist', gpu_id=2, min_child_weight = 4,subsample = 0.8, colsample_bytree = 0.9,
            scale_pos_weight = 1,max_delta_step = 1,seed = seed)

        clf.fit(x['train'], y['train'])
        preds = {}
        scores = {}
        for split in ['train','valid','test']:
            preds[split] = clf.predict_proba(np.array(x[split]))[:, 1]
            scores[split] = return_scores(y[split], preds[split])
            print(f'{split.upper():5} set | Acc: {scores[split][0]*100:.2f}% | AUROC: {scores[split][1]:.4f} | AUPR: {scores[split][2]:.4f} | F1-score: {scores[split][3]:.4f}')
        
        with open(modelf,'wb') as fw:
            pickle.dump(clf, fw)
        print(f'saved XGBoost classifier: {modelf}')
        print('='*50)
          
if __name__ == '__main__':
    args=parse_args()
    predict_dda(**args)