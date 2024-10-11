import dgl
import errno
import numpy as np
import torch
import csv
import os
import pandas as pd
import rdkit
import pandas as pd
import numpy as np
from PyBioMed import Pyprotein
from PyBioMed.PyProtein import CTD
from dgllife.utils import load_smiles_from_txt
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def mkdir_p(path, log=True):
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise

def feature_pro(seq, device):

    protein_descriptor=[]
    for i in range(len(seq)):
        ctd = CTD.CalculateCTD(seq[i]).values()
        protein_descriptor.append(list(ctd))
    return protein_descriptor
def collate(graphs):
    return dgl.batch(graphs)
def feature_drug(smiles, device):

    graphs = []
    success = []
    for smi in smiles:
        try:
            mol = rdkit.Chem.MolFromSmiles(smi)
            if mol is None:
                success.append(False)
                continue
            g = mol_to_bigraph(mol, add_self_loop=True,
                               node_featurizer=PretrainAtomFeaturizer(),
                               edge_featurizer=PretrainBondFeaturizer(),
                               canonical_atom_order=False)
            graphs.append(g)
            success.append(True)
        except:
            success.append(False)
    data_loader = torch.utils.data.DataLoader(graphs, batch_size=256,
                             collate_fn=collate, shuffle=False)
    model = load_pretrained('gin_supervised_contextpred').to(device)
    model.eval()
    readout = dgl.nn.pytorch.glob.AvgPooling()
    mol_emb = []
    for batch_id, bg in enumerate(data_loader):
        bg = bg.to(device)
        nfeats = [bg.ndata.pop('atomic_number').to(device),
                  bg.ndata.pop('chirality_type').to(device)]
        efeats = [bg.edata.pop('bond_type').to(device),
                  bg.edata.pop('bond_direction_type').to(device)]
        with torch.no_grad():
            node_repr = model(bg, nfeats, efeats)
        mol_emb.append(readout(bg, node_repr))
    mol_emb = torch.cat(mol_emb, dim=0).detach().cpu().numpy()
    
    return mol_emb

def feature(smiles_file:str, pro_file:str, DF_file:str, PF_file=PF_file:str):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    df = pd.read_csv(smiles_file)
    drug = df['drug']
    smiles = df['smiles'].tolist()
    pro = pd.read_csv(pro_file)
    protein = pro['pro']
    seq = pro['seq'].values.astype(str)
    DF = feature_drug(smiles, device)
    PF = feature_pro(seq, device)

    StorFile(DF,DF_file)
    StorFile(PF,PF_file)
    
if __name__ == '__main__':
    args=parse_args()
    feature(**args)