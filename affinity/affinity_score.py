import os
import random
import sys
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import torch.nn.functional as F
import argparse
import csv
from affinity.metrics import get_cindex, get_rm2
from affinity.model import ColdDTA
from affinity.utils import *
#from log.train_logger import TrainLogger
from torch_geometric.data import InMemoryDataset
from rdkit import Chem, RDConfig
import networkx as nx
import pandas as pd
from torch_geometric import data as DATA

VOCAB_PROTEIN = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                 "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                 "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                 "U": 19, "T": 20, "W": 21,
                 "V": 22, "Y": 23, "X": 24,
                 "Z": 25}


def seqs2int(target):
    return [VOCAB_PROTEIN[s] for s in target.upper()]


def atom_features(atom):
    encoding = one_of_k_encoding_unk(atom.GetSymbol(),
                                     ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                      'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
                                      'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg',
                                      'Pb', 'Unknown'])
    encoding += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + one_of_k_encoding_unk(
        atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    encoding += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    encoding += one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, 'other'])
    encoding += [atom.GetIsAromatic()]

    try:
        encoding += one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except:
        encoding += [0, 0] + [atom.HasProp('_ChiralityPossible')]

    return np.array(encoding)


def remove_subgraph(Graph, center, percent):
    assert percent <= 1
    G = Graph.copy()
    num = int(np.floor(len(G.nodes()) * percent))
    removed = []
    temp = [center]
    while len(removed) < num and temp:
        neighbors = []

        try:
            for n in temp:
                neighbors.extend([i for i in G.neighbors(n) if i not in temp])
        except Exception as e:
            print(e)
            return None, None

        for n in temp:
            if len(removed) < num:
                G.remove_node(n)
                removed.append(n)
            else:
                break

        temp = list(set(neighbors))

    return G, removed


# data augmentation
def mol_to_graph(mol, times=2):
    start_list = random.sample(list(range(mol.GetNumAtoms())), times)

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    molGraph = nx.Graph(edges)
    percent = 0.2
    removed_list = []
    for i in range(1, times + 1):
        G, removed = remove_subgraph(molGraph, start_list[i - 1], percent)
        removed_list.append(removed)

    for removed_i in removed_list:
        if not removed_i:
            return None, None, None

    features_list = []
    for i in range(times):
        features_list.append([])

    for index, atom in enumerate(mol.GetAtoms()):
        for i, removed_i in enumerate(removed_list):
            if index not in removed_i:
                feature = atom_features(atom)
                features_list[i].append(feature / np.sum(feature))

    edges_list = []
    for i in range(times):
        edges_list.append([])

    g = nx.DiGraph()
    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j,
                           b_type=e_ij.GetBondType(),
                           IsConjugated=int(e_ij.GetIsConjugated()),
                           )

    edge_attr_list = []
    for i in range(times):
        edge_attr_list.append([])
    for i, removed_i in enumerate(removed_list):
        e = {}
        for n1, n2, d in g.edges(data=True):

            if n1 not in removed_i and n2 not in removed_i:
                start_i = n1 - sum(num < n1 for num in removed_i)
                e_i = n2 - sum(num < n2 for num in removed_i)
                edges_list[i].append([start_i, e_i])
                e_t = [int(d['b_type'] == x)
                       for x in (Chem.rdchem.BondType.SINGLE, \
                                 Chem.rdchem.BondType.DOUBLE, \
                                 Chem.rdchem.BondType.TRIPLE, \
                                 Chem.rdchem.BondType.AROMATIC)]
                e_t.append(int(d['IsConjugated'] == False))
                e_t.append(int(d['IsConjugated'] == True))
                e[(n1, n2)] = e_t
        edge_attr = list(e.values())
        edge_attr_list[i] = edge_attr

    if len(e) == 0:
        return features_list, torch.LongTensor([[0], [0]]), torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

    edge_index_list = []
    for i in range(times):
        edge_index_list.append([])
    for i, edges_i in enumerate(edges_list):
        g_i = nx.Graph(edges_i).to_directed()
        for e1, e2 in g_i.edges():
            edge_index_list[i].append([e1, e2])

    return features_list, edge_index_list, edge_attr_list


def mol_to_graph_without_rm(mol):
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / np.sum(feature))

    g = nx.DiGraph()
    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j,
                           b_type=e_ij.GetBondType(),
                           IsConjugated=int(e_ij.GetIsConjugated()),
                           )
    e = {}
    for n1, n2, d in g.edges(data=True):
        e_t = [int(d['b_type'] == x)
               for x in (Chem.rdchem.BondType.SINGLE, \
                         Chem.rdchem.BondType.DOUBLE, \
                         Chem.rdchem.BondType.TRIPLE, \
                         Chem.rdchem.BondType.AROMATIC)]
        e_t.append(int(d['IsConjugated'] == False))
        e_t.append(int(d['IsConjugated'] == True))
        e[(n1, n2)] = e_t

    if len(e) == 0:
        return features, torch.LongTensor([[0], [0]]), torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

    edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
    edge_attr = torch.FloatTensor(list(e.values()))

    return features, edge_index, edge_attr


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


class GNNDataset(InMemoryDataset):
    def __init__(self, root, index=0, types='test', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        if types == 'train':
            self.data, self.slices = torch.load(self.processed_paths[index])
        elif types == 'test':
            self.data, self.slices = torch.load(self.processed_paths[index + len(self.raw_paths)])

    @property
    def raw_file_names(self):
        #return ['davis_fold_0.csv', 'davis_fold_1.csv', 'davis_fold_2.csv', 'davis_fold_3.csv', 'davis_fold_4.csv']
        return  ["gen_smi.csv"]


    @property
    def processed_file_names(self):
        # return ['processed_data_fold_0.pt', 'processed_data_fold_1.pt', 'processed_data_fold_2.pt',
        #         'processed_data_fold_3.pt', 'processed_data_fold_4.pt',
        #         'processed_test_data_fold_0.pt', 'processed_test_data_fold_1.pt', 'processed_test_data_fold_2.pt',
        #         'processed_test_data_fold_3.pt', 'processed_test_data_fold_4.pt']
        return ['processed_data_fold_0.pt', 'processed_test_data_fold_0.pt']

    def download(self):
        pass

    def process(self):
        fold_0_list = self.process_train_data(self.raw_paths[0])
        # fold_1_list = self.process_train_data(self.raw_paths[1])
        # fold_2_list = self.process_train_data(self.raw_paths[2])
        # fold_3_list = self.process_train_data(self.raw_paths[3])
        # fold_4_list = self.process_train_data(self.raw_paths[4])

        fold_0_test_list = self.process_data(self.raw_paths[0])
        # fold_1_test_list = self.process_data(self.raw_paths[1])
        # fold_2_test_list = self.process_data(self.raw_paths[2])
        # fold_3_test_list = self.process_data(self.raw_paths[3])
        # fold_4_test_list = self.process_data(self.raw_paths[4])

        print('Graph construction done. Saving to file.')

        # save preprocessed train data:
        data, slices = self.collate(fold_0_list)
        torch.save((data, slices), self.processed_paths[0])
        # data, slices = self.collate(fold_1_list)
        # torch.save((data, slices), self.processed_paths[1])
        # data, slices = self.collate(fold_2_list)
        # torch.save((data, slices), self.processed_paths[2])
        # data, slices = self.collate(fold_3_list)
        # torch.save((data, slices), self.processed_paths[3])
        # data, slices = self.collate(fold_4_list)
        # torch.save((data, slices), self.processed_paths[4])

        # save preprocessed test data:
        data, slices = self.collate(fold_0_test_list)
        torch.save((data, slices), self.processed_paths[1])
        # data, slices = self.collate(fold_1_test_list)
        # torch.save((data, slices), self.processed_paths[6])
        # data, slices = self.collate(fold_2_test_list)
        # torch.save((data, slices), self.processed_paths[7])
        # data, slices = self.collate(fold_3_test_list)
        # torch.save((data, slices), self.processed_paths[8])
        # data, slices = self.collate(fold_4_test_list)
        # torch.save((data, slices), self.processed_paths[9])

    def process_data(self, data_path):
        df = pd.read_csv(data_path)
        data_list = []
        delete_list = []
        for i, row in df.iterrows():
            smi = row['SMILES']
            sequence = row['target_sequence'].upper()
            #affinity = row['affinity']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue
            x, edge_index, edge_attr = mol_to_graph_without_rm(mol)

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]

            data = DATA.Data(
                x=torch.FloatTensor(x),
                smi=smi,
                edge_index=edge_index,
                edge_attr=edge_attr,
                #y=torch.FloatTensor([affinity]),
                target=torch.LongTensor([target])
            )
            data_list.append(data)

        if len(delete_list) > 0:
            df = df.drop(delete_list, axis=0, inplace=False)
            df.to_csv(data_path, index=False)
        print('mol_to_graph_without_rm')
        return data_list

    def process_train_data(self, data_path):
        df = pd.read_csv(data_path)
        data_list = []
        delete_list = []
        for i, row in df.iterrows():
            smi = row['SMILES']
            sequence = row['target_sequence'].upper()
            #affinity = row['affinity']
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print("Unable to process: ", smi)
                continue

            x_list, edge_index_list, edge_attr_list = mol_to_graph(mol, times=2)

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]

            if x_list and edge_index_list:
                for x, edge_index, edge_attr in zip(x_list, edge_index_list, edge_attr_list):
                    try:
                        data = DATA.Data(
                            x=torch.FloatTensor(x),
                            smi=smi,
                            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                            edge_attr=torch.FloatTensor(edge_attr),
                            #y=torch.FloatTensor([affinity]),
                            target=torch.LongTensor([target])
                        )
                        data_list.append(data)
                    except Exception as e:
                        print(e)
                        continue

        print('process train data')
        if len(delete_list) > 0:
            df = df.drop(delete_list, axis=0, inplace=False)
            df.to_csv(data_path, index=False)

        return data_list


def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            label = data.y
            running_loss.update(loss.item(), label.size(0))

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss

def test_result(model, criterion, test_loader, device):
    criterion = nn.MSELoss()
    pred = test_val(model, criterion, test_loader, device)
    return pred


def test_val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            #loss = criterion(pred.view(-1), data.y.view(-1))
            #label = data.y
            pred_list.append(pred.view(-1).detach().cpu().numpy())
            #label_list.append(label.detach().cpu().numpy())
            #running_loss.update(loss.item(), label.size(0))

    pred = np.concatenate(pred_list, axis=0)
    # print(pred)
   # label = np.concatenate(label_list, axis=0)
    # print(label)
    # for i in range(len(pred)):
    #     print("data_{}     pred:{} ".format(i, pred[i]))

    # epoch_cindex = get_cindex(label, pred)
    # epoch_r2 = get_rm2(label, pred)
    # epoch_loss = running_loss.get_average()
    # running_loss.reset()

    return pred


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data', help='task name')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    args = parser.parse_args()
    params = dict(
        data_root="hybrid",
        save_dir="save",
        dataset=args.dataset,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )

    # logger = TrainLogger(params)
    # logger.info(__file__)

    DATASET = params.get("dataset")
    save_model = params.get("save_model")
    data_root = params.get("data_root")
    batch_size = params.get("batch_size")
    fpath = os.path.join(data_root, DATASET)

    test_set_list = []
    set_list = []
    for index in range(1):
        test_set_list.append(GNNDataset(fpath, types='test', index=index))
        set_list.append(GNNDataset(fpath, types='train', index=index))

    num_workers = 8 if sys.platform == 'linux' else 0
    print('Number of workers: ', num_workers)


    device = torch.device('cuda:0')
    model = ColdDTA()
    model.load_state_dict(torch.load("affinity/model_affinity/epoch728test_loss0.1186.pt"))

    model.to("cuda")
    model.eval()

    criterion = nn.MSELoss()
    test_loader = DataLoader(test_set_list[0], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             drop_last=True)
    pred = test_result(model, criterion, test_loader, device)
    return pred



def pred_affinity(list_smile,root,prodeins = "MPPYTVVYFPVRGRCAALRMLLADQGQSWKEEVVTVETWQEGSLKASCLYGQLPKFQDGDLTLYQSNTILRHLGRTLGLYGKDQQEAALVDMVNDGVEDLRCKYISLIYTNYEAGKDDYVKALPGQLKPFETLLSQNQGGKTFIVGDQISFADYNLLDLLLIHEVLAPGCLDAFPLLSAYVGRLSARPKLKAFLASPEYVNLPINGNGKQ" ):
    #MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNPGILPIQPALTPRKRATVQPPPQAAGSSNQPGLLASVPQPKPQAPPSQPLPQTQAKQPQAPPTPQQTPSTQAQGLPAQAQATPQHQQQLFLKQQQQQQQPPPAQQQPAGTFYQQQQAQTQQFQAVHPATQKPAIAQFPVVSQGGSQQQLMQNFYQQQQQQQQQQQQQQLATALHQQQLMTQQAALQQKPTMAAGQQPQPQPAAAPQPAPAQEPAIQAPVRQQPKVQTTPPPAVQGQKVGSLTPPSSPKTQRAGHRRILSDVTHSAVFGVPASKSTQLLQAAAAEASLNKSKSATTTPSGSPRTSQQNVYNPSEGSTWNPFDDDNFSKLTAEELLNKDFAKLGEGKHPEKLGGSAESLIPGFQSTQGDAFATTSFSAGTAEKRKGGQTVDSGLPLLSVSDPFIPLQVPDAPEKLIEGLKSPDTSLLLPDLLPMTDPFGSTSDAVIEKADVAVESLIPGLEPPVPQRLPSQTESVTSNRTDSLTGEDSLLDCSLLSNPTTDLLEEFAPTAISAPVHKAAEDSNLISGFDVPEGSDKVAEDEFDPIPVLITKNPQGGHSRNSSGSSESSLPNLARSLLLVDQLIDL
    # if sys.platform == 'linux':
    #     with open(__file__, "r", encoding="utf-8") as f:
    #         for line in f.readlines():
    #             print(line)

    list_gen_smiles = list_smile
    prodeins = prodeins
    list_prodeins = [prodeins] * len(list_gen_smiles)


    with open(root, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["index", "SMILES", "target_sequence"])

        for i in range(len(list_gen_smiles)):
            writer.writerow([i , list_gen_smiles[i], list_prodeins[i]])

    folder_path = 'hybrid/data/processed'
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print("delete successfully")


    GNNDataset('hybrid/data')

    score = main()
    return score

