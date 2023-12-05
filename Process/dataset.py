import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data

max_len = 140


class GraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..', '..', 'data', 'Weibograph')):
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.droprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])]))


def collate_fn(data):
    return data


class GraphDataset2seq(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..', '..', 'data', 'Weibograph')):
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        edge_source_list = data['edgeindex'][0, :].tolist()
        edge_target_list = data['edgeindex'][1, :].tolist()
        seq_depth_list = {}
        a_matrix = np.zeros((data['edgeindex'].shape[1] + 1, data['edgeindex'].shape[1] + 1), dtype=int)
        a_matrix[edge_source_list, edge_target_list] = 1
        dep_node_dic = {}
        max_dep = 1
        for i in range(len(edgeindex.shape[1] + 1)):
            dep_node_dic[str(i + 1)] = []
            if i == 0:
                current_root_path = a_matrix[0, :]
                dep_node_dic[str(i + 1)].append(current_root_path.nonzero()[0].tolist())
            current_root_path = np.matmul(current_root_path, a_matrix)
            if current_root_path.nonzero()[0].shape[0] == 0:
                max_dep = i + 1
                break
            dep_node_dic[str(i + 1)].append(current_root_path.nonzero()[0].tolist())

        dic_key_list = list(range(1, max_dep+1))
        dep_node_path = {}
        total_list = []

        for i in range(len(dic_key_list)):
            current_dep = str(dic_key_list[max_dep-i-1])
            dep_node_path[current_dep] = {}
            for leaf_node_id in range(len(dep_node_dic[current_dep])):
                if i != 0 and dep_node_dic[current_dep][leaf_node_id] in total_list:
                    continue
                current_dep_int = int(current_dep)
                dep_node_path[current_dep][leaf_node_id] = []
                for j in range(current_dep_int):
                    if j == 0:
                        current_node_col = dep_node_dic[current_dep][leaf_node_id]
                    dep_node_path[current_dep][leaf_node_id].append(current_node_col)
                    if j != current_dep_int - 1:
                        current_node_row = a_matrix[:, current_node_col].nonzero()[0][0]
                        current_node_col = current_node_row
                    else:
                        total_list.append(dep_node_path[current_dep][leaf_node_id])
                        dep_node_path[current_dep][leaf_node_id].append(0)


        return dep_node_path
        # return Data(x=x_features,
        #             edge_index=torch.LongTensor(new_edgeindex), BU_edge_index=torch.LongTensor(new_edgeindex),
        #             y=torch.LongTensor([int(data['y'])]), root=torch.from_numpy(data['root']),
        #             rootindex=torch.LongTensor([int(data['rootindex'])]), seqlen=seqlen, eid=id)


class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, tddroprate=0, budroprate=0,
                 data_path=os.path.join('..', '..', 'data', 'Weibograph')):
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        seqlen = torch.LongTensor([(data['seqlen'])]).squeeze()

        # x_tokens_feat=list(data['x'])
        # x = np.zeros([len(x_tokens_feat), max_len, 768])
        # for item in range(len(x_tokens_feat)):
        #    for i in range(x_tokens_feat[item].size()[0]):
        #        x[item][i] = x_tokens_feat[item][i].detach().numpy()

        x_features = torch.tensor([item.detach().numpy() for item in list(data['x'])], dtype=torch.float32)

        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(1, length), int((length-1) * (1 - self.tddroprate)))
            poslist.append(0)
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
            # if len(poslist) == 0 or 0 not in poslist:
            #     new_edgeindex = edgeindex
            if len(poslist) == 1:
                new_edgeindex = edgeindex
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(1, length), int((length-1) * (1 - self.budroprate)))
            poslist.append(0)
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
            if len(poslist) == 1:
                bunew_edgeindex = edgeindex
        else:
            bunew_edgeindex = [burow, bucol]
        return Data(x=x_features,
                    edge_index=torch.LongTensor(new_edgeindex), BU_edge_index=torch.LongTensor(bunew_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=torch.from_numpy(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])]), seqlen=seqlen, eid=id)


class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..', '..', 'data', 'Weibograph')):
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]

        return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])]))
