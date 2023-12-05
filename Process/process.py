import os

import numpy as np

from Process.dataset import GraphDataset, BiGraphDataset, UdGraphDataset, GraphDataset2seq

cwd=os.getcwd()
# cwd = os.path.dirname(os.path.dirname(os.getcwd()))


################################### load tree#####################################
def loadTree(dataname):
    if 'PHEME' in dataname:
        print("reading PHEME tree")
        Source_treePath = os.path.join(cwd, 'datasets', 'PHEME', 'Source_graph')
        Target_treePath = os.path.join(cwd, 'datasets', 'PHEME', 'Target_graph')
        treePath = [Source_treePath, Target_treePath]
        treeDic = {}
        for path in treePath:
            for event in os.listdir(path):
                if event[0] == ".":
                    continue
                post = np.load(os.path.join(path, event), allow_pickle=True)
                eid = event.split('.')[0]
                treeDic[eid] = {}
                edge_list = post['edgeindex'].tolist()
                root_index = int(post.f.rootindex)
                treeDic[eid][root_index] = {'parent': None}
                for node_id in range(len(edge_list[0])):
                    current_id = edge_list[1][node_id]
                    parent_id = edge_list[0][node_id]
                    treeDic[eid][current_id] = {'parent': parent_id}

    if 'Twitter' in dataname:
        print("reading Twitter tree")
        Source_treePath = os.path.join(cwd, 'datasets', 'Twitter', 'Source_graph')
        Target_treePath = os.path.join(cwd, 'datasets', 'Twitter', 'Target_graph')
        treePath = [Source_treePath, Target_treePath]
#         target_tree_path = [Target_treePath]
        treeDic = {}
        for path in treePath:
#             print(path)
            for event in os.listdir(path):
#                 print(event)
#                 print(os.path.join(path, event))
                if event[0] == ".":
                    continue
                post = np.load(os.path.join(path, event), allow_pickle=True)
                eid = event.split('.')[0]
                treeDic[eid] = {}
                edge_list = post['edgeindex'].tolist()
                root_index = int(post.f.rootindex)
                treeDic[eid][root_index] = {'parent': None}
                for node_id in range(len(edge_list[0])):
                    current_id = edge_list[1][node_id]
                    parent_id = edge_list[0][node_id]
                    treeDic[eid][current_id] = {'parent': parent_id}
    
    if 'Twitter15_16' in dataname:
        print("reading Twitter15&16 tree")
        Source_treePath = os.path.join(cwd, 'datasets', 'Twitter15_16', 'Source_graph')
        Target_treePath = os.path.join(cwd, 'datasets', 'Twitter15_16', 'Target_graph')
        treePath = [Source_treePath, Target_treePath]
#         target_tree_path = [Target_treePath]
        treeDic = {}
        for path in treePath:
#             print(path)
            for event in os.listdir(path):
#                 print(event)
#                 print(os.path.join(path, event))
                if event[0] == ".":
                    continue
                post = np.load(os.path.join(path, event), allow_pickle=True)
                eid = event.split('.')[0]
                treeDic[eid] = {}
                edge_list = post['edgeindex'].tolist()
                root_index = int(post.f.rootindex)
                treeDic[eid][root_index] = {'parent': None}
                for node_id in range(len(edge_list[0])):
                    current_id = edge_list[1][node_id]
                    parent_id = edge_list[0][node_id]
                    treeDic[eid][current_id] = {'parent': parent_id}
    
    if 'Weibo' in dataname:
        print("reading Weibo tree")
        Source_treePath = os.path.join(cwd, 'datasets', 'Weibo', 'Source_graph')
        Target_treePath = os.path.join(cwd, 'datasets', 'Weibo', 'Target_graph')
        treePath = [Source_treePath, Target_treePath]
#         target_tree_path = [Target_treePath]
        treeDic = {}
        for path in treePath:
#             print(path)
            for event in os.listdir(path):
#                 print(event)
#                 print(os.path.join(path, event))
                if event[0] == ".":
                    continue
                post = np.load(os.path.join(path, event), allow_pickle=True)
                eid = event.split('.')[0]
                treeDic[eid] = {}
                edge_list = post['edgeindex'].tolist()
                root_index = int(post.f.rootindex)
                treeDic[eid][root_index] = {'parent': None}
                for node_id in range(len(edge_list[0])):
                    current_id = edge_list[1][node_id]
                    parent_id = edge_list[0][node_id]
                    treeDic[eid][current_id] = {'parent': parent_id}
    

    if 'Twitter_origin' in dataname:
        treePath = os.path.join(cwd, 'data/' + dataname + '/Twitter' + '/Twitter_data_all.txt')
        print("reading twitter tree")
        treeDic = {}
        for line in open(treePath):
            line = line.strip('\n')
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            time_delay, text = float(line.split('\t')[3]), str(line.split('\t')[4])
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'time_delay': time_delay, 'text': text}
        print('Twitter tree no:', len(treeDic))

        tree_path_weibo = os.path.join(cwd, 'data/Weibo/Weibo/Weibo_data_all.txt')
        print("reading Weibo tree")
        for line in open(tree_path_weibo):
            line = line.strip('\n')
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            text = str(line.split('\t')[3])
            # time_delay, text = float(line.split('\t')[3]), str(line.split('\t')[4])
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'time_delay': time_delay, 'text': text}
        print('total tree no:', len(treeDic))

    if "Weibo_origin" in dataname:
        treePath = os.path.join(cwd, 'data/Weibo-COVID19/Weibo/weibo_covid19_data.txt')
        print("reading Weibo tree")
        treeDic = {}
        for line in open(treePath):
            line = line.strip('\n')
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            time_delay, text = float(line.split('\t')[3]), str(line.split('\t')[4])
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'time_delay': time_delay, 'text': text}
        print('weibo tree no:', len(treeDic))

        tree_path_twitter = os.path.join(cwd, 'data/Twitter/Twitter/Twitter_data_all.txt')
        print("reading twitter tree")
        for line in open(tree_path_twitter):
            line = line.strip('\n')
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            time_delay, text = float(line.split('\t')[3]), str(line.split('\t')[4])
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'time_delay': time_delay, 'text': text}
        print('total tree no:', len(treeDic))

    return treeDic


################################# load data ###################################

def loadseqData(dataname, treeDic, fold_x_train, fold_x_test, droprate):
    data_path = os.path.join(cwd, 'data', dataname + dataname.split('-')[0] + 'graph')
    print("loading train set", )
    traindata_list = GraphDataset2seq(fold_x_train, treeDic, droprate=droprate, data_path=data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = GraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadData(dataname, treeDic, fold_x_train, fold_x_test, droprate):
    data_path = os.path.join(cwd, 'data', dataname + dataname.split('-')[0] + 'graph')
    print("loading train set", )
    traindata_list = GraphDataset(fold_x_train, treeDic, droprate=droprate, data_path=data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = GraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list


def loadUdData(dataname, treeDic, fold_x_train, fold_x_test, droprate):
    data_path = os.path.join(cwd, 'data', dataname + 'graph')
    print("loading train set", )
    traindata_list = UdGraphDataset(fold_x_train, treeDic, droprate=droprate, data_path=data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = UdGraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list


def loadBiData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate, BUdroprate):
    data_path = os.path.join(cwd, 'data', dataname, dataname.split('-')[0] + 'graph')
    print("loading train set", )
    traindata_list = BiGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate,
                                    data_path=data_path)
    print("train no:", len(traindata_list))
    if len(fold_x_test) > 0:
        print("loading test set", )
        testdata_list = BiGraphDataset(fold_x_test, treeDic, data_path=data_path)
        print("test no:", len(testdata_list))
        return traindata_list, testdata_list
    # twitter_path = os.path.join(cwd, 'data/Twittergraph')
    # print('loading twitter set')
    # twitterdata_list = BiGraphDataset(twitter_train)
    else:
        return traindata_list


def loadBiData_PHEME(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate, BUdroprate, data_class):
    if dataname == 'SCL_PHEME':
        dataname = 'PHEME'

    if dataname == 'SCL_Twitter':
        dataname = 'Twitter'
        
    if dataname == 'SCL_Twitter15_16':
        dataname = 'Twitter15_16'
        
    if dataname == 'SCL_Weibo':
        dataname = 'Weibo'

    Source_data_path = os.path.join(cwd, 'datasets', dataname, data_class + '_graph')
    print("loading {} PHEME train set".format(data_class))
    traindata_list = BiGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate,
                                    data_path=Source_data_path)
    print("train no:", len(traindata_list))
    if len(fold_x_test) > 0:
        print("loading test set", )
        testdata_list = BiGraphDataset(fold_x_test, treeDic, data_path=Source_data_path)
        print("test no:", len(testdata_list))
        return traindata_list, testdata_list
    # twitter_path = os.path.join(cwd, 'data/Twittergraph')
    # print('loading twitter set')
    # twitterdata_list = BiGraphDataset(twitter_train)
    else:
        return traindata_list