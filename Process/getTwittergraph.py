# -*- coding: utf-8 -*-
import os
import gc
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import objgraph
import psutil
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
token_model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
# from transformers import RobertaTokenizer, RobertaModel
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# token_model = RobertaModel.from_pretrained('roberta-base')

# cwd=os.getcwd()
cwd = os.path.dirname(os.getcwd())

class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        # self.idx = idx
        # self.word = []
        self.inputs_features=[]
        self.index = []
        self.sen_len=0
        self.parent = None

    # def __del__(self):
    #     print("****del****")

####initialize the cross-lingual token feature
def str2vec(str):
    inputs = tokenizer(str, return_tensors="pt")
    
    outputs = token_model(**inputs)
    last_hidden_states = outputs.last_hidden_state #torch.Size([1, sen_len, 768])
    sen_len=int(last_hidden_states.size()[-2])-2
    # word_vec= last_hidden_states.squeeze(0)[1:-1]  #drop out the [CLS] and [SEP]
    word_vec = last_hidden_states.squeeze(0)[0]   #sentece_vector
    return word_vec, sen_len


def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq=float(pair.split(':')[1])
        index=int(pair.split(':')[0])
        if index<=5000:
            wordFreq.append(freq)
            wordIndex.append(index)
    return wordFreq, wordIndex

def constructMat(tree):
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        token_features, sen_len = str2vec(tree[j]['text'])
        nodeC.inputs_features.append(token_features)
        nodeC.sen_len = sen_len

        # nodeC.index = wordIndex
        # nodeC.word = wordFreq
        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            rootindex=indexC-1
            rootfeat=nodeC.inputs_features[0]   #tensor
            # root_index=nodeC.index
            # root_word=nodeC.word
    # rootfeat = np.zeros([1, 5000])
    # if len(root_index)>0:
    #     rootfeat[0, np.array(root_index)] = np.array(root_word)
    matrix=np.zeros([len(index2node),len(index2node)])
    row=[]
    col=[]
    x_features=[]
    x_senlen=[]
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i+1].children != None and index2node[index_j+1] in index2node[index_i+1].children:
                matrix[index_i][index_j]=1
                row.append(index_i)
                col.append(index_j)
        # x_word.append(index2node[index_i+1].word)
        # x_index.append(index2node[index_i+1].index)
        x_features.append(index2node[index_i+1].inputs_features[0])  #[tensor(seq_len*emb), tensor, ...]
        x_senlen.append(index2node[index_i+1].sen_len)
    edgematrix=[row,col]
    return np.array(x_features), np.array(x_senlen), np.array(edgematrix),rootfeat.detach().numpy(),np.array(rootindex)

# def getfeature(x_word,x_index):
#     x = np.zeros([len(x_index), 5000])
#     for i in range(len(x_index)):
#         if len(x_index[i])>0:
#             x[i, np.array(x_index[i])] = np.array(x_word[i])
#     return x

def memory_print(id):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024.
    print('the id is{}, usd{} MB'.format(id, memory))


def loadEid(event,id,y):
    if event is None:
        return None
    if len(event) < 2:
        return None
    if len(event)>1:
        if os.path.exists(os.path.join(cwd, 'data/Twitter-COVID19/Twittergraph/'+id+'.npz')):
            pass
        else:
            memory_print(id)
            x_x, senlen, tree, rootfeat_, rootindex_ = constructMat(event)
            # rootfeat_, tree, x_x, rootindex_, y, senlen = root_feat.detach().numpy(), np.array(tree_), np.array(x_features), np.array(
                #     root_index), np.array(y), np.array(x_senlen)

            print('1')
            memory_print(id)
            # print('x_features {}, x_senlen {}, tree_ {}, root_feat {}, root_index {}'
            #       .format(sys.getrefcount(x_features),sys.getrefcount(x_senlen),sys.getrefcount(tree_),sys.getrefcount(root_feat),sys.getrefcount(root_index)))
            # del x_features, x_senlen, tree_, root_feat, root_index
            # print('2**')

            # memory_print(id)
            # gc.collect()
            # print('3**')
            # memory_print(id)

            # print(root_index)
            # objgraph.show_most_common_types(limit=50)
            np.savez( os.path.join(cwd, 'data/Twitter-COVID19/Twittergraph/'+id+'.npz'), x=x_x,root=rootfeat_,edgeindex=tree,rootindex=rootindex_,y=y, seqlen=senlen)
            memory_print(id)
            print('x_x {}, rootfeat_ {}, tree {}, rootindex_ {}, y {}, senlen {}'
                  .format(sys.getrefcount(x_x), sys.getrefcount(rootfeat_), sys.getrefcount(tree),sys.getrefcount(rootindex_), sys.getrefcount(y),sys.getrefcount(senlen)))
            del x_x, rootfeat_, tree, rootindex_, y, senlen
            print('4**')
            memory_print(id)
            gc.collect()
            memory_print(id)
        return None

def main(obj):
    treePath = os.path.join(cwd, 'data/Twitter-COVID19/Twitter/Twitter_data_all.txt')
    print("reading tree")
    treeDic = {}
    for line in open(treePath):
        line = line.strip('\n')
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        time_delay, text = float(line.split('\t')[3]), str(line.split('\t')[4])
        #print(text)
        #exit(0)

        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'time_delay': time_delay, 'text': text}
    print('tree no:', len(treeDic))

    labelPath = os.path.join(cwd, 'data/Twitter-COVID19/Twitter/Twitter_label_all.txt')
    labelset_nonR, labelset_f = ['news', 'non-rumor', '0'], ['false', '1']
    #need to be changed into the binary classification about the non_rumor and rumor
    print("loading Twitter tree label")
    event, y = [], []
    l1 = l2 = 0
    labelDic = {}
    for line in open(labelPath):
        line = line.strip('\n')
        line = line.rstrip()
        label, eid = line.split('\t')[1], line.split('\t')[0]
        label=label.lower()
        event.append(eid)
        if label in labelset_nonR:
            labelDic[eid]=0
            l1 += 1
        if label  in labelset_f:
            labelDic[eid]=1
            l2 += 1
    
    print(len(labelDic), len(event), len(y))
    print(l1, l2)


    print("loading dataset", )
    # for eid in tqdm(event):
    #     loadEid(treeDic[eid] if eid in treeDic else None,eid,labelDic[eid])
    # Parallel(n_jobs=2, backend='threading')(delayed(loadEid)(treeDic[eid] if eid in treeDic else None,eid,labelDic[eid]) for eid in tqdm(event))
    # loadEid(treeDic['1236289649737371648'],'1236289649737371648',labelDic['1236289649737371648'])
    for eid in tqdm(event):
        loadEid(treeDic[eid] if eid in treeDic else None,eid,np.array(labelDic[eid]))
    return

if __name__ == '__main__':
    # obj= sys.argv[1]
    obj = "Twitter"
    main(obj)
