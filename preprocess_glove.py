# -*- coding: utf-8 -*-
import os
import gc
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from transformers import BertTokenizer, ErnieForMaskedLM
import json
import datetime
from tqdm import tqdm
import argparse
from configs.Config import get_config
sys.path.append('./Glove/SIF-master/src')
import data_io, params, SIF_embedding
np.seterr(divide='ignore',invalid='ignore')

# wordfile = './Glove/glove.840B.300d.txt' # word vector file, can be downloaded from GloVe website
# weightfile = './Glove/SIF-master/auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency
wordfile = './Glove/CN_300d.txt' # word vector file, can be downloaded from GloVe website
weightfile = './Glove/Weibo_frequency.txt' # each line is a word and its frequency

weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme

(words, We) = data_io.getWordmap_CN(wordfile)
word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word
# set parameters
params = params.params()
params.rmpc = rmpc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh")
# token_model = ErnieForMaskedLM.from_pretrained("nghuyong/ernie-3.0-base-zh")
# token_model = token_model.to(device)

PHEME_path = 'datasets/PHEME/all-rnr-annotated-threads'
Source_graph_path = 'datasets/PHEME/Source_graph'
Target_graph_path = 'datasets/PHEME/Target_graph'

Twitter_path_source = 'datasets/Twitter/Twitter/Twitter_data_all.txt'
Twitter_path_target = 'datasets/Twitter/Twitter_covid/Twitter_data_all.txt'
Twitter_path_source_label = 'datasets/Twitter/Twitter/Twitter_label_all.txt'
Twitter_path_target_label = 'datasets/Twitter/Twitter_covid/Twitter_label_all.txt'
Twitter_Source_graph_path = 'datasets/Twitter/Source_graph/'
Twitter_Target_graph_path = 'datasets/Twitter/Target_graph/'

Weibo_path_source = 'datasets/Weibo/Weibo/Weibo'
Weibo_path_target = 'datasets/Weibo/Weibo_covid/weibo_covid19_data.txt'
Weibo_path_source_label = 'datasets/Weibo/Weibo/Weibo.txt'
Weibo_path_target_label = 'datasets/Weibo/Weibo_covid/weibo_covid19_label.txt'
Weibo_Source_graph_path = 'datasets/Weibo/Source_graph/'
Weibo_Target_graph_path = 'datasets/Weibo/Target_graph/'

Twitter_1516_source_context_path = 'datasets/Twitter15_16/Twitter15/tweet_response15_clean.txt'
Twitter_1516_target_context_path = 'datasets/Twitter15_16/Twitter16/tweet_response16_clean.txt'
Twitter_1516_path_source = 'datasets/Twitter15_16/Twitter15/new_tree15'
Twitter_1516_path_target = 'datasets/Twitter15_16/Twitter16/new_tree16'
Twitter_1516_path_source_label = 'datasets/Twitter15_16/Twitter15/Twitter15_label_All.txt'
Twitter_1516_path_target_label = 'datasets/Twitter15_16/Twitter16/Twitter16_label_All.txt'
Twitter_1516_source_claim_context_path = 'datasets/Twitter15_16/Twitter15/source_tweets15.txt'
Twitter_1516_target_claim_context_path = 'datasets/Twitter15_16/Twitter16/source_tweets16.txt'
Twitter_1516_Source_graph_path = 'datasets/Twitter15_16/Source_graph/'
Twitter_1516_Target_graph_path = 'datasets/Twitter15_16/Target_graph/'


class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        # self.idx = idx
        # self.word = []
        self.inputs_features = []
        # self.index = []
        self.sen_len = 0
        self.parent = None


def ori_PHEME_class():
    file_list = os.listdir(PHEME_path)
    s_list = ['c', 's', 'f', 'o', 'g', 'r']
    event_name_list = [event_name for event_name in file_list if event_name[0] != '.']
    source_event_list = [event_name for event_name in event_name_list if event_name[0] in s_list]
    target_event_list = [event_name for event_name in event_name_list if event_name[0] not in s_list]
    return event_name_list, source_event_list, target_event_list


def ori_Twitter_class(source_label_path, target_label_path):
    source_event_list = []
    # for line in open(label_path):
    #     event_name = line.split('\t')[0]
    #     source_event_list.append(event_name)
    source_event_list = [line.split('\t')[0] for line in open(source_label_path)]
    target_event_list = [line.split('\t')[0] for line in open(target_label_path)]
    source_label_list = [line.split('\t')[1] for line in open(source_label_path)]
    target_label_list = [line.split('\t')[1] for line in open(target_label_path)]
    return source_event_list, target_event_list, source_label_list, target_label_list


def ori_Weibo_class(source_label_path, target_label_path):
    source_event_list = []
    # for line in open(label_path):
    #     event_name = line.split('\t')[0]
    #     source_event_list.append(event_name)
    source_event_list = [(line.split('\t')[0]).split(':')[1] for line in open(Weibo_path_source_label)]
    target_event_list = [line.split('\t')[0] for line in open(Weibo_path_target_label)] 
    source_label_list = [(line.split('\t')[1]).split(':')[1] for line in open(Weibo_path_source_label)]
    target_label_list = [line.split('\t')[1] for line in open(Weibo_path_target_label)]
    return source_event_list, target_event_list, source_label_list, target_label_list


def ori_Twitter1516_class(source_label_path, target_label_path):
    source_event_list = []
    # for line in open(label_path):
    #     event_name = line.split('\t')[0]
    #     source_event_list.append(event_name)
    source_event_list = [line.split('\t')[2] for line in open(source_label_path)]
    target_event_list = [line.split('\t')[2] for line in open(target_label_path)]
    source_label_list = [line.split('\t')[0] for line in open(source_label_path)]
    target_label_list = [line.split('\t')[0] for line in open(target_label_path)]
    
    #label=0 for non-rumor, 'unverified,true,false' means rumor(label=1)
    for i in range(len(source_label_list)):
        if source_label_list[i] == 'non-rumor':
            source_label_list[i] = '0'
        else:
            source_label_list[i] = '1'
    
    for i in range(len(target_label_list)):
        if target_label_list[i] == 'non-rumor':
            target_label_list[i] = '0'
        else:
            target_label_list[i] = '1'
            
    
    print('Twitter15')
    print(source_label_list.count('0'))
    print(source_label_list.count('1'))
    
    print('Twitter16')
    print(target_label_list.count('0'))
    print(target_label_list.count('1'))
    
    return source_event_list, target_event_list, source_label_list, target_label_list


def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq = float(pair.split(':')[1])
        index = int(pair.split(':')[0])
        if index <= 5000:
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
            rootindex = indexC - 1
            rootfeat = nodeC.inputs_features[0]  # tensor
            # root_index=nodeC.index
            # root_word=nodeC.word
    # rootfeat = np.zeros([1, 5000])
    # if len(root_index)>0:
    #     rootfeat[0, np.array(root_index)] = np.array(root_word)
    matrix = np.zeros([len(index2node), len(index2node)])
    row = []
    col = []
    x_features = []
    x_senlen = []
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i + 1].children is not None and index2node[index_j + 1] in index2node[
                index_i + 1].children:
                matrix[index_i][index_j] = 1
                row.append(index_i)
                col.append(index_j)
        # x_word.append(index2node[index_i+1].word)
        # x_index.append(index2node[index_i+1].index)
        x_features.append(index2node[index_i + 1].inputs_features[0])  # [tensor(seq_len*emb), tensor, ...]
        x_senlen.append(index2node[index_i + 1].sen_len)
    edgematrix = [row, col]
    # return x_features, x_senlen, edgematrix, rootfeat
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
            rootindex = indexC - 1
            rootfeat = nodeC.inputs_features[0]  # tensor
            # root_index=nodeC.index
            # root_word=nodeC.word
    # rootfeat = np.zeros([1, 5000])
    # if len(root_index)>0:
    #     rootfeat[0, np.array(root_index)] = np.array(root_word)
    matrix = np.zeros([len(index2node), len(index2node)])
    row = []
    col = []
    x_features = []
    x_senlen = []
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i + 1].children is not None and index2node[index_j + 1] in index2node[
                index_i + 1].children:
                matrix[index_i][index_j] = 1
                row.append(index_i)
                col.append(index_j)
        # x_word.append(index2node[index_i+1].word)
        # x_index.append(index2node[index_i+1].index)
        x_features.append(index2node[index_i + 1].inputs_features[0])  # [tensor(seq_len*emb), tensor, ...]
        x_senlen.append(index2node[index_i + 1].sen_len)
    edgematrix = [row, col]
    # return x_features, x_senlen, edgematrix, rootfeat, rootindexdef constructMat(tree):
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
            rootindex = indexC - 1
            rootfeat = nodeC.inputs_features[0]  # tensor
            # root_index=nodeC.index
            # root_word=nodeC.word
    # rootfeat = np.zeros([1, 5000])
    # if len(root_index)>0:
    #     rootfeat[0, np.array(root_index)] = np.array(root_word)
    matrix = np.zeros([len(index2node), len(index2node)])
    row = []
    col = []
    x_features = []
    x_senlen = []
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i + 1].children is not None and index2node[index_j + 1] in index2node[
                index_i + 1].children:
                matrix[index_i][index_j] = 1
                row.append(index_i)
                col.append(index_j)
        # x_word.append(index2node[index_i+1].word)
        # x_index.append(index2node[index_i+1].index)
        x_features.append(index2node[index_i + 1].inputs_features[0])  # [tensor(seq_len*emb), tensor, ...]
        x_senlen.append(index2node[index_i + 1].sen_len)
    edgematrix = [row, col]
    return x_features, x_senlen, edgematrix, rootfeat, rootindex


def eid2inedx(tree_list):
    eid_list, parent_eid_list, resort_eid_list = [], [], []
    index_dic = {}
    i = 0
    n = 0
    for post_dic in tree_list:
        if str(post_dic['current_eid']) in eid_list:
            count = eid_list.count(str(post_dic['current_eid']))
            eid_list.append(str(post_dic['current_eid']))
            # index_dic[str(post_dic['current_eid'])] = i
            if post_dic['parent'] == None and i != 0:
                post_dic['parent'] = eid_list[0]
            if post_dic['parent'] != None:
                parent_eid_list.append(post_dic['parent'])
                if parent_eid_list[i - 1] not in eid_list:
                    resort_eid_list.append((i, parent_eid_list[i - 1]))
                    # del (tree_list[i])
                    continue
            post_dic['current_eid'] = i
            i = i + 1
            continue

        eid_list.append(str(post_dic['current_eid']))

        if post_dic['parent'] == None and i != 0:
            post_dic['parent'] = eid_list[0]
        if post_dic['parent'] != None:
            parent_eid_list.append(post_dic['parent'])
            if parent_eid_list[i - 1] not in eid_list:
                post_dic['parent'] = eid_list[0]
#                 resort_eid_list.append((i, parent_eid_list[i - 1]))
                # del (tree_list[i])
#                 continue
        index_dic[str(post_dic['current_eid'])] = i

        post_dic['current_eid'] = index_dic[str(post_dic['current_eid'])]
        i += 1

    for k in resort_eid_list:
        del(tree_list[k[0]])
        # n += 1
    for post_dic in tree_list:
        if post_dic['parent'] != None:
            post_dic['parent'] = index_dic[post_dic['parent']]
    return tree_list



def get_timestamp(date):
    month_dic = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    date_split = date.split(' ')
    date = date_split[-1]+'-'+month_dic[date_split[1]]+'-'+date_split[2]+' '+date_split[3]
    return datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").timestamp()


def get_edgematrix(tree_list):
    row, col = [], []
    mark = 0
    for event in tree_list:
        if event['parent'] == None:
            if event['current_eid'] == 1:
                mark =1
            continue
        if mark == 1:
            row.append(event['parent'] - 1)
            col.append(event['current_eid'] - 1)
        else:
            row.append(event['parent'])
            col.append(event['current_eid'])
    edgematrix= [row, col]

    return edgematrix


def str2vec(str1, str2, tokenizer, token_model):
    if str2 == None:
        inputs = tokenizer(str1, return_tensors="pt").to(device)
        # print("Encoded sequence(AB):", inputs["input_ids"].tolist()[0])
        # decoded_ab = tokenizer.decode(inputs["input_ids"].tolist()[0])
        # print("Decoded sequence(AB):", decoded_ab)
        outputs = token_model(**inputs)
        last_hidden_states = outputs.last_hidden_state  # torch.Size([1, sen_len, 768])
#         print(last_hidden_states.size())
        sen_len = int(last_hidden_states.size()[-2]) - 2
        # word_vec= last_hidden_states.squeeze(0)[1:-1]  #drop out the [CLS] and [SEP]
        word_vec = last_hidden_states.squeeze(0)[0]  # sentece_vector
#         print(word_vec.size())
        return word_vec.cpu(), sen_len
    else:
        inputs = tokenizer(str1, str2, return_tensors="pt").to(device)
        if inputs['input_ids'].size()[1] > 513:
            inputs = tokenizer(str2, return_tensors="pt").to(device)
        # print("Encoded sequence(AB):", inputs["input_ids"].tolist()[0])
        # decoded_ab = tokenizer.decode(inputs["input_ids"].tolist()[0])
        # print("Decoded sequence(AB):", decoded_ab)
        outputs = token_model(**inputs)
        last_hidden_states = outputs.last_hidden_state  # torch.Size([1, sen_len, 768])
        sen_len = int(last_hidden_states.size()[-2]) - 4
        # word_vec= last_hidden_states.squeeze(0)[1:-1]  #drop out the [CLS] and [SEP]
        word_vec = last_hidden_states.squeeze(0)[0]  # sentece_vector
        return word_vec.cpu(), sen_len


def str2vec_ernie(str1, str2, tokenizer, token_model):
    if str2 == None:
        inputs = tokenizer(str1, return_tensors="pt").to(device)
        
        last_hidden_states = token_model(**inputs, output_hidden_states=True).hidden_states[-1].squeeze(0)[0]
#         last_hidden_states = outputs.last_hidden_state  # torch.Size([1, sen_len, 768])
        sen_len = inputs['input_ids'].size()[1]
#         sen_len = int(last_hidden_states.size()[-2]) - 2
        # word_vec= last_hidden_states.squeeze(0)[1:-1]  #drop out the [CLS] and [SEP]
        word_vec = last_hidden_states  # sentece_vector
        return word_vec.cpu(), sen_len
    else:
        inputs = tokenizer(str1, str2, return_tensors="pt").to(device)
        if inputs['input_ids'].size()[1] > 513:
            inputs = tokenizer(str2, return_tensors="pt").to(device)
        # print("Encoded sequence(AB):", inputs["input_ids"].tolist()[0])
        # decoded_ab = tokenizer.decode(inputs["input_ids"].tolist()[0])
        # print("Decoded sequence(AB):", decoded_ab)
        last_hidden_states = token_model(**inputs, output_hidden_states=True).hidden_states[-1].squeeze(0)[0]
#         last_hidden_states = outputs.last_hidden_state  # torch.Size([1, sen_len, 768])
        sen_len = inputs['input_ids'].size()[1]
#         sen_len = int(last_hidden_states.size()[-2]) - 4
        # word_vec= last_hidden_states.squeeze(0)[1:-1]  #drop out the [CLS] and [SEP]
        word_vec = last_hidden_states  # sentece_vector
        return word_vec.cpu(), sen_len

    
def str2vec_glove(str1):
    # load sentences
    sentences = [str1]
    x, m = data_io.sentences2idx(sentences, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    sen_len = x.shape[1]
    w = data_io.seq2weight(x, m, weight4ind) # get word weights
    # get SIF embedding
    embedding = SIF_embedding.SIF_embedding(We, x, w, params) # embedding[i,:] is the embedding for sentence i
    emb = torch.tensor(embedding).squeeze()
    return emb*10e10, sen_len


def save_npz(x_features, x_senlen, root_feature, root_index, edge_matrix_list, y, id, save_path):
    rootfeat = root_feature.detach().numpy()
    tree =  np.array(edge_matrix_list)
    x_x = np.array(x_features)
    rootindex = np.array(root_index)
    y = np.array(y)
    senlen = np.array(x_senlen)
    
#     rootfeat, tree, x_x, rootindex, y, senlen = root_feature.detach().numpy(), np.array(edge_matrix_list), np.array(x_features), np.array(root_index), np.array(y), np.array(x_senlen)
    np.savez(os.path.join(save_path + id + '.npz'), x=x_x, root=rootfeat,
             edgeindex=tree, rootindex=rootindex, y=y, seqlen=senlen)
    return None


def read_data(event_name, source_event_list, target_event_list,save_path, tokenizer, token_model):

    # event_name = 'ferguson-all-rnr-threads'

    event_path_0 = os.path.join(PHEME_path, event_name, 'non-rumours')
    event_path_1 = os.path.join(PHEME_path, event_name, 'rumours')
    event_path = [event_path_0, event_path_1]
    for path in range(2):
        non_rumor_event_path = event_path[path]
        for i in tqdm(range(len(os.listdir(non_rumor_event_path))), desc='Processing'):
            event = os.listdir(non_rumor_event_path)[i]
                
#         for event in os.listdir(non_rumor_event_path):
            if os.path.exists(os.path.join(save_path + event + '.npz')):
                continue
            non_rumor_tree_list = []
            non_rumor_tree_dict = {}
            if event[0] == '.':
                continue

            # event = '498235547685756928'

            with open(os.path.join(non_rumor_event_path, event, 'structure.json'), 'r', encoding='utf8') as fp:
                structure = json.load(fp)
            reaction_path = os.path.join(non_rumor_event_path, event, 'reactions')
            if len(os.listdir(reaction_path)) == 0:
                print(non_rumor_event_path + '/' + event)
                continue
            claim_path = os.path.join(non_rumor_event_path, event, 'source-tweets')
            for claim in os.listdir(claim_path):
                if claim[0] == '.':
                    continue
                with open(claim_path + '/' + claim, 'r', encoding='utf8') as fp:
                    claim_tweet = json.load(fp)
                root_eid = claim_tweet['id']
                non_rumor_tree_dict = {'current_eid': claim_tweet['id'],
                                       'parent': claim_tweet['in_reply_to_status_id_str'],
                                       'context': claim_tweet['text']}
                non_rumor_tree_list.append(non_rumor_tree_dict.copy())
            time_list = [claim_tweet['created_at']]
            for post in os.listdir(reaction_path):
                if post[0] == '.':
                    continue
                with open(reaction_path + '/' + post, 'r', encoding='utf8') as fp:
                    reaction_tweet = json.load(fp)
                time_list.append(reaction_tweet['created_at'])
                non_rumor_tree_dict = {'current_eid': reaction_tweet['id'],
                                       'parent': reaction_tweet['in_reply_to_status_id_str'],
                                       'context': reaction_tweet['text']}
                non_rumor_tree_list.append(non_rumor_tree_dict.copy())
            s = sorted(range(len(time_list)), key=lambda k: get_timestamp(time_list[k]))
            non_rumor_tree_list = [non_rumor_tree_list[s[k]] for k in range(len(s))]
            tree_list = eid2inedx(non_rumor_tree_list)
            edge_matrix_list = get_edgematrix(tree_list)
            x_features = []
            x_senlen = []
            for post in tree_list:
                if post['parent'] == None:
                    node_feature, sen_len = str2vec_glove(post['context'])
                    claim_context = post['context']
                    x_features.append(node_feature)
                    claim_len = sen_len
                    x_senlen.append(sen_len)
                    root_feature = node_feature
                    root_index = post['current_eid']
                    continue
                node_feature, sen_len = str2vec_glove(post['context'])
                x_features.append(node_feature)
                x_senlen.append(sen_len)
               
            save_npz(x_features, x_senlen, root_feature, root_index, edge_matrix_list, y=path, id=event, save_path=save_path)
    return None


def read_data_Twitter_1516(file_path, context_path, claim_path, event_name, label, source_event_list, target_event_list,source_label_list, target_label_list, save_path, tokenizer, token_model):
    tree_dic = {}
    tree_list = []
    num_of_lines = 0
    mark = 0
    for file_name in os.listdir(file_path):
        if event_name == file_name.split('.')[0]:
            mark = 1
            with open(file_path + '/' + file_name) as f:
                true_event_eid = f.readline().split('\t')[0]
            if event_name != true_event_eid:
                mark = 2
            for claim in open(claim_path):
                claim = claim.strip('\n')
                claim = claim.rstrip()
                if claim.split('\t')[0] == event_name:
                    text = claim.split('\t')[1]
                    break
            if mark==2:
                tree_dic = {'current_eid': true_event_eid, 'parent': None, 'time_delay': 0.0, 'context': text}
            else:
                tree_dic = {'current_eid': event_name, 'parent': None, 'time_delay': 0.0, 'context': text}
            tree_list.append(tree_dic.copy())
            time_list = [0.0]
            for line in open(file_path + '/' + file_name):
                line = line.strip('\n')
                line = line.rstrip()
                if line == '':
                    return None
                indexP, time_delay_parent, indexC, time_delay_current = line.split('\t')[0], float(line.split('\t')[1]), line.split('\t')[2], float(line.split('\t')[3])
                
                for context in open(context_path):
                    context = context.strip('\n')
                    context = context.rstrip()
                    if context.split('\t') == indexC:
                        text = context.split('\t')[1]
                        break
                
                tree_dic = {'current_eid': indexC, 'parent': indexP, 'time_delay': time_delay_current, 'context': text}
                tree_list.append(tree_dic.copy())
                # sort tree_list by delay time
                time_list.append(tree_dic['time_delay'])
            s = sorted(range(len(time_list)))
            sorted_tree_list = [tree_list[s[k]] for k in range(len(s))]
            break
    if mark == 0:
        return None

    tree_list = eid2inedx(sorted_tree_list)
    edge_matrix_list = get_edgematrix(tree_list)
    x_features = []
    x_senlen = []
    
    for post in tree_list:
        if post['parent'] == None:
#             node_feature, sen_len = str2vec(post['context'], None, tokenizer, token_model)
            node_feature, sen_len = str2vec_glove(post['context'])
            claim_context = post['context']
            x_features.append(node_feature)
            claim_len = sen_len
            x_senlen.append(sen_len)
            root_feature = node_feature
            root_index = post['current_eid']
            continue
#         node_feature, sen_len = str2vec(claim_context, post['context'], tokenizer, token_model)
        node_feature, sen_len = str2vec_glove(post['context'])    
        x_features.append(node_feature)
        x_senlen.append(sen_len)
  
    save_npz(x_features, x_senlen, root_feature, root_index, edge_matrix_list, y=label, id=event_name, save_path=save_path)

    return None


def read_data_Twitter(file_path, event_name, label, source_event_list, target_event_list,source_label_list, target_label_list, save_path, tokenizer, token_model):
    tree_dic = {}
    tree_list = []
    num_of_lines = 0
    for line in open(file_path):
        line = line.strip('\n')
        line = line.rstrip()
        eid = line.split('\t')[0]
        if eid == event_name:
            indexP, indexC, time_delay, text = line.split('\t')[1], int(line.split('\t')[2]), float(line.split('\t')[3]), str(line.split('\t')[4])
            if indexP == "None":
                indexP = None
            else:
                indexP = int(indexP)
            tree_dic = {'current_eid': indexC, 'parent': indexP, 'time_delay': time_delay, 'context': text}
            tree_list.append(tree_dic.copy())
        else:
            continue
    edge_matrix_list = get_edgematrix(tree_list)
    x_features = []
    x_senlen = []
    if len(tree_list) == 1:
        return None
    for post in tree_list:
        if post['parent'] == None:
            node_feature, sen_len = str2vec(post['context'], None, tokenizer, token_model)
            claim_context = post['context']
            x_features.append(node_feature.detach())
            claim_len = sen_len
            x_senlen.append(sen_len)
            root_feature = node_feature
            root_index = post['current_eid']
            continue
        node_feature, sen_len = str2vec(claim_context, post['context'], tokenizer, token_model)
        x_features.append(node_feature.detach())
        x_senlen.append(sen_len - claim_len)
#     for x in x_features:
#         print(x.requires_grad)
    save_npz(x_features, x_senlen, root_feature, root_index, edge_matrix_list, y=label, id=event_name, save_path=save_path)

    return None


def read_data_Weibo_Source(file_path, event_name, label, source_event_list, target_event_list,source_label_list, target_label_list, save_path):
    tree_dic = {}
    tree_list = []
    time_list = []
    num_of_lines = 0
    mark = 0
    for file_name in os.listdir(file_path):
        if event_name == file_name.split('.')[0]:
            mark = 1
            with open(os.path.join(Weibo_path_source, file_name), 'r', encoding='utf8') as fp:
                    structure = json.load(fp)
            for node in structure:
                if node['parent'] == None or node['parent'] == 'None':
                    indexP = None
                    claim_text = node['text']
                    text = claim_text
                else:
                    indexP = node['parent']
                    if '转发微博' in node['text']:
                        text = claim_text
                    else:
                        if len(node['text'])>0:
                            text = node['text']
                        else:
                            text = '无信息'
                indexC = node['mid']
                time_delay = int(node['t'])
                tree_dic = {'current_eid': indexC, 'parent': indexP, 'time_delay': time_delay, 'context': text}
                tree_list.append(tree_dic.copy())
                time_list.append(time_delay)
            s = sorted(range(len(time_list)))
            sorted_tree_list = [tree_list[s[k]] for k in range(len(s))]
            tree_list = eid2inedx(sorted_tree_list)
            edge_matrix_list = get_edgematrix(tree_list)
    x_features = []
    x_senlen = []
    for post in tree_list:
        if post['parent'] == None:
#             node_feature, sen_len = str2vec_ernie(post['context'], None, tokenizer, token_model)
#             with open('log.txt', 'a') as f:
#                 f.write(post['context'] + '\n')
#             print("#####################")
#             print(post['context'])
#             s = '今天凌晨襄阳宣布封城，整个湖北省封省了！新年第一天，愿亲人朋友能平安！愿战斗在一线的医护人员平安！愿那些素不相识的人们都能平安！[心][心][心]'
            node_feature, sen_len = str2vec_glove(post['context'])
#             print('######################')
            claim_context = post['context']
            x_features.append(node_feature)
            claim_len = sen_len
            x_senlen.append(sen_len)
            root_feature = node_feature
            root_index = post['current_eid']
            continue
#         node_feature, sen_len = str2vec_ernie(claim_context, post['context'], tokenizer, token_model)
#         print("!!!!!!!!!!!!!")
#         print(post['context'])
#         with open('log.txt', 'a') as f:
#             f.write(post['context'] + '\n')
        node_feature, sen_len = str2vec_glove(post['context'])
#         print("~~~~~~~~~~~~~~~~~~~~")
        x_features.append(node_feature)
        x_senlen.append(sen_len)

    save_npz(x_features, x_senlen, root_feature, root_index, edge_matrix_list, y=label, id=event_name, save_path=save_path)

    return None


def read_data_Weibo_Target(file_path, event_name, label, source_event_list, target_event_list,source_label_list, target_label_list, save_path):
    tree_dic = {}
    tree_list = []
    num_of_lines = 0
    for line in open(file_path):
        line = line.strip('\n')
        line = line.rstrip()
        eid = line.split('\t')[0]
        if eid == event_name:
            num_of_lines += 1
            indexP, indexC, time_delay, text = line.split('\t')[1], int(line.split('\t')[2]) - 1, float(line.split('\t')[3]), str(line.split('\t')[4])
            if indexP == "None":
                indexP = None
            else:
                indexP = int(indexP) - 1
            tree_dic = {'current_eid': indexC, 'parent': indexP, 'time_delay': time_delay, 'context': text}
            tree_list.append(tree_dic.copy())
        else:
            continue
    if num_of_lines == 1:
        return None
    edge_matrix_list = get_edgematrix(tree_list)
    x_features = []
    x_senlen = []
    for post in tree_list:
        if post['parent'] == None:
            node_feature, sen_len = str2vec_glove(post['context'])
#             node_feature, sen_len = str2vec_ernie(post['context'], None, tokenizer, token_model)
            claim_context = post['context']
            x_features.append(node_feature)
            claim_len = sen_len
            x_senlen.append(sen_len)
            root_feature = node_feature
            root_index = post['current_eid']
            continue
        node_feature, sen_len = str2vec_glove(post['context'])
#         node_feature, sen_len = str2vec_ernie(claim_context, post['context'], tokenizer, token_model)
        x_features.append(node_feature)
        x_senlen.append(sen_len)

    save_npz(x_features, x_senlen, root_feature, root_index, edge_matrix_list, y=label, id=event_name, save_path=save_path)

    return None



def print_label(event_name, save_path, data_class):
    event_path_0 = os.path.join(PHEME_path, event_name, 'non-rumours')
    event_path_1 = os.path.join(PHEME_path, event_name, 'rumours')
    event_path = [event_path_0, event_path_1]
    for path in range(2):
        non_rumor_event_path = event_path[path]
        for event in os.listdir(non_rumor_event_path):
            if event[0] == '.':
                continue
            with open(save_path + data_class + "_label_all.txt", "a") as f:
                f.write(event + '\t' + str(path) + '\n')
    return None


def write_label_txt(root_path):
    source_event_path = os.path.join(root_path, 'Source_graph')
    target_event_path = os.path.join(root_path, 'Target_graph')
    data_path = [source_event_path, target_event_path]
    domain_list = ['/Source', '/Target']
    for i in range(2):
        for file_name in os.listdir(data_path[i]):
            if file_name[0] == '.':
                continue
            event_name = file_name.split('.')[0]
            event = np.load(data_path[i] + '/' + file_name, allow_pickle=True)
            label = int(event.f.y)
            with open(root_path + domain_list[i] + "_label_all.txt", "a") as f:
                f.write(event_name + '\t' + str(label) + '\n')
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processdataset', default=None, type=str, help='Choose from:PHEME, Twitter15_16, Twitter, Weibo')
    
    args, unparsed = parser.parse_known_args(args=[])
    config = get_config(args, process='PRE')

    return args, config


def main():
    args, config = parse_args()
    obj = config['PRE']['DATASET_NAME']
    if obj == 'Weibo':
        tokenizer=None
#         tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh")
#         token_model = ErnieForMaskedLM.from_pretrained("nghuyong/ernie-3.0-base-zh")
#         token_model = token_model.to(device) 
    else:
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        token_model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        token_model = token_model.to(device)
    if obj == 'PHEME':
        event_name_list, source_event_list, target_event_list = ori_PHEME_class()
        for event_name in source_event_list:
            read_data(event_name, source_event_list, target_event_list, save_path='datasets/PHEME/Source_graph/', tokenizer=tokenizer, token_model=token_model)
            print_label(event_name, save_path='datasets/PHEME/', data_class='Source')
        print(1)
        for event_name in target_event_list:
            read_data(event_name, source_event_list, target_event_list, save_path='datasets/PHEME/Target_graph/', tokenizer=tokenizer, token_model=token_model)
            print_label(event_name, save_path='datasets/PHEME/', data_class='Target')
        print(2)

    if obj == 'Twitter':
        source_event_list, target_event_list, source_label_list, target_label_list = \
            ori_Twitter_class(source_label_path=Twitter_path_source_label,
                              target_label_path=Twitter_path_target_label)
        for i in tqdm(range(len(source_event_list)), desc='Processing'):
            if os.path.exists(os.path.join(Twitter_Source_graph_path + source_event_list[i] + '.npz')):
                continue
            read_data_Twitter(file_path=Twitter_path_source,
                              event_name=source_event_list[i],
                              label=source_label_list[i],
                              source_event_list=source_event_list,
                              target_event_list=target_event_list,
                              source_label_list=source_label_list,
                              target_label_list=target_label_list,
                              save_path=Twitter_Source_graph_path,
                              tokenizer=tokenizer, token_model=token_model)
        print(1)

        for i in tqdm(range(len(target_event_list)), desc='Processing'):
            if os.path.exists(os.path.join(Twitter_Target_graph_path + target_event_list[i] + '.npz')):
                continue
            read_data_Twitter(file_path=Twitter_path_target,
                              event_name=target_event_list[i],
                              label=target_label_list[i],
                              source_event_list=source_event_list,
                              target_event_list=target_event_list,
                              source_label_list=source_label_list,
                              target_label_list=target_label_list,
                              save_path=Twitter_Target_graph_path,
                              tokenizer=tokenizer, token_model=token_model)
        print(2)

    if obj == 'Weibo':
        source_event_list, target_event_list, source_label_list, target_label_list = \
            ori_Weibo_class(source_label_path=Weibo_path_source_label, target_label_path=Weibo_path_target_label)
        for i in tqdm(range(len(source_event_list)), desc='Processing'):
            if os.path.exists(os.path.join(Weibo_Source_graph_path + source_event_list[i] + '.npz')):
                continue
#             print(source_event_list[i])
            if source_event_list[i] == '3495745049431351' or source_event_list[i] == '3518511814617272':
                continue
#             print(source_event_list[i])
            read_data_Weibo_Source(file_path=Weibo_path_source,
                                  event_name=source_event_list[i],
                                  label=source_label_list[i],
                                  source_event_list=source_event_list,
                                  target_event_list=target_event_list,
                                  source_label_list=source_label_list,
                                  target_label_list=target_label_list,
                                  save_path=Weibo_Source_graph_path)
        print(1)

        for i in tqdm(range(len(target_event_list)), desc='Processing'):
            if os.path.exists(os.path.join(Weibo_Target_graph_path + target_event_list[i] + '.npz')):
                continue
            read_data_Weibo_Target(file_path=Weibo_path_target,
                                  event_name=target_event_list[i],
                                  label=target_label_list[i],
                                  source_event_list=source_event_list,
                                  target_event_list=target_event_list,
                                  source_label_list=source_label_list,
                                  target_label_list=target_label_list,
                                  save_path=Weibo_Target_graph_path)
        print(2)
        
#         root_path = 'datasets/Weibo'
#         write_label_txt(root_path)
        
    if obj == 'Twitter15_16':
        source_event_list, target_event_list, source_label_list, target_label_list = \
            ori_Twitter1516_class(source_label_path=Twitter_1516_path_source_label, target_label_path=Twitter_1516_path_target_label)
        print(os.listdir(Twitter_1516_path_source)[0])
        for i in tqdm(range(len(source_event_list)), desc='Processing'):
            if os.path.exists(os.path.join(Twitter_1516_Source_graph_path + source_event_list[i] + '.npz')):
                continue
            read_data_Twitter_1516(file_path=Twitter_1516_path_source,
                                  context_path=Twitter_1516_source_context_path,
                                  claim_path=Twitter_1516_source_claim_context_path,     
                                  event_name=source_event_list[i],
                                  label=source_label_list[i],
                                  source_event_list=source_event_list,
                                  target_event_list=target_event_list,
                                  source_label_list=source_label_list,
                                  target_label_list=target_label_list,
                                  save_path=Twitter_1516_Source_graph_path,
                                  tokenizer=tokenizer, token_model=token_model)
        print(1)

        for i in tqdm(range(len(target_event_list)), desc='Processing'):
            if os.path.exists(os.path.join(Twitter_1516_Target_graph_path + target_event_list[i] + '.npz')):
                continue
            read_data_Twitter_1516(file_path=Twitter_1516_path_target,
                                  context_path=Twitter_1516_target_context_path,
                                  claim_path=Twitter_1516_target_claim_context_path,  
                                  event_name=target_event_list[i],
                                  label=target_label_list[i],
                                  source_event_list=source_event_list,
                                  target_event_list=target_event_list,
                                  source_label_list=source_label_list,
                                  target_label_list=target_label_list,
                                  save_path=Twitter_1516_Target_graph_path,
                                  tokenizer=tokenizer, token_model=token_model)
        print(2)


if __name__ == '__main__':
    main()
