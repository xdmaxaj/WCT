import numpy as np
import json
import os
from tqdm import tqdm


def get_breadth_seq(index, data_path=os.path.join('..', '..', 'data', 'Weibograph')):

    id = index
    data = np.load(os.path.join(data_path, id + ".npz"), allow_pickle=True)
    edgeindex = data['edgeindex']
    edge_source_list = data['edgeindex'][0, :].tolist()
    edge_target_list = data['edgeindex'][1, :].tolist()
    a_matrix = np.zeros((data['edgeindex'].shape[1] + 1, data['edgeindex'].shape[1] + 1), dtype=int)
    a_matrix[edge_source_list, edge_target_list] = 1
    dep_node_dic = {}
    max_dep = 1
    for i in range(edgeindex.shape[1] + 1):
        if i == 0:
            current_root_path = a_matrix[0, :]
            dep_node_dic[str(i + 1)] = []
            dep_node_dic[str(i + 1)].extend(current_root_path.nonzero()[0].tolist())
            continue
        current_root_path = np.matmul(current_root_path, a_matrix)
        if current_root_path.nonzero()[0].shape[0] == 0:
            max_dep = i
            break
        dep_node_dic[str(i + 1)] = []
        dep_node_dic[str(i + 1)].extend(current_root_path.nonzero()[0].tolist())

    # dic_key_list = list(range(1, max_dep + 1))
    # dep_node_path = {}
    # total_list = []
    #
    # for i in range(len(dic_key_list)):
    #     current_dep = str(dic_key_list[max_dep - i - 1])
    #     dep_node_path[current_dep] = {}
    #     for leaf_node_id in range(len(dep_node_dic[current_dep])):
    #         if i != 0 and dep_node_dic[current_dep][leaf_node_id] in total_list:
    #             continue
    #         current_dep_int = int(current_dep)
    #         dep_node_path[current_dep][leaf_node_id] = []
    #         for j in range(current_dep_int):
    #             if j == 0:
    #                 current_node_col = dep_node_dic[current_dep][leaf_node_id]
    #             dep_node_path[current_dep][leaf_node_id].append(current_node_col)
    #             if j != current_dep_int - 1:
    #                 current_node_row = int(a_matrix[:, current_node_col].nonzero()[0][0])
    #                 current_node_col = current_node_row
    #             else:
    #                 total_list.append(dep_node_path[current_dep][leaf_node_id])
    #                 dep_node_path[current_dep][leaf_node_id].append(0)

    return dep_node_dic


def dep_seq2json(index, dep_node_dic, save_path, treePath):
    treeDic = {}
    mark = 0
    treeDic[index] = {}
    for line in open(treePath):
        line = line.strip('\n')
        line = line.rstrip()
        if line.split('\t')[0] != index and mark == 1:
            break
        if line.split('\t')[0] != index:
            continue
        else:
            mark = 1
            eid, indexC = line.split('\t')[0], int(line.split('\t')[2])
            text = str(line.split('\t')[4])
            treeDic[eid][indexC] = text
    seq_dic = {}
    seq_dic['index'] = index
    seq_dic['node_seq'] = {}
    seq_dic['context_seq'] = {}
    seq_dic['context_seq']['Breadth_first_seq'] = []
    seq_dic['context_seq']['Breadth_first_seq'].append(treeDic[index][1])
    for key in dep_node_dic:
        seq_dic['node_seq']['dep'+key] = dep_node_dic[key].copy()
        seq_dic['context_seq']['dep' + key] = {}
        node_list = dep_node_dic[key]
        context_list = []
        for i in range(len(node_list)):
            context_list.append(treeDic[index][node_list[i] + 1])
        seq_dic['context_seq']['dep'+key] = context_list.copy()
        seq_dic['context_seq']['Breadth_first_seq'].extend(context_list)
    seq_dic['context_seq']['root_claim'] = treeDic[index][1]
    str_seq_dic = json.dumps(seq_dic, ensure_ascii=False, indent=3)
    with open(os.path.join(save_path, index + '.json'), 'w') as f:
        f.write(str_seq_dic)
    return None


def main():
    dataset_name = 'Weibo-COVID19'  # 'Weibo-COVID19'/'Twitter-COVID19'
    cwd = os.path.dirname(os.getcwd())
    data_path = os.path.join(cwd, 'data', dataset_name, dataset_name.split('-')[0] + 'graph')
    save_path = os.path.join(cwd, 'data', dataset_name, dataset_name.split('-')[0] + 'breadth_first_seq')
    treePath = os.path.join(cwd, 'data', dataset_name, dataset_name.split('-')[0], 'weibo_covid19_data.txt')
    npz_file = os.listdir(data_path)
    for file in tqdm(npz_file):
        index = file.split('.')[0]
        dep_node_dic = get_breadth_seq(index, data_path=data_path)
        dep_seq2json(index, dep_node_dic, save_path, treePath)


if __name__ == "__main__":
    main()
