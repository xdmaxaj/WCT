import json
import os

data_dir = '../data/Weibo/Weibo_ori_data'
label_path = '../data/Weibo/Weibo.txt'


def label_read(label_dir):
    label_dic = {}
    with open(label_dir, 'r') as f:
        for label_ori in f.readlines():
            if 'label:' not in label_ori:
                continue
            else:
                eid, label = label_ori.split('\t')[0][4:], label_ori.split('\t')[1][6:7]
                label_dic[eid] = label
    return label_dic


def read_data(data_path):
    write_sign = 0
    with open(data_path, 'r') as f:
        o_data = json.load(f)
    out_list, user_id_list, aben_id_list = [], [], []
    out_dic, user_id_dict = {}, {}
    out_dic['eid'] = o_data[0]['id']
    id_no = 1
    for data in o_data:

        if data['parent'] is None and data['original_text'] == '转发微博':
            write_sign = 1
            return out_list, write_sign
        if data['original_text'].rstrip() == '':
            aben_id_list.append(data['mid'])
            continue
        if data['parent'] in aben_id_list:
            aben_id_list.append(data['mid'])
            continue
        out_dic['parent'] = None
        out_dic['current'] = None
        out_dic['content'] = ''

        user_id_dict[data['mid']] = str(id_no)
        user_id_list.append(data['mid'])
        # out_dic['current'] = user_id_dict[data['mid']]
        # if data['parent'] is not None:
        #     out_dic['parent'] = user_id_dict[data['parent']]

        out_dic['parent'] = data['parent']
        out_dic['current'] = data['mid']
        out_dic['content'] = data['original_text']
        out_list.append(out_dic.copy())
        id_no += 1
    # print(user_id_list)
    for out in out_list:
        out['current'] = user_id_dict[out['current']]
        if out['parent'] is not None:
            out['parent'] = user_id_dict[out['parent']]
        if out['content'] == '转发微博' or out['content'] == '轉發微博':
            out['content'] = out_list[int(out['parent']) - 1]['content']
    # o_data = json.loads(data_path)
    return out_list, write_sign


def write_data(out_data_list, label_dic):
    eid=''
    for data in out_data_list:
        with open("../data/Weibo/weibo_data_all.txt", "a") as f:
            f.write(data['eid'] + '\t' + str(data['parent']) + '\t' + data['current'] + '\t' + data['content'] + '\n')
        if eid != data['eid']:
            eid = data['eid']
            with open('../data/Weibo/weibo_label_all.txt', 'a') as f:
                f.write(data['eid'] + '\t' + label_dic[data['eid']] + '\n')


def main(data_dir):
    fs = os.listdir(data_dir)
    label_dic = label_read(label_path)
    for filename in fs:
        data_path = os.path.join(data_dir, filename)
        out_data_list, write_sign = read_data(data_path)
        if write_sign == 0:
            write_data(out_data_list, label_dic)


if __name__ == '__main__':
    main(data_dir)

