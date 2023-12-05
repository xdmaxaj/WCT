import json
import sys, os
import argparse
from configs.Config import get_config
import torch

sys.path.append(os.getcwd())
# sys.path.append('../../')
print(os.getcwd())
from torch_sparse import SparseTensor
from Process.process import *
from Process.data_augment import *
from Process.Pseudo_labels import *
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tools.earlystopping2class import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import copy
import random
from tensorboardX import SummaryWriter
import time
from torch.distributions import beta

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
run_time = '%s' % time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


class TDrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, x, data, Pseudo=False):
        edge_index = data.edge_index

        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        grad_list = []
        if Pseudo == True:
            def hook_backward(layer, input_grad, output_grad):
                grad_list.append(input_grad[1])

            h = self.conv2.register_backward_hook(hook_backward)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = scatter_mean(x, data.batch, dim=0)
        x = th.tanh(x)

        return x


class BUrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, x, data):
        edge_index = data.BU_edge_index

        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))  # data.batch起到了类似attention
            # mask的作用，标明了对应位置向量的batch_num，对batch中不同事件图的数据进行划分
            root_extend[index] = x1[rootindex[num_batch]]  # 此处input chanel的大小为hid_feats +
            # in_feats是因为在第一层GCN后，为了强化claim的节点特征，将原始的root_feature拼接到了该事件图的所有节点特征上
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = scatter_mean(x, data.batch, dim=0)
        x = th.tanh(x)
        return x

class SCL(th.nn.Module):
    def __init__(self, temperature=0.1):
        super(SCL, self).__init__()
        self.temperature = temperature

    def forward(self, inrep_1, inrep_2, label_1, label_2=None, label_weight=None, match=None):
        inrep_1.to(device)
        inrep_2.to(device)
        bs_1 = int(inrep_1.shape[0])
        bs_2 = int(inrep_2.shape[0])

        if label_2 is None:  # 源域或目标域数据同类自监督对比损失函数
            normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)
            normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)
            # 先将输入的样本特征向量进行L2归一化（normalize_inrep_1[bs,feature_len]）
            cosine_similarity = th.matmul(normalize_inrep_1,
                                          normalize_inrep_2.t())  # bs_1, bs_2，得到normalize_inrep_1的法矩阵（AA^T，[bs,bs]）
            # 得到cosine_similarity矩阵的对角向量，并将其作为矩阵的对角元
            diag = th.diag(cosine_similarity)
            cos_diag = th.diag_embed(diag)  # bs,bs
            # 先将label（[bs,]）进行维度扩充，变为[bs,1]，之后沿着第二维进行扩充，label_mat为[bs,bs]
            label = th.unsqueeze(label_1, -1)
            if label.shape[0] == 1:
                cos_loss = th.zeros(1).to(device)
                return cos_loss
            else:
                for i in range(label.shape[0] - 1):
                    if i == 0:
                        label_mat = th.cat((label, label), -1)
                    else:
                        label_mat = th.cat((label_mat, label), -1)  # bs, bs
                # print(label_mat.size())
                # print(label.size())
                # exit(0)

                mid_mat_ = (
                    label_mat.eq(label_mat.t()))  # 比较label矩阵与转置之后的label矩阵之间元素的差异，比较结果的每行代表batch中的一个样本与其他所有样本标签对比的结果
                mid_mat = mid_mat_.float()  # 将true或false的布尔值转化为浮点型的0、1
                # 计算输入的两个batch数据特征矩阵（batch_size*embeding_length，A、B）中样本之间的相似度（A*B^T）,之后去掉自相关的对角线元素
                # 去掉cosine_similarity对角元，即要求对比损失的两个对象为不同的样本，并除以温度系数
                cosine_similarity = (cosine_similarity - cos_diag) / self.temperature  # the diag is 0
                mid_diag = th.diag_embed(th.diag(mid_mat))
                mid_mat = mid_mat - mid_diag

                cosine_similarity = cosine_similarity.masked_fill_(mid_diag.byte(), -float('inf'))  # mask the diag
                # softmax(-inf)=0,下式中对角线元素加一的目的是将对角线元素（自相关的部分）在log后置为0下式中对角线元素加一的目的是将对角线元素（自相关的部分）在log后置为0
                cos_loss = th.log(
                    th.clamp(F.softmax(cosine_similarity, dim=1) + mid_diag, 1e-10, 1e10))  # the sum of each row is 1

                cos_loss = cos_loss * mid_mat  # 只需要相同类别的样本之间进行计算，拉近相同类别样本之间的距离

                cos_loss = th.sum(cos_loss, dim=1) / (
                        th.sum(mid_mat, dim=1) + 1e-10)  # 为原文中的源域数据同类自监督对比损失函数
        else:  # 源域与目标域数据同类自监督对比损失函数
            if bs_1 != bs_2:
                while bs_1 < bs_2:
                    inrep_2 = inrep_2[:bs_1]
                    label_2 = label_2[:bs_1]
                    break
                while bs_2 < bs_1:
                    inrep_2_ = inrep_2
                    ra = random.randint(0, int(inrep_2_.shape[0]) - 1)
                    pad = inrep_2_[ra].unsqueeze(0)
                    lbl_pad = label_2[ra].unsqueeze(0)
                    inrep_2 = th.cat((inrep_2, pad), 0)
                    label_2 = th.cat((label_2, lbl_pad), 0)
                    bs_2 = int(inrep_2.shape[0])

            normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)  # 计算归一化后的源域数据特征向量与目标域数据特征向量之间的余弦相似度
            normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)
            cosine_similarity = th.matmul(normalize_inrep_1, normalize_inrep_2.t())  # bs_1, bs_2

            label_1 = th.unsqueeze(label_1, -1)
            label_1_mat = th.cat((label_1, label_1), -1)
            for i in range(label_1.shape[0] - 1):  # 此处for循环的作用是将label_1扩充为一个32*32的tensor，每一行的32个值都是对应的batch中数据的标签
                if i == 0:
                    label_1_mat = label_1_mat
                else:
                    label_1_mat = th.cat((label_1_mat, label_1), -1)  # bs, bs

            label_2 = th.unsqueeze(label_2, -1)
            label_2_mat = th.cat((label_2, label_2), -1)
            for i in range(label_2.shape[0] - 1):
                if i == 0:
                    label_2_mat = label_2_mat
                else:
                    label_2_mat = th.cat((label_2_mat, label_2), -1)  # bs, bs

            mid_mat_ = (label_1_mat.t().eq(label_2_mat))  # 比较label_1转置后的tensor方阵与label_2对应位置元素标签是否相同，即将目标域batch
            # 中的每一个数据分别与源域batch中每一个数据的标签进行对比，判断是否属于同一类
            mid_mat = mid_mat_.float()

            cosine_similarity = cosine_similarity / self.temperature
            cos_loss = th.log(th.clamp(F.softmax(cosine_similarity, dim=1), 1e-10, 1e10))  # th.clamp的作用是将后续的返回值限制在一个范围内
            cos_loss = cos_loss * mid_mat  # find the sample with the same label
            cos_loss = th.sum(cos_loss, dim=1) / (th.sum(mid_mat, dim=1) + 1e-10)
        if match == True:
            # label_weight = label_weight.to(device)
            cos_loss = cos_loss * label_weight
        cos_loss = -th.mean(cos_loss, dim=0)  # 此处为公式前的负均值系数，至此对比损失实现完毕

        return cos_loss




def EMA(m, last_value, current_value):
    new_value = m * last_value + (1 - m) * current_value
    return new_value


class Net(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, temperature):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        # self.TDrumorGCN_Pos = TDrumorGCN_Pos(in_feats, hid_feats, out_feats)
        self.fc = th.nn.Linear((out_feats + hid_feats) * 2, 2)
        self.scl = SCL(temperature)

    def forward(self, data=None, source_data=None, target_data=None, pseudo_label=None, indices=None, twitter_data=None,
                Pseudo=False, confidence_constant=None, temperature=0.1, teacher_target_fea=None, weight_0=1, weight_1=1,
                branch="source_supervised"):
        if branch == "label_space_contrast":
            batch_size = len(target_data.eid)
            target_x = target_data.x
            TD_target = self.TDrumorGCN(target_x, target_data)
            BU_target = self.BUrumorGCN(target_x, target_data)
            student_target_feature = th.cat((BU_target, TD_target), 1)
            student_logits = self.fc(student_target_feature)
            student_prob = F.softmax(student_logits, dim=1)
            _, pred_label = student_prob.max(dim=-1)

            stu_normalize_target_feature = F.normalize(student_target_feature, p=2, dim=1)
            tea_normalize_target_feature = F.normalize(teacher_target_fea, p=2, dim=1)
            cos_similarity = torch.matmul(tea_normalize_target_feature, stu_normalize_target_feature.t())

            pseudo_label = torch.unsqueeze(pseudo_label, -1)
            pred_label = torch.unsqueeze(pred_label, -1)
            pseudo_label_mat = pseudo_label.repeat(1, batch_size)
            pred_label_mat = pred_label.repeat(1, batch_size)

            match_mat = (pseudo_label_mat.eq(pred_label_mat.t())).float()
            exp_similarity = torch.exp(cos_similarity / temperature)

            same_class_num = match_mat.sum(dim=0)
            same_class_num_list = same_class_num.int().tolist()
            non_zero_pos = torch.nonzero(same_class_num, as_tuple=False)
            non_zero_pos = non_zero_pos.squeeze(-1)

            total_similarity = exp_similarity.sum(dim=0)
            total_similarity_mat_whole = total_similarity.repeat(batch_size, 1)
            total_similarity_mat = total_similarity_mat_whole * match_mat
            same_class_similarity_mat = exp_similarity * match_mat

            same_class_similarity = same_class_similarity_mat.t().reshape((-1, 1))
            same_class_similarity = same_class_similarity.squeeze(-1)
            same_class_similarity = same_class_similarity[torch.nonzero(same_class_similarity)[:, 0]]

            total_similarity_nonzero = total_similarity_mat.t().reshape((-1, 1))
            total_similarity_nonzero = total_similarity_nonzero.squeeze(-1)
            total_similarity_nonzero = total_similarity_nonzero[torch.nonzero(total_similarity_nonzero)[:, 0]]

            base_contrast_loss = torch.log(same_class_similarity) - torch.log(total_similarity_nonzero)
            if len(non_zero_pos) != batch_size:
                if len(non_zero_pos) == 0:
                    non_zero_pos = torch.tensor([0])
                same_class_num_con = torch.clamp(-1 / same_class_num[non_zero_pos], -1, -0.00001)
                confidence_constant_vec = confidence_constant[non_zero_pos]
            else:
                same_class_num_con = torch.clamp(-1 / same_class_num, -1, -0.00001)
                confidence_constant_vec = confidence_constant
            st_len, i = 0, 0
            for length in same_class_num_list:
                if length == 0:
                    continue
                base_contrast_loss[st_len: length + st_len] = base_contrast_loss[st_len: length + st_len] * same_class_num_con[i] * confidence_constant_vec[i]
                st_len = length + st_len
                i += 1

            label_space_contrast_loss = base_contrast_loss.sum(dim=0) / len(base_contrast_loss)

            # ========================================
            # ====== feature space contrast loss =====
            # ========================================

            exp_similarity_diag = torch.diag(exp_similarity)
            fea_loss = torch.log(exp_similarity_diag) - torch.log(total_similarity)
            fea_loss = - fea_loss.sum(dim=0) / batch_size

            return label_space_contrast_loss, fea_loss

        if branch == "source_supervised":
            source_x = source_data.x
            TD_source = self.TDrumorGCN(source_x, source_data)
            BU_source = self.BUrumorGCN(source_x, source_data)
            source_feature = th.cat((BU_source, TD_source), 1)
            source_logits = self.fc(source_feature)
            # pred_prob = F.softmax(source_logits, dim=1)
            source_prob = F.log_softmax(source_logits, dim=1)
            # loss_fun = th.nn.CrossEntropyLoss()
            # source_CEloss = loss_fun(source_prob, source_data.y)
            source_CEloss = F.nll_loss(source_prob, source_data.y)
            _, pred = source_prob.max(dim=-1)
            correct = pred.eq(source_data.y).sum().item()
            train_acc = correct / len(source_data.y)
            return source_CEloss, train_acc

        if branch == "target_unsupervised":
            target_x = target_data.x
            TD_target = self.TDrumorGCN(target_x, target_data)
            BU_target = self.BUrumorGCN(target_x, target_data)
            target_feature = th.cat((BU_target, TD_target), 1)
            target_logits = self.fc(target_feature)
            target_prob = F.softmax(target_logits, dim=1)
            return target_feature, target_prob

        if branch == "target_unsupervised_pseudo":
            target_x = target_data.x
            TD_target = self.TDrumorGCN(target_x, target_data)
            BU_target = self.BUrumorGCN(target_x, target_data)
            target_feature = th.cat((BU_target, TD_target), 1)
            target_logits = self.fc(target_feature)
            target_prob = F.softmax(target_logits, dim=1)
            target_prob_select = target_prob[indices]
            loss_fun = th.nn.CrossEntropyLoss()
            target_CEloss_pseude = loss_fun(target_prob_select, pseudo_label)
            CE_loss = -torch.log(target_prob_select)
            weight_mat = torch.ones_like(CE_loss)
            one_hot_mat = torch.zeros_like(CE_loss)
            pos_vec = torch.arange(0, weight_mat.size()[0], 1)  
            one_hot_mat[pos_vec, pseudo_label] = 1
#             weight_vec = F.softmax(torch.tensor([weight_0, weight_1]))
            weight_vec = torch.tensor([weight_0, weight_1]) / torch.tensor([weight_0, weight_1]).sum()
            one_hot_mat[:, 0] = weight_vec[0] * one_hot_mat[:, 0]
            one_hot_mat[:, 1] = weight_vec[1] * one_hot_mat[:, 1]
            weight_CE_loss = (one_hot_mat * CE_loss).mean() * 2
            _, pred = target_prob.max(dim=-1)
            correct = pred.eq(target_data.y).sum().item()
            train_acc = correct / len(target_data.y)
            return target_CEloss_pseude, train_acc, weight_CE_loss

        if branch == "target_test":
            target_x = target_data.x
            TD_target = self.TDrumorGCN(target_x, target_data)
            BU_target = self.BUrumorGCN(target_x, target_data)
            target_feature = th.cat((BU_target, TD_target), 1)
            target_logits = self.fc(target_feature)
            target_prob = F.softmax(target_logits, dim=1)
            loss_fun = th.nn.CrossEntropyLoss()
            target_CEloss = loss_fun(target_prob, target_data.y)
            _, pred = target_prob.max(dim=-1)
            correct = pred.eq(target_data.y).sum().item()
            test_acc = correct / len(target_data.y)
            return target_CEloss, test_acc, pred

        if Pseudo == 'Fix_Match':
            Source_graph = twitter_data.x.clone()
            Target_graph = data.x.clone()
            # 计算源域数据的域内SCL损失以及源域数据分类的CE loss
            t = twitter_data.x  # Source
            TD_t = self.TDrumorGCN(t, twitter_data)
            BU_t = self.BUrumorGCN(t, twitter_data)
            t_ = th.cat((BU_t, TD_t), 1)
            twitter_scloss = self.scl(t_, t_, twitter_data.y)
            t = self.fc(t_)
            t = F.softmax(t, dim=1)
            loss_fun = th.nn.CrossEntropyLoss()
            twitter_CEloss = loss_fun(t, twitter_data.y.long())

            # 完成与源域数据相关的计算，开始输出目标域数据的伪标签
            x = data.x
            TD_x = self.TDrumorGCN(x, data)
            BU_x = self.BUrumorGCN(x, data)
            x_ = th.cat((BU_x, TD_x), 1)  # bs, (out_feats+hid_feats)*2
            x = self.fc(x_)
            P_target = F.softmax(x, dim=1)
            confidence, P_label = P_target.max(dim=1)

            Mix_root_vec, Mix_label_for_root = Root_Mix(Source_graph, Target_graph, data, twitter_data, P_target)
            Source_graph[twitter_data.rootindex] = Mix_root_vec
            twitter_data.x = Source_graph
            mix_TD_t = self.TDrumorGCN(twitter_data.x, twitter_data)
            mix_BU_t = self.BUrumorGCN(twitter_data.x, twitter_data)
            mix_t_ = th.cat((mix_BU_t, mix_TD_t), 1)
            mix_twitter_scloss = self.scl(mix_t_, mix_t_, twitter_data.y)
            mix_t = self.fc(mix_t_)
            mix_t = F.softmax(mix_t, dim=1)
            mix_CEloss = loss_fun(mix_t, twitter_data.y.long())

            threshold = 0.8
            indices = th.where(confidence > threshold)
            P_label_select = P_label[indices]
            golden_label_select = data.y[indices]
            correct = P_label_select.eq(golden_label_select).sum().item()
            if len(golden_label_select) == 0:
                temp_acc = 1.0
            else:
                temp_acc = correct / len(golden_label_select)
            select_ratio = len(indices[0]) / len(P_target)
            target_scloss = self.scl(x_[indices], x_[indices], P_label_select)
            target_CEloss = loss_fun(P_target[indices], P_label_select)
            return twitter_scloss, twitter_CEloss, select_ratio, temp_acc, mix_twitter_scloss, mix_CEloss, P_target






def train_AT(treeDic, x_test, x_train, target_train, TDdroprate, BUdroprate, noise_rate,lr, weight_decay, patience, n_epochs,
             batchsize, dataname, iter, lThreshold,hThreshold,lWeight,hWeight,EMA_m, init_model_path):
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    #  creat log file
    log_train_source ='model/' + dataname.split('_')[1] + '/logs/' + dataname.split('_')[0] + '_with_glove' + '/' + 'Source_train' + run_time
    writer_train_source = SummaryWriter(log_train_source)

    log_train = 'model/' + dataname.split('_')[1] + '/logs/' + dataname.split('_')[0] + '_with_glove' + '/' + 'Target_train' + run_time
    writer_train = SummaryWriter(log_train)

    log_test = 'model/' + dataname.split('_')[1] + '/logs/' + dataname.split('_')[0] + '_with_glove' + '/' + 'Target_test' + run_time
    writer_test = SummaryWriter(log_test)

    #  build student model
    model = Net(300, 512, 128, 0.1).to(device)  # 768/300,512,128
    print(model)
    model.load_state_dict(th.load(init_model_path))

    #  build teacher model
    model_teacher = Net(300, 512, 128, 0.1).to(device)

    #  optimizer
    optimizer = th.optim.AdamW([
        {'params': model.parameters()}
    ], lr=lr, weight_decay=weight_decay)

    #  train_loop
    # =====================================================
    # ================== training==========================
    # =====================================================
    for epoch in range(n_epochs):
        traindata_list, testdata_list = loadBiData_PHEME(dataname, treeDic, x_train, x_test, TDdroprate, BUdroprate,
                                                         data_class='Target')
#         twitterdata_list = loadBiData_PHEME(dataname, treeDic=treeDic, fold_x_train=target_train, fold_x_test=[],
#                                             TDdroprate=TDdroprate, BUdroprate=BUdroprate, data_class='Source')
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=0)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=0)
#         twitter_loader = DataLoader(twitterdata_list, batch_size=batchsize, shuffle=True, num_workers=1)
        avg_loss = []
        avg_acc = []
        pseudo_loss_list = []
        target_train_acc_list = []
        avg_temp_acc, avg_temp_acc_beta = [], []
        avg_select_ratio, avg_select_ratio_beta = [], []
        ultimate_label_1_count_list, ultimate_label_0_count_list, init_label_1_count_list, init_label_0_count_list = [], [], [], []
        batch_idx = 0
        tqdm_Target_loader = tqdm(train_loader)
#         tqdm_Source_loader = tqdm(twitter_loader)
        model.train()
        index = 0
        # =====================================================
        # =========== 1.updata teacher model ==================
        # =====================================================
        if epoch == 0:
            #  load checkpoint
            # model.load_state_dict(th.load('BiGCNPHEME.m'))
            model_teacher = update_teacher_model(model_teacher, model, keep_rate=0.00)
            # torch.save(model.state_dict(), 'PHEME_source' + '_labeled.m')

        if epoch > 0:
            #  updata teacher model with keep_rate 0.996
            model_teacher = update_teacher_model(model_teacher, model, keep_rate=EMA_m)
#             model_teacher = update_teacher_model(model_teacher, model, keep_rate=1.0)

        # =====================================================
        # ===== 2.teacher model produce pseudo label ==========
        # =====================================================
        for Batch_data in tqdm_Target_loader:
            ori_label = Batch_data.y.clone().to(device)

            #  target data augment: q-strong augment, k-weak augment
            labeled_target_data_q = strong_augment(Batch_data, batch_size=len(Batch_data.eid), noise_rate=noise_rate)
            target_batch_q = Batch_data.clone()
            target_batch_q.x = labeled_target_data_q

            #  weak augment(TD and BU drop rate for edge)
            target_batch_k = Batch_data.clone()

            #  teacher model produce pseudo label for weak augment target
            target_batch_k.to(device)
            with torch.no_grad():
                teacher_target_fea, target_batch_k_prob = model_teacher(target_data=target_batch_k,
                                                                        branch="target_unsupervised")

            #  select pseudo label with threshold > 0.8(one hot pseudo label)
            # P_label_select, indices = process_pseudo_label(target_batch_k_prob, threshold=0.8)
#             P_label_select_beta, indices_beta, beta_label_1_count, beta_label_0_count = process_pseudo_label(target_batch_k_prob, threshold=hThreshold)
            P_label_select, indices, ultimate_label_1_count, ultimate_label_0_count, init_label_1_count, init_label_0_count,ultimate_weight_0, ultimate_weight_1\
                = balance_process_pseudo_label(batch_size=len(Batch_data.eid), prob=target_batch_k_prob, threshold=hThreshold)
            ultimate_label_0_count_list.append(ultimate_label_0_count)
            ultimate_label_1_count_list.append(ultimate_label_1_count)
            init_label_0_count_list.append(init_label_0_count)
            init_label_1_count_list.append(init_label_1_count)

            P_label_select.to(device)
            
            pseudo_correct = P_label_select.eq(target_batch_k.y[indices[0]]).sum().item()
            pseudo_correct = pseudo_correct / len(indices[0])
            select_ratio = len(indices[0]) / len(target_batch_k.y)

            avg_temp_acc.append(pseudo_correct)
            avg_select_ratio.append(select_ratio)

            # =====================================================
            # ==== 3.student model process strong augment data ====
            # =====================================================

            #  process strong augment data with student model
            target_batch_q.to(device)
            target_CEloss_pseude, pseudo_train_acc, weight_target_CEloss = model(target_data=target_batch_q, pseudo_label=P_label_select,
                                                           indices=indices[0], weight_0=ultimate_weight_0, weight_1=ultimate_weight_1,
                                                           branch="target_unsupervised_pseudo")
            
            pseudo_loss_list.append(target_CEloss_pseude)
            target_train_acc_list.append(pseudo_train_acc)

            # =====================================================
            # ======== 4.student model contrast learning =========
            # =====================================================

            # 1) label space contrast:samples-pair <student:target_batch_q, teacher:target_batch_k>

            P_confidence, P_label = target_batch_k_prob.max(dim=1)
            confidence_constant = confidence_mapping(P_confidence)

            label_space_contrast_loss, feature_space_contrast_loss = model(target_data=target_batch_q,
                                              teacher_target_fea=teacher_target_fea,
                                              pseudo_label=P_label,
                                              confidence_constant=confidence_constant,
                                              branch="label_space_contrast")

            # 2) feature space contrast:samples-pair <same smaple processed by student and teacher>

            # =====================================================
            # 5.update student model with CE loss and contrast loss
            # =====================================================
            alf_list = [(0.8,0.1,0.1)]
#             alf_list = [(1.0,0.0,0.0)]
            alf1 = alf_list[0][0]
            alf2 = alf_list[0][1]
            alf3 = alf_list[0][2]
            loss = alf1 * weight_target_CEloss + alf2 * label_space_contrast_loss + alf3 * feature_space_contrast_loss
#           
            avg_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            avg_acc.append(pseudo_train_acc)
            postfix = "Iter {:03d} | Epoch {:05d} | Train_loss {:.4f} | Train_Accuracy {:.4f}| select_Accuracy {:.4f} | select_ratio {:.4f} | con_loss {:.4f} | CE loss {:.4f} ".format(
                iter, epoch,
                loss.item(),
                pseudo_train_acc,
                pseudo_correct,
                select_ratio,
                label_space_contrast_loss,
                target_CEloss_pseude,
            )
            tqdm_Target_loader.set_postfix_str(postfix)

        writer_train.add_scalar('train_loss', torch.mean(torch.stack(pseudo_loss_list)).item(),
                                global_step=epoch)
        writer_train.add_scalar('train_acc', np.mean(target_train_acc_list), global_step=epoch)
        writer_train.add_scalar('select_acc', np.mean(avg_temp_acc), global_step=epoch)
        writer_train.add_scalar('select_ratio', np.mean(avg_select_ratio), global_step=epoch)
        writer_train.add_scalar('ultimate_label_0', np.mean(ultimate_label_0_count_list), global_step=epoch )
        writer_train.add_scalar('ultimate_label_1', np.mean(ultimate_label_1_count_list), global_step=epoch)
        writer_train.add_scalar('init_label_0', np.mean(init_label_0_count_list), global_step=epoch)
        writer_train.add_scalar('init_label_1', np.mean(init_label_1_count_list), global_step=epoch)

        # =====================================================
        # ============== test for target data =================
        # =====================================================

        temp_test_accs, temp_test_losses, temp_test_Acc_all, temp_test_Acc1, temp_test_Prec1, temp_test_Recll1, temp_test_F1, temp_test_Acc2, temp_test_Prec2, temp_test_Recll2, temp_test_F2 = \
            [], [], [], [], [], [], [], [], [], [], []
        test_losses, test_accs = [], []
        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            test_loss, test_acc, test_pred = model_teacher(target_data=Batch_data, branch="target_test")
            temp_test_losses.append(test_loss.item())
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = evaluationclass(
                test_pred, Batch_data.y)
            temp_test_Acc_all.append(Acc_all), temp_test_Acc1.append(Acc1), temp_test_Prec1.append(
                Prec1), temp_test_Recll1.append(Recll1), temp_test_F1.append(F1), \
            temp_test_Acc2.append(Acc2), temp_test_Prec2.append(Prec2), temp_test_Recll2.append(
                Recll2), temp_test_F2.append(F2)
            temp_test_accs.append(test_acc)
        
        root_path = 'model/'+ dataname[4:] +'/training_results/'
        with open(root_path + run_time + ".txt", "a") as f:
                f.write(str(iter) +'\t'+str(epoch) +'\t' + str(np.mean(avg_temp_acc)) +'\t' + str(np.mean(avg_select_ratio)) +'\t' + str(np.mean(temp_test_Acc_all)) + '\t' + str(np.mean(temp_test_F1)) + '\t' + str(np.mean(temp_test_F2)) + '\n')
        
        writer_test.add_scalar('test_loss', np.mean(temp_test_losses), global_step=epoch )
        writer_test.add_scalar('test_accs', np.mean(temp_test_accs), global_step=epoch )
        writer_test.add_scalar('test_Acc1', np.mean(temp_test_Acc1), global_step=epoch)
        writer_test.add_scalar('test_Acc2', np.mean(temp_test_Acc2), global_step=epoch )
        writer_test.add_scalar('test_Prec1', np.mean(temp_test_Prec1), global_step=epoch )
        writer_test.add_scalar('test_Prec2', np.mean(temp_test_Prec2), global_step=epoch )
        writer_test.add_scalar('test_Recll1', np.mean(temp_test_Recll1), global_step=epoch )
        writer_test.add_scalar('test_Recll2', np.mean(temp_test_Recll2), global_step=epoch )
        writer_test.add_scalar('test_F1', np.mean(temp_test_F1), global_step=epoch )
        writer_test.add_scalar('test_F2', np.mean(temp_test_F2), global_step=epoch )
        test_losses.append(np.mean(temp_test_losses))
        test_accs.append(np.mean(temp_test_accs))

        print("Epoch {:05d} | Test_Loss {:.4f}| Test_Accuracy {:.4f}".format(epoch, np.mean(temp_test_losses),
                                                                             np.mean(temp_test_accs)))

        res_test = ['acc:{:.4f}'.format(np.mean(temp_test_Acc_all)),
                    'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_test_Acc1), np.mean(temp_test_Prec1),
                                                            np.mean(temp_test_Recll1), np.mean(temp_test_F1)),
                    'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_test_Acc2), np.mean(temp_test_Prec2),
                                                            np.mean(temp_test_Recll2), np.mean(temp_test_F2))]
        print('Test results:', res_test)
        early_stopping(np.mean(temp_test_losses), np.mean(temp_test_Acc_all), np.mean(temp_test_Acc1),
                       np.mean(temp_test_Acc2), np.mean(temp_test_Prec1), np.mean(temp_test_Prec2),
                       np.mean(temp_test_Recll1), np.mean(temp_test_Recll2), np.mean(temp_test_F1),
                       np.mean(temp_test_F2),
                       model_teacher, 'BiGCN', str(iter)+dataname + '_' + run_time, epoch, dataname)
        accs = np.mean(temp_test_Acc_all)
        acc1 = np.mean(temp_test_Acc1)
        acc2 = np.mean(temp_test_Acc2)
        pre1 = np.mean(temp_test_Prec1)
        pre2 = np.mean(temp_test_Prec2)
        rec1 = np.mean(temp_test_Recll1)
        rec2 = np.mean(temp_test_Recll2)
        F1 = np.mean(temp_test_F1)
        F2 = np.mean(temp_test_F2)
        if epoch == n_epochs - 1:
            early_stopping.early_stop = True
        if early_stopping.early_stop:
            print("***Early stopping, source domain classifier has been converged.***")
            accs = early_stopping.accs
            acc1 = early_stopping.acc1
            acc2 = early_stopping.acc2
            pre1 = early_stopping.pre1
            pre2 = early_stopping.pre2
            rec1 = early_stopping.rec1
            rec2 = early_stopping.rec2
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            # break
            return accs, acc1, pre1, rec1, F1, acc2, pre2, rec2, F2
    # return accs, acc1, pre1, rec1, F1, acc2, pre2, rec2, F2

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=None, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=None, type=float, help='weight decay of optimizer')
    parser.add_argument('--patience', default=None, type=int, help='early stop patience')
    parser.add_argument('--epochs', default=None, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=None, type=int, help='training batch size')
    parser.add_argument('--TDdroprate', default=None, type=float, help='training epochs')
    parser.add_argument('--BUdroprate', default=None, type=float, help='training epochs')
    parser.add_argument('--noise_rate', default=None, type=float, help='training epochs')
    parser.add_argument('--datasetname', default=None, type=str, help='training epochs')
    parser.add_argument('--iterations', default=None, type=int, help='training epochs')
    parser.add_argument('--lThreshold', default=None, type=float, help='learning rate')
    parser.add_argument('--hThreshold', default=None, type=float, help='learning rate')
    parser.add_argument('--lWeight', default=None, type=float, help='learning rate')
    parser.add_argument('--hWeight', default=None, type=float, help='learning rate')
    parser.add_argument('--EMA_m', default=None, type=float, help='learning rate')
    parser.add_argument('--init_model_path', default=None, type=str, help='output root path')
    
    args, unparsed = parser.parse_known_args(args=[])
    config = get_config(args, 'target_train')

    return args, config
    

def main():
    args, config = parse_args()
    print(config)
    lr = config['hyper_parameters']['lr']
    patience = config['hyper_parameters']['patience']
    n_epochs = config['hyper_parameters']['EPOCHS']
    batchsize = config['DATA']['BATCH_SIZE']
    TDdroprate = config['hyper_parameters']['TDdroprate']
    BUdroprate = config['hyper_parameters']['BUdroprate']
    datasetname = config['hyper_parameters']['datasetname']
    iterations = config['hyper_parameters']['Iterations']
    lThreshold = config['hyper_parameters']['lThreshold']
    hThreshold = config['hyper_parameters']['hThreshold']
    lWeight = config['hyper_parameters']['lWeight']
    hWeight = config['hyper_parameters']['hWeight']
    EMA_m = config['hyper_parameters']['EMA_m']
    noise_rate = config['hyper_parameters']['noise_rate']
    init_model_path = config['hyper_parameters']['init_model_path']
    weight_decay = config['hyper_parameters']['WEIGHT_DECAY']
    model = "GCN"

    test_accs = []
    ACC1, ACC2, PRE1, PRE2, REC1, REC2, F1, F2 = [], [], [], [], [], [], [], []

    for iter in range(iterations):
        fold0_x_test, fold0_x_train, \
        fold1_x_test, fold1_x_train, \
        fold2_x_test, fold2_x_train, \
        fold3_x_test, fold3_x_train, \
        fold4_x_test, fold4_x_train, target_train = load5foldData(datasetname)
        treeDic = loadTree(datasetname)
        accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0 = train_AT(
            treeDic,
            fold0_x_test,
            fold0_x_train, target_train,
            TDdroprate, BUdroprate,noise_rate,
            lr, weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            0,
            lThreshold,
            hThreshold,
            lWeight,
            hWeight,
            EMA_m,
            init_model_path)
        accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 = train_AT(
            treeDic,
            fold1_x_test,
            fold1_x_train, target_train,
            TDdroprate, BUdroprate, noise_rate,lr,
            weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            1,
            lThreshold,
            hThreshold,
            lWeight,
            hWeight,
            EMA_m,
            init_model_path)
        accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 = train_AT(
            treeDic,
            fold2_x_test,
            fold2_x_train, target_train,
            TDdroprate, BUdroprate,noise_rate, lr,
            weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            2,
            lThreshold,
            hThreshold,
            lWeight,
            hWeight,
            EMA_m,
            init_model_path)
        accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3 = train_AT(
            treeDic,
            fold3_x_test,
            fold3_x_train, target_train,
            TDdroprate, BUdroprate, noise_rate,lr,
            weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            3,
            lThreshold,
            hThreshold,
            lWeight,
            hWeight,
            EMA_m,
            init_model_path)
        accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4 = train_AT(
            treeDic,
            fold4_x_test,
            fold4_x_train, target_train,
            TDdroprate, BUdroprate, noise_rate,lr,
            weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            4,
            lThreshold,
            hThreshold,
            lWeight,
            hWeight,
            EMA_m,
            init_model_path)
        test_accs.append((accs_0 + accs_1 + accs_2 + accs_3 + accs_4) / 5)
        ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
        ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4) / 5)
        PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
        PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
        REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
        REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
        F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
        F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    print("Twitter:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
          "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(sum(test_accs) / iterations, sum(ACC1) / iterations,
                                                                    sum(ACC2) / iterations, sum(PRE1) / iterations,
                                                                    sum(PRE2) / iterations,
                                                                    sum(REC1) / iterations, sum(REC2) / iterations,
                                                                    sum(F1) / iterations, sum(F2) / iterations))


if __name__ == '__main__':
    main()
