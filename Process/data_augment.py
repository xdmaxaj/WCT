import torch
import math
from collections import OrderedDict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def strong_augment(batch_data, batch_size, noise_rate=0.2):
    feature = batch_data.x.clone()
    total_index = 0
    for num_batch in range(batch_size):
        index = (torch.eq(batch_data.batch, num_batch))
        current_sample = batch_data.x[index]
        sample_weight = torch.ones(current_sample.size()[0])
        sample_weight[0] = 0
        sampled_indices = torch.multinomial(sample_weight, num_samples=math.ceil(current_sample.size()[0] * 0.2),
                                            replacement=False)
        select_node_vec = current_sample[sampled_indices]
        noise = torch.normal(0, 1, size=select_node_vec.size())
        feature[total_index + sampled_indices] = torch.add(feature[index][sampled_indices], noise)
        index_num = index.int().sum()
        total_index += index_num
    return feature


def update_teacher_model(model_teacher, model, keep_rate=0.996):
    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in list(model.state_dict().keys()):
            new_teacher_dict[key] = (
                    model.state_dict()[key] *
                    (1 - keep_rate) + value * keep_rate
            )
    model_teacher.load_state_dict(new_teacher_dict)
    return model_teacher


def process_pseudo_label(prob, threshold=0.8):
    confidence, P_label = prob.max(dim=1)
    indices = torch.where(confidence > threshold)
    det_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    for i in det_list:
        if len(indices[0]) == 0:
            indices = torch.where(confidence > confidence.max() - i)
        if len(indices[0]) > 0:
            break
    P_label_select = P_label[indices]
    beta_label_1_count = P_label_select.sum(dim=0).item()
    beta_label_0_count = len(P_label_select) - beta_label_1_count
    return P_label_select, indices, beta_label_1_count, beta_label_0_count


def balance_process_pseudo_label(batch_size, prob, threshold=0.8):
    confidence, P_label = prob.max(dim=1)
#     print(confidence)
    indices = torch.where(confidence > threshold)
    det_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    for i in det_list:
        if len(indices[0]) == 0:
            indices = torch.where(confidence > confidence.max() - i)
            threshold = confidence.max() - i
        if len(indices[0]) > 0:
            break
    P_label_select = P_label[indices[0]]
    init_label_1_count = P_label_select.sum(dim=0).item()
    init_label_0_count = len(indices[0]) - init_label_1_count

    # ======================================
    # ==== count delta to balance model ====
    # ======================================
    class_1_count_vec = torch.ones_like(P_label_select)
    class_1_count_num = P_label_select.eq(class_1_count_vec)
    class_1_count_num = class_1_count_num.float().sum(dim=0)
    if class_1_count_num.item() == 0:
#         print('########## 1=0 ###########')
        class_1_count_num = torch.tensor(0.0)
        class_0_count_num = torch.tensor(len(P_label_select)).float()
    else:
        class_0_count_num = len(P_label_select) - class_1_count_num
        if class_0_count_num == 0:
#             print('########## 0=0 ###########')
            class_0_count_num = torch.tensor(0.0)
    class_0_count_num = class_0_count_num.unsqueeze(dim=0)
    class_1_count_num = class_1_count_num.unsqueeze(dim=0)
    class_0_count_num = class_0_count_num.to(device)
    class_1_count_num = class_1_count_num.to(device)
    
    temp_total_label_num = len(P_label_select)
    max_class_count_num, max_class = torch.max(torch.cat((class_0_count_num, class_1_count_num), dim=0), dim=0)
    temp_left_object_num = batch_size - temp_total_label_num
    denominator = max(max_class_count_num.item(), temp_left_object_num)
    delta_for_0 = (class_0_count_num / denominator).item()
    delta_for_1 = (class_1_count_num / denominator).item()
    denominator_for_loss = max_class_count_num.item()
    delta_for_0_loss = (class_0_count_num / denominator_for_loss).item()
    delta_for_1_loss = (class_1_count_num / denominator_for_loss).item()
    # ============ map function ============
    delta_map_0 = delta_for_0 / (2 - delta_for_0)
    delta_map_1 = delta_for_1 / (2 - delta_for_1)

    loss_delta_0 = (1 - delta_for_0_loss) / (delta_for_0_loss + 1)
    loss_delta_1 = (1 - delta_for_1_loss) / (delta_for_1_loss + 1)
    low_bound_loss = 1.0
    ultimate_weight_0 = low_bound_loss + loss_delta_0 * (3 - low_bound_loss)
    ultimate_weight_1 = low_bound_loss + loss_delta_1 * (3 - low_bound_loss)
#     print(ultimate_threshold_1)
    # ========= change threshold ===========
    lower_bound = 0.6
    if lower_bound > threshold:
        lower_bound = max(0.5, (threshold - 0.05))
        print('lower_bound > threshold')
    ultimate_threshold_0 = lower_bound + delta_map_0 * (threshold - lower_bound)
    ultimate_threshold_1 = lower_bound + delta_map_1 * (threshold - lower_bound)
    # ultimate_threshold_0 = max(threshold * delta_map_0, lower_bound)
    # ultimate_threshold_1 = max(threshold * delta_map_1, lower_bound)

    # 原本的这个系数只能小于1，相当于缩小阈值，能否改变公式，使其可以大于1，从而提高阈值，在模型训练的初始阶段，将某一类模型倾向性更强的类别阈值提高，另一类阈值降低。
    # 可以从如何评价模型的分类倾向性入手

    # ========= select new pseudo label ========
    confidence_0 = prob[:, 0]
    confidence_1 = prob[:, 1]
    indices_0 = torch.where(confidence_0 > ultimate_threshold_0)[0]
    indices_1 = torch.where(confidence_1 > ultimate_threshold_1)[0]
    ultimate_indices = torch.cat((indices_0, indices_1), dim=0).sort(dim=0)
    ultimate_P_label = P_label[ultimate_indices[0]]

    ultimate_label_1_count = ultimate_P_label.sum(dim=0).item()
    ultimate_label_0_count = len(ultimate_P_label) - ultimate_label_1_count
#     print(ultimate_threshold_0)
#     print(ultimate_threshold_1)
#     print('000')
#     print(confidence_0[indices_0])
#     print('111')
#     print(confidence_1[indices_1])
#     print(ultimate_indices)
    return ultimate_P_label, ultimate_indices, ultimate_label_1_count, ultimate_label_0_count, init_label_1_count, init_label_0_count,ultimate_weight_0, ultimate_weight_1

def confidence_mapping(P_confidence):
    confidence_constant_relu = torch.nn.ReLU()(P_confidence - 0.5)
    confidence_constant = torch.clamp(1 - torch.exp(-10 * confidence_constant_relu), min=0, max=1)
    return confidence_constant
