import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
sys.path.append("..")
from diagnosisRepAnalysis import read_best_params
from train import build_tree


def get_attention(params):
    """Get all attention for every diagnosis

    Args:
        params ([OrderedDict]): [Paramters from trained model]

    Returns:
        [list, list]: [List of ancestors and attentions]
    """
    # reading the tree file
    leaves_list = []
    ancestors_list = []
    for i in range(5, 0, -1):
        leaves, ancestors = build_tree('../data/origin_gram/MIMIC_with_ancestors.level' + str(i) + '.pk')
        leaves_list.append(leaves)
        ancestors_list.append(ancestors)
    print('All parameters:', params.keys())
    w_emb = params['W_emb']
    w_attention = params['attentionLayer.W_attention']
    v_attention = params['attentionLayer.v_attention']
    b_attention = params['attentionLayer.b_attention']
    # getting attention after softmax for each diagnosis
    anc_list = []
    att_list = []
    for leaves, ancestors in zip(leaves_list, ancestors_list):
        if len(leaves.shape) == 2:
            leaves = leaves[np.newaxis, :, :]
            ancestors = ancestors[np.newaxis, :, :]
        leave_tmp = w_emb[leaves]
        anc_tmp = w_emb[ancestors]
        attention_input = torch.cat((leave_tmp, anc_tmp), 2)
        tmp = torch.matmul(attention_input, w_attention)
        tmp = tmp + b_attention
        mlp_output = torch.tanh(tmp)
        pre_attention = torch.matmul(mlp_output, v_attention)
        temp_attention = F.softmax(pre_attention, dim=1)
        # We can know current disease and their ancestors from ancestors
        # Also we can know the weight for the disease itself from the temp_attention
        # print(ancestors[0])
        # print(temp_attention.size())
        anc_list.append(ancestors[0])
        att_list.append(temp_attention)
    return anc_list, att_list


def attention_analysis(anc_list, att_list):
    # read the type file
    types = pickle.load(open('../data/MIMIC_with_ancestors.types', 'rb'))
    # count all disease number
    disease_num = []
    with open('../output/origin_gram/output_all_acc.txt', 'r') as f:
        lines = f.readlines()
        # data example
        # TypeNum:72, icd_code:D_428.0, Count:4581, avg_rank:5.822091, Accuracy:0.967693
        for line in lines:
            buf = line.strip().split(', ')
            icd_code = buf[1].split(':')[1]
            count = buf[2].split(':')[1]
            disease_num.append((types[icd_code], count))
    disease_num.reverse()

    # split the disease into 5 parts
    disease_divided = [[] for _ in range(10)]
    disease_divided_num = [[] for _ in range(10)]
    batch = len(disease_num) // 10
    for idx, item in enumerate(disease_num):
        for i in range(10):
            if i*batch <= idx < (i+1)*batch:
                disease_divided[i].append(item[0])
                disease_divided_num[i].append(float(item[1]))

    # Get the attention for first 0~10% rare-diseases
    all_att = [[] for _ in range(10)]
    num_disease = [0] * 10
    for anc,att in zip(anc_list, att_list):
        for each_anc, each_att in zip(anc, att):
            disease = each_anc[0]
            for i in range(10):
                if disease in disease_divided[i]:
                    num_disease[i] += 1
                    all_att[i].append(each_att[0].item())
    for i in range(10):
        all_att[i].sort()
        print('part ' + str(i*10) + '-' + str((i+1)*10))
        print('middle num: ', all_att[i][len(all_att[i])//2])   # middle num
        print('average num: ', sum(all_att[i])/num_disease[i])   # average num
        # Draw the distribution of diseases
        distribution = [0] * 10
        for att in all_att[i]:
            distribution[int((att-0.0000001)//0.1)] += 1


if __name__ == "__main__":
    parameters = read_best_params('../output/origin_gram/output')
    trees, weights = get_attention(parameters)
    attention_analysis(trees, weights)

