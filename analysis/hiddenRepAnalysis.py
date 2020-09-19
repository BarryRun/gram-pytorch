import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import sys
import pickle
import collections
import numpy as np
from diagnosisRepAnalysis import read_best_params, get_diagnosis_rep 
from collections import defaultdict

sys.path.append("..")
from dataLoader import load_data, padMatrix
from train import build_tree, calculate_dimSize


def patient_query(admissionFile, diagnosisFile, patientList, target_disease = [], isLabel=False):
    """根据患者的id，返回他的所有信息。输入是一个id的list，返回是信息的list（三维）。

    Args:
        admissionFile ([type]): [description]
        diagnosisFile ([type]): [description]
        patientList ([type]): [description]
        isLabel (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    pidAdmMap = {}
    admDateMap = {}
    infd = open(admissionFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        admId = int(tokens[2])
        admTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
        admDateMap[admId] = admTime
        if pid in pidAdmMap:
            pidAdmMap[pid].append(admId)
        else:
            pidAdmMap[pid] = [admId]
    infd.close()

    def convert_to_icd9(dxStr):
        if dxStr.startswith('E'):
            if len(dxStr) > 4:
                return dxStr[:4] + '.' + dxStr[4:]
            else:
                return dxStr
        else:
            if len(dxStr) > 3:
                return dxStr[:3] + '.' + dxStr[3:]
            else:
                return dxStr

    admDxMap = {}
    infd = open(diagnosisFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[2])
        dxStr = 'D_' + convert_to_icd9(tokens[4][1:-1])  
        if admId in admDxMap:
            admDxMap[admId].append(dxStr)
        else:
            admDxMap[admId] = [dxStr]
    infd.close()

    pidSeqMap = {}
    for pid, admIdList in pidAdmMap.items():
        # At least 2 visits for a patient
        if len(admIdList) < 2:
            continue
        # 可以看到列表的value是一个由时间与疾病list所组成的二元组
        sortedList = sorted([(admDateMap[admId], admDxMap[admId]) for admId in admIdList])
        pidSeqMap[pid] = sortedList

    # searching out target patients
    # if patientList:
    #     for p in pidSeqMap.keys():
    #         # 这里要求必须不是第一次的adm检查出来
    #         for v in pidSeqMap[p][1:]:
    #             if 'D_191.8' in v[1]:
    #                 print(p)

    # searching the target patients
    res = []      # res : patients * visits * diagnoses
    for patient in patientList:
        buf = []
        if patient in pidSeqMap.keys():
            for visit in pidSeqMap[patient]:
                buf.append(visit[1])
            res.append(buf)
        else:
            print("patient", patient, "is not in data")

    # search that which visit the target diseases is in for every patients selected
    if target_disease:
        label_site = [[] for _ in range(len(res))]
        for idx, _p in enumerate(res):
            for _idx, v in enumerate(_p):
                if target_disease in v:
                    label_site[idx].append(_idx)
        print(label_site)

    # 替换成编码
    if isLabel:
        types = pickle.load(open('../data/origin_gram/MIMIC_processed.types', 'rb'))
    else:
        types = pickle.load(open('../data/origin_gram/MIMIC_with_ancestors.types', 'rb'))

    for i in range(len(res)):
        for j in range(len(res[i])):
            for k in range(len(res[i][j])):
                res[i][j][k] = types[res[i][j][k]]

    return res


def get_hidden_rep(params, patients, target_disease=''):
    """Given patients and admissions, return it's hidden representation after GRU
    """
    # Get the diagnosis embedding after attention 
    emb = get_diagnosis_rep(params)

    # Get the target patients data
    patients_seqs = patient_query('../data/ADMISSIONS.csv','../data/DIAGNOSES_ICD.csv', patients, target_disease, isLabel=False)
    sorted_idx = sorted(range(len(patients_seqs)), key=lambda x: len(patients_seqs[x]))
    patients_seqs = [patients_seqs[i] for i in sorted_idx]
    lengths = np.array([len(patients_seqs) for seq in patients_seqs]) - 1
    n_samples = len(patients_seqs)
    maxlen = np.max(lengths)
    # inputDimSize = calculate_dimSize('../data/origin_gram/MIMIC_with_ancestors.seqs')
    inputDimSize = 4894
    x = np.zeros((maxlen, n_samples, inputDimSize), dtype=float)
    mask = np.zeros((maxlen, n_samples))
    for idx, seq in enumerate(patients_seqs):
        for xvec, subseq in zip(x[:, idx, :], seq[:-1]):
            xvec[subseq] = 1.
        mask[:lengths[idx], idx] = 1.
    lengths = np.array(lengths)

    # Get the admission representation after GRU
    x = torch.from_numpy(x).float().to('cuda:0')
    mask = torch.from_numpy(mask).float().to('cuda:0')
    x_emb = torch.tanh(torch.matmul(x, emb))
    # GRU parameters
    gru = torch.nn.GRU(input_size=128, hidden_size=128).to('cuda:0')
    gru.load_state_dict(collections.OrderedDict({'weight_ih_l0':params['gru.weight_ih_l0'], 'weight_hh_l0':params['gru.weight_hh_l0'], 'bias_ih_l0':params['gru.bias_ih_l0'], 'bias_hh_l0':params['gru.bias_hh_l0']}))
    hidden, _ = gru(x_emb)
    hidden = hidden * mask[:, :, None]
    return hidden


# def rep_analysis(rep):
#     """Do some analysis upon given representations of patents

#     Args:
#         rep ([type]): [description]
#     """    

if __name__ == "__main__":
    parameters = read_best_params('../output/origin_gram/output')
    target_patients = [7175, 13589, 16947, 42067, 93806, 27387]
    get_hidden_rep(parameters, target_patients, '')
