import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper
import argparse
import numpy as np
import pickle
from collections import defaultdict, OrderedDict
from model.gram import GRAM
from train import build_tree, calculate_dimSize, get_rootCode, parse_arguments, load_data, padMatrix, print2file
from loss import CrossEntropy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_whole_data(
        seqFile='seqFile.txt',
        labelFile='labelFile.txt',
        treeFile='tree.txt',
        embFile='embFile.txt',
        outFile='out.txt',
        inputDimSize=100,
        numAncestors=100,
        embDimSize=100,
        hiddenDimSize=200,
        attentionDimSize=200,
        max_epochs=100,
        L2=0.,
        numClass=26679,
        batchSize=100,
        dropoutRate=0.5,
        logEps=1e-8,
        verbose=True,
        ignore_level=0
    ):
    options = locals().copy()
    # get the best model through log
    # with open(outFile+'.log') as f:
    #     line = f.readlines()[-2]
    #     best_epoch = line.split(',')[0].split(':')[1]
    #     print('Best parameters occur epoch:', best_epoch)

    leavesList = []
    ancestorsList = []
    for i in range(5, 0, -1):
        leaves, ancestors = build_tree(treeFile + '.level' + str(i) + '.pk')
        leavesList.append(leaves)
        ancestorsList.append(ancestors)

    print('Loading the model ... ')
    # create the model
    gram = GRAM(inputDimSize, numAncestors, embDimSize, hiddenDimSize, attentionDimSize, numClass, dropoutRate, '').to(device)
    # read the best parameters
    # gram.load_state_dict(torch.load(outFile + '.' + best_epoch))
    gram.load_state_dict(torch.load(embFile))
    loss_fn = CrossEntropy()
    loss_fn.to(device)

    print('Loading the data ... ')
    dataset, _, _ = load_data(seqFile, labelFile, test_ratio=0, valid_ratio=0)
    typeFile = labelFile.split('.seqs')[0] + '.types'
    types = pickle.load(open(typeFile, 'rb'))
    rTypes = dict([(v,u) for u,v in types.items()])
    print('Data length:', len(dataset[0]))
    n_batches = int(np.ceil(float(len(dataset[0])) / float(batchSize)))

    print('Calculating the result ...')
    cost_vec = []
    acc_vec = []
    num_for_each_disease = defaultdict(float)
    TP_for_each_disease = defaultdict(float)
    rank_for_each_disease = defaultdict(float)

    for index in range(n_batches):
        batchX = dataset[0][index * batchSize:(index + 1) * batchSize]
        batchY = dataset[1][index * batchSize:(index + 1) * batchSize]
        x, y, mask, lengths = padMatrix(batchX, batchY, options)
        x = torch.from_numpy(x).to(device).float()
        mask = torch.from_numpy(mask).to(device).float()
        y_hat = gram(x, mask, leavesList, ancestorsList)
        y = torch.from_numpy(y).float().to(device)
        lengths = torch.from_numpy(lengths).float().to(device)
        loss, acc = loss_fn(y_hat, y, lengths)
        cost_vec.append(loss.item())
        acc_vec.append(acc)

        # Calculating the accuracy for each disease
        y_sorted, indices = torch.sort(y_hat, dim=2, descending=True)
        # indices = indices[:, :, :20]
        for i, j, k in torch.nonzero(y, as_tuple=False):
            k = k.item()
            num_for_each_disease[k] += 1
            # search the rank for k
            m = torch.nonzero(indices[i][j]==k, as_tuple=False).view(-1).item()
            # calculate the top20 accuracy
            if m < 20:
                TP_for_each_disease[k] += 1
            rank_for_each_disease[k] += (m+1)

    cost = np.mean(cost_vec)
    acc = np.mean(acc_vec)
    print('Whole data average loss:%f, average accuracy@20:%f,' % (cost, acc))

    print('Recording the accuracy for each disease ...')
    acc_out_file = outFile + '_all_acc.txt'
    # sort the disease by num
    num_for_each_disease = OrderedDict(sorted(num_for_each_disease.items(), key=lambda item: item[1], reverse=True))
    for disease in num_for_each_disease.keys():
        d_acc = TP_for_each_disease[disease] / num_for_each_disease[disease]
        avg_rank = rank_for_each_disease[disease] / num_for_each_disease[disease]
        buf = 'TypeNum:%d, icd_code:%s, Count:%d, avg_rank:%f, Accuracy:%f' % \
              (disease, rTypes[disease], num_for_each_disease[disease], avg_rank, d_acc)
        print2file(buf, acc_out_file)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    inputDimSize = calculate_dimSize(args.seq_file)
    print('inputDimSize:%d' % inputDimSize)

    numClass = calculate_dimSize(args.label_file)
    print('numClass:%d' % numClass)

    numAncestors = get_rootCode(args.tree_file + '.level2.pk') - inputDimSize + 1

    test_whole_data(
        seqFile=args.seq_file,
        inputDimSize=inputDimSize,
        treeFile=args.tree_file,
        numAncestors=numAncestors,
        labelFile=args.label_file,
        numClass=numClass,
        outFile=args.out_file,
        embFile=args.embed_file,
        embDimSize=args.embed_size,
        hiddenDimSize=args.rnn_size,
        attentionDimSize=args.attention_size,
        batchSize=args.batch_size,
        max_epochs=args.n_epochs,
        L2=args.L2,
        dropoutRate=args.dropout_rate,
        logEps=args.log_eps,
        verbose=args.verbose
    )