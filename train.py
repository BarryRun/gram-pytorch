import argparse
import pickle
import random
import time
import os
import torchsnooper
import torch
import torch.nn as nn
import numpy as np
from model.gram import GRAM
from dataLoader import load_data, padMatrix
from loss import CrossEntropy
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 0
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def build_tree(treeFile):
    treeMap = pickle.load(open(treeFile, 'rb'))
    # 形成一个矩阵，每一行代表着第一个元素的祖先的序列
    # shape形如疾病种数*祖先个数
    ancestors = np.array(list(treeMap.values())).astype('int32')

    # ancSize记录为祖先的个数
    ancSize = ancestors.shape[1]
    leaves = []
    for k in treeMap.keys():
        leaves.append([k] * ancSize)

    # leaves也是一个数组，大小与ancestors相同，均为 疾病个数*祖先个数
    # 但leaves每一行为ancSize个相同的值，均表示对应的icd码
    leaves = np.array(leaves).astype('int32')
    return leaves, ancestors


def train_GRAM(
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
    # 这里的leavesList， ancestorsList蕴含着每一个疾病的类别信息
    leavesList = []
    ancestorsList = []
    np_ancestorsList = []
    for i in range(5, 0, -1):
        leaves, ancestors = build_tree(treeFile + '.level' + str(i) + '.pk')
        # 设置为全局变量
        leavesList.append(leaves)
        ancestorsList.append(ancestors)

    print('Building the model ... ')
    gram = GRAM(inputDimSize, numAncestors, embDimSize, hiddenDimSize, attentionDimSize, numClass, dropoutRate, embFile)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     gram = nn.DataParallel(gram)
    gram.to(device)
    # gram.train()
    print(list(gram.state_dict()))
    loss_fn = CrossEntropy()
    loss_fn.to(device)

    print('Constructing the optimizer ... ')
    optimizer = torch.optim.Adadelta(gram.parameters(), lr=0.01, weight_decay=L2)

    print('Loading data ... ')
    trainSet, validSet, testSet = load_data(seqFile, labelFile, test_ratio=0.15, valid_ratio=0.1)
    print('Data length:', len(trainSet[0]))
    n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))
    val_batches = int(np.ceil(float(len(validSet[0])) / float(batchSize)))
    test_batches = int(np.ceil(float(len(testSet[0])) / float(batchSize)))

    print('Optimization start !!')
    # setting the tensorboard
    loss_writer = SummaryWriter('{}/{}'.format(outFile+'TbLog', 'Loss'))
    acc_writer = SummaryWriter('{}/{}'.format(outFile+'TbLog', 'Acc'))
    # test_writer = SummaryWriter('{}/{}'.format(outFile+'TbLog', 'Test'))

    logFile = outFile + '.log'
    bestTrainCost = 0.0
    bestValidCost = 100000.0
    bestTestCost = 0.0
    bestTrainAcc = 0.0
    bestValidAcc = 0.0
    bestTestAcc = 0.0
    epochDuration = 0.0
    bestEpoch = 0
    # set the random seed for test
    random.seed(seed)
    # with torchsnooper.snoop():
    for epoch in range(max_epochs):
        iteration = 0
        cost_vec = []
        acc_vec = []
        startTime = time.time()
        gram.train()
        for index in random.sample(range(n_batches), n_batches):
            optimizer.zero_grad()
            batchX = trainSet[0][index * batchSize:(index + 1) * batchSize]
            batchY = trainSet[1][index * batchSize:(index + 1) * batchSize]
            x, y, mask, lengths = padMatrix(batchX, batchY, options)
            x = torch.from_numpy(x).to(device).float()
            mask = torch.from_numpy(mask).to(device).float()
            # print('x,', x.size())
            y_hat = gram(x, mask, leavesList, ancestorsList)
            # print('y_hat', y_hat.size())
            y = torch.from_numpy(y).float().to(device)
            # print('y', y.size())
            lengths = torch.from_numpy(lengths).float().to(device)
            # print(y.size(), y_hat.size())
            loss, acc = loss_fn(y_hat, y, lengths)
            loss.backward()
            optimizer.step()
            if iteration % 100 == 0 and verbose:
                buf = 'Epoch:%d, Iteration:%d/%d, Train_Cost:%f, Train_Acc:%f' % (
                    epoch, iteration, n_batches, loss, acc)
                print(buf)
            cost_vec.append(loss.item())
            acc_vec.append(acc)
            iteration += 1
        duration_optimize = time.time() - startTime
        gram.eval()
        cost = np.mean(cost_vec)
        acc = np.mean(acc_vec)
        startTime = time.time()
        with torch.no_grad():
            # calculate the loss and acc of valid dataset
            cost_vec = []
            acc_vec = []
            for index in range(val_batches):
                validX = validSet[0][index * batchSize:(index + 1) * batchSize]
                validY = validSet[1][index * batchSize:(index + 1) * batchSize]
                val_x, val_y, mask, lengths = padMatrix(validX, validY, options)
                val_x = torch.from_numpy(val_x).float().to(device)
                mask = torch.from_numpy(mask).float().to(device)
                val_y_hat = gram(val_x, mask, leavesList, ancestorsList)
                val_y = torch.from_numpy(val_y).float().to(device)
                lengths = torch.from_numpy(lengths).float().to(device)
                valid_cost, valid_acc = loss_fn(val_y_hat, val_y, lengths)
                cost_vec.append(valid_cost.item())
                acc_vec.append(valid_acc)
            valid_cost = np.mean(cost_vec)
            valid_acc = np.mean(acc_vec)

            # calculate the loss and acc of test dataset
            cost_vec = []
            acc_vec = []
            for index in range(test_batches):
                testX = testSet[0][index * batchSize:(index + 1) * batchSize]
                testY = testSet[1][index * batchSize:(index + 1) * batchSize]
                test_x, test_y, mask, lengths = padMatrix(testX, testY, options)
                test_x = torch.from_numpy(test_x).float().to(device)
                mask = torch.from_numpy(mask).float().to(device)
                test_y_hat = gram(test_x, mask, leavesList, ancestorsList)
                test_y = torch.from_numpy(test_y).float().to(device)
                lengths = torch.from_numpy(lengths).float().to(device)
                test_cost, test_acc = loss_fn(test_y_hat, test_y, lengths)
                cost_vec.append(test_cost.item())
                acc_vec.append(test_acc)
            test_cost = np.mean(cost_vec)
            test_acc = np.mean(acc_vec)
        # record the loss and acc
        loss_writer.add_scalar('Train Loss', cost, epoch)
        loss_writer.add_scalar('Test Loss', test_cost, epoch)
        loss_writer.add_scalar('Valid Loss', valid_cost, epoch)
        acc_writer.add_scalar('Train Acc', acc, epoch)
        acc_writer.add_scalar('Test Acc', test_acc, epoch)
        acc_writer.add_scalar('Valid Acc', valid_acc, epoch)

        # print the loss
        duration_metric = time.time() - startTime
        buf = 'Epoch:%d, Train_Cost:%f, Valid_Cost:%f, Test_Cost:%f' % (
            epoch, cost, valid_cost, test_cost)
        print(buf)
        print2file(buf, logFile)
        buf = 'Train_Acc:%f, Valid_Acc:%f, Test_Acc:%f' % (acc, valid_acc, test_acc)
        print(buf)
        print2file(buf, logFile)
        buf = 'Optimize_Duration:%f, Metric_Duration:%f' % (duration_optimize, duration_metric)
        print(buf)
        print2file(buf, logFile)

        # save the best model
        if valid_cost < bestValidCost:
            bestValidCost = valid_cost
            bestTestCost = test_cost
            bestTrainCost = cost
            bestEpoch = epoch
            bestTrainAcc = acc
            bestValidAcc = valid_acc
            bestTestAcc = test_acc

        torch.save(gram.state_dict(), outFile + f'.{epoch}')

    buf = 'Best Epoch:%d, Avg_Duration:%f, Train_Cost:%f, Valid_Cost:%f, Test_Cost:%f' % (
        bestEpoch, epochDuration / max_epochs, bestTrainCost, bestValidCost, bestTestCost)
    print(buf)
    print2file(buf, logFile)
    buf = 'Train_Acc:%f, Valid_Acc:%f, Test_Acc:%f' % (bestTrainAcc, bestValidAcc, bestTestAcc)
    print(buf)
    print2file(buf, logFile)


def parse_arguments(parser):
    parser.add_argument('seq_file', type=str, metavar='<visit_file>',
                        help='The path to the Pickled file containing visit information of patients')
    parser.add_argument('label_file', type=str, metavar='<label_file>',
                        help='The path to the Pickled file containing label information of patients')
    parser.add_argument('tree_file', type=str, metavar='<tree_file>',
                        help='The path to the Pickled files containing the ancestor information of the input medical '
                             'codes. Only use the prefix and exclude ".level#.pk".')
    parser.add_argument('out_file', metavar='<out_file>',
                        help='The path to the output models. The models will be saved after every epoch')
    parser.add_argument('--embed_file', type=str, default='',
                        help='The path to the Pickled file containing the representation vectors of medical codes. If '
                             'you are not using medical code representations, do not use this option')
    parser.add_argument('--embed_size', type=int, default=128,
                        help='The dimension size of the visit embedding. If you are providing your own medical code '
                             'vectors, this value will be automatically decided. (default value: 128)')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='The dimension size of the hidden layer of the GRU (default value: 128)')
    parser.add_argument('--attention_size', type=int, default=100,
                        help='The dimension size of hidden layer of the MLP that generates the attention weights ('
                             'default value: 128)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='The size of a single mini-batch (default value: 100)')
    parser.add_argument('--n_epochs', type=int, default=100, help='The number of training epochs (default value: 100)')
    parser.add_argument('--L2', type=float, default=0.0001,
                        help='L2 regularization coefficient for all weights except RNN (default value: 0.0001)')
    parser.add_argument('--dropout_rate', type=float, default=0.6,
                        help='Dropout rate used for the hidden layer of RNN (default value: 0.6)')
    parser.add_argument('--log_eps', type=float, default=1e-8,
                        help='A small value to prevent log(0) (default value: 1e-8)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print output after every 100 mini-batches (default True)')
    args = parser.parse_args()
    return args


def calculate_dimSize(seqFile):
    seqs = pickle.load(open(seqFile, 'rb'))
    codeSet = set()
    for patient in seqs:
        for visit in patient:
            for code in visit:
                codeSet.add(code)
    return max(codeSet) + 1


def get_rootCode(treeFile):
    tree = pickle.load(open(treeFile, 'rb'))
    rootCode = list(tree.values())[0][1]
    return rootCode


def print2file(buf, outFile):
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    # 计算icd_code的维度，即统计seq中出现的所有code的数量！！！
    inputDimSize = calculate_dimSize(args.seq_file)
    print('inputDimSize:%d' % inputDimSize)

    # 计算输出类别的维度，即统计label_file中出现的所有code的数量
    numClass = calculate_dimSize(args.label_file)
    print('numClass:%d' % numClass)

    # 获取rootCode的代码，表示总的ancestors的种类数
    numAncestors = get_rootCode(args.tree_file + '.level2.pk') - inputDimSize + 1

    train_GRAM(
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
