import numpy as np
import pickle


def load_data(seqFile, labelFile, timeFile='', test_ratio=0.15, valid_ratio=0.1):
    # 打开文件
    sequences = np.array(pickle.load(open(seqFile, 'rb')), dtype=object)
    labels = np.array(pickle.load(open(labelFile, 'rb')), dtype=object)
    if len(timeFile) > 0:
        times = np.array(pickle.load(open(timeFile, 'rb')), dtype=object)

    np.random.seed(4396)
    dataSize = len(labels)
    # 生成0到dataSize-1的随机排序序列
    ind = np.random.permutation(dataSize)
    nTest = int(test_ratio * dataSize)
    nValid = int(valid_ratio * dataSize)

    # 随机划分训练集、验证集和测试集，因为第一个维度代表每一个患者的数据
    test_indices = ind[:nTest]
    valid_indices = ind[nTest:nTest + nValid]
    train_indices = ind[nTest + nValid:]

    train_set_x = sequences[train_indices]
    train_set_y = labels[train_indices]
    test_set_x = sequences[test_indices]
    test_set_y = labels[test_indices]
    valid_set_x = sequences[valid_indices]
    valid_set_y = labels[valid_indices]
    train_set_t = None
    test_set_t = None
    valid_set_t = None

    if len(timeFile) > 0:
        train_set_t = times[train_indices]
        test_set_t = times[test_indices]
        valid_set_t = times[valid_indices]

    def len_argsort(seq):
        # 按照seq中每一项的长度进行排序
        # 也就是按每一个patient的visit的次数进行升序排序
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]

    if len(timeFile) > 0:
        train_set_t = [train_set_t[i] for i in train_sorted_index]
        valid_set_t = [valid_set_t[i] for i in valid_sorted_index]
        test_set_t = [test_set_t[i] for i in test_sorted_index]

    # 将训练、验证、测试数据分别打包
    train_set = (train_set_x, train_set_y, train_set_t)
    valid_set = (valid_set_x, valid_set_y, valid_set_t)
    test_set = (test_set_x, test_set_y, test_set_t)

    return train_set, valid_set, test_set


def padMatrix(seqs, labels, options):
    # length记录了每一个每一个患者visit的次数
    # 这里-1是因为要找maxlen，而x和y的次数都少一次
    lengths = np.array([len(seq) for seq in seqs]) - 1
    # 表示有n个患者的样本
    n_samples = len(seqs)
    # 最长的visit次数
    maxlen = np.max(lengths)

    # maxlen：最长的sequence，即最多的访问次数
    # n_samples：表示有n条患者的数据，实际实验中均为100
    # 注意这里的数据形式，改为了：visit维度 * patients维度 * 预定义的dimension
    x = np.zeros((maxlen, n_samples, options['inputDimSize']), dtype=float)
    y = np.zeros((maxlen, n_samples, options['numClass']), dtype=float)
    mask = np.zeros((maxlen, n_samples))

    # 对于seqs与labels中的每一个患者的visit序列
    for idx, (seq, lseq) in enumerate(zip(seqs, labels)):
        # 对于visit序列中除去第一个的所有visit
        # y与x形式相同，只是最终label的类别数目不一样
        for yvec, subseq in zip(y[:, idx, :], lseq[1:]):
            # 同样是标记
            yvec[subseq] = 1.

        # 对于visit序列中除去最后一个的所有visit
        # x[:, idx, :] 表示针对某个患者的输入，其形式为 最多visit次数*某个病人*inputDimSize,这里遍历是遍历每一个visit
        # 这里seq的形式为visits*icd_codes， seqs[-1]表示不关注最后一次visit
        # 这个for就相当于为每一个visit制作恰当的binary的输入
        # for xvec, subseq in zip(x[:, idx, :], seq[:-1]):
        for xvec, subseq in zip(x[:, idx, :], seq[:-1]):  # 这里修改seq为new_seq
            # subseq是一个疾病的list
            # 将该次visit中所有出现的疾病标为1
            # print(len(xvec)) 输出均为4894
            xvec[subseq] = 1.

        # 用mask矩阵来记录每个患者有多长得sequence
        mask[:lengths[idx], idx] = 1.

    lengths = np.array(lengths)

    # 总结一下，这里x的维度是 maxlen, n_samples=100, options['inputDimSize']=4894
    # y的维度是 maxlen, n_samples=100, options['numClass']=942
    return x, y, mask, lengths