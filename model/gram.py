import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper

seed = 4396
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_embedding(embFile):
    m = np.load(embFile)
    w = (m['w'] + m['w_tilde']) / 2.0
    return w


class Attention(nn.Module):
    def __init__(self, embDimSize, attentionDimSize):
        super(Attention, self).__init__()
        # attention parameters
        self.W_attention = nn.Parameter(torch.rand(embDimSize * 2, attentionDimSize, requires_grad=True))
        self.b_attention = nn.Parameter(torch.zeros(attentionDimSize, requires_grad=True))
        self.v_attention = nn.Parameter(torch.rand(attentionDimSize, requires_grad=True))

    # @torchsnooper.snoop()
    def forward(self, W_emb, leavesList, ancestorsList):
        embList = []
        for leaves, ancestors in zip(leavesList, ancestorsList):
            if len(leaves.shape) == 2:
                leaves = leaves[np.newaxis, :, :]
                ancestors = ancestors[np.newaxis, :, :]
            # leave_tmp.size: data_num_of_this_ancestor_num * ancestor_num * embed
            leave_tmp = W_emb[leaves]
            anc_tmp = W_emb[ancestors]
            attentionInput = torch.cat((leave_tmp, anc_tmp), 2)
            tmp = torch.matmul(attentionInput, self.W_attention)
            # print(self.b_attention)
            tmp = tmp + self.b_attention
            mlpOutput = torch.tanh(tmp)
            preAttention = torch.matmul(mlpOutput, self.v_attention)
            tempAttention = F.softmax(preAttention, dim=1)
            tempEmb = (W_emb[ancestors] * tempAttention[:, :, None]).sum(axis=1)
            embList.append(tempEmb)
        return embList


class GRAM(nn.Module):
    def __init__(self, inputDimSize, numAncestors, embDimSize, hiddenDimSize, attentionDimSize, numClass, dropout_rate, embFile=""):
        super(GRAM, self).__init__()
        # initial embedding
        if len(embFile) > 0:
            self.W_emb = nn.Parameter(torch.from_numpy(load_embedding(embFile)), requires_grad=True)
        else:
            self.W_emb = nn.Parameter(torch.rand((inputDimSize + numAncestors), embDimSize), requires_grad=True)

        # attention layer
        self.attentionLayer = Attention(embDimSize, attentionDimSize)

        # GRU parameters
        self.gru = torch.nn.GRU(input_size=embDimSize, hidden_size=hiddenDimSize)
        self.dropout_rate = dropout_rate
        # if not hasattr(self, '_flattened'):
        #     self.gru.flatten_parameters()
        #     setattr(self, '_flattened', True)
        # self.gru = torch.nn.GRU(input_size=embDimSize, hidden_size=hiddenDimSize, dropout=0.6)
        self.W_output = nn.Parameter(torch.rand(hiddenDimSize, numClass), requires_grad=True)
        self.b_output = nn.Parameter(torch.zeros(numClass), requires_grad=True)

    # @torchsnooper.snoop()
    def forward(self, x, mask, leavesList, ancestorsList):
        embList = self.attentionLayer(self.W_emb, leavesList, ancestorsList)
        emb = torch.cat(embList, 0)
        tpm_emb = torch.matmul(x, emb)
        x_emb = torch.tanh(tpm_emb)
        hidden, hn = self.gru(x_emb)
        # dropout
        hidden = F.dropout(hidden, p=self.dropout_rate)

        # softmax layer
        nom = torch.exp(torch.matmul(hidden, self.W_output) + self.b_output)
        denom = torch.sum(nom, 2, keepdim=True)
        output = nom / denom
        y = output * mask[:, :, None]
        return y


if __name__ == '__main__':
    gram = GRAM(100, 100, 100, 100, 100, 100)
    att = Attention(100, 100)
    print(list(gram.state_dict()))
    print(list(att.state_dict()))
