import numpy as np
import torch
import torch.utils.data as tud
from utils import *
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve

import esm

# 不在20个碱基内的用X表示
aa = {"C": 0, "S": 1, "T": 2, "P": 3, "A": 4, "G": 5, "N": 6, "D": 7, "E": 8, "Q": 9, "H": 10, "R": 11, "K": 12,
      "M": 13, "I": 14, "L": 15, "V": 16, "F": 17, "Y": 18, "W": 19}

aa_blosum50={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}

AAfea_phy_dict = get_AAfea_phy()
blosum50_matrix = blosum50()
blosum62_matrix = blosum62()

embedding_dict = get_embedding()

# 辅助 encode_seq_list 函数
# 将一条序列中的字母编码后返回
def encode_seq(seq, ENCODING_TYPE):
    encoded_seq = []
    if ENCODING_TYPE == 'AAfea_phy_BLOSUM62':
        for residue in seq:
            encoded_residue_tmp2 = []
            encoded_residue_tmp = []
            if residue not in AAfea_phy_dict.keys():
                for i in range(28):
                    encoded_residue_tmp2.append(0)
            else:
                encoded_residue_tmp2 = AAfea_phy_dict[residue]
            if residue not in aa.keys():
                for i in range(20):
                    encoded_residue_tmp.append(0)
            else:
                residue_idx = aa[residue]
                encoded_residue_tmp = blosum62_matrix[residue_idx]
            encoded_residue = encoded_residue_tmp2 + encoded_residue_tmp
                # print(len(blosum62_matrix[residue_idx]))
            # print(str(len(encoded_residue)))
            encoded_seq.append(encoded_residue)
    elif ENCODING_TYPE == 'AAfea_phy':
        for residue in seq:
            if residue not in AAfea_phy_dict.keys():
                encoded_residue = []
                for i in range(28):
                    encoded_residue.append(0)
            else:
                encoded_residue = AAfea_phy_dict[residue]
            encoded_seq.append(encoded_residue)

    elif ENCODING_TYPE == 'encoded':
        encoded_seq = seq

    elif ENCODING_TYPE == 'num':
        for residue in seq:
            if residue in aa.keys():
                encoded_seq.append(aa[residue])
            else:
                encoded_seq.append(20)
    # 循环编码一条序列中的字符
    elif ENCODING_TYPE == 'one-hot':
        for residue in seq:
            encoded_residue = []
            if residue not in aa.keys():
                for i in range(20):
                    encoded_residue.append(0)
            else:
                residue_idx = aa[residue]
                for i in range(20):
                    if i == residue_idx:
                        encoded_residue.append(1)
                    else:
                        encoded_residue.append(0)
            encoded_seq.append(encoded_residue)
    elif ENCODING_TYPE == 'BLOSUM50':
        for residue in seq:
            if residue not in aa_blosum50.keys():
                encoded_residue = []
                for i in range(20):
                    encoded_residue.append(0)
            else:
                residue_idx = aa_blosum50[residue]
                encoded_residue = blosum50_matrix[residue_idx]
            encoded_seq.append(encoded_residue)
    elif ENCODING_TYPE == 'BLOSUM62':
        for residue in seq:
            if residue not in aa.keys():
                encoded_residue = []
                for i in range(20):
                    encoded_residue.append(0)
            else:
                residue_idx = aa[residue]
                encoded_residue = blosum62_matrix[residue_idx]
            encoded_seq.append(encoded_residue)
    # 使用场景：直接把预训练的embedding参数转移到这里进行序列编码，embedding在这里的模型不在学习的情况
    # 如果想继续在训练时继续调整embedding的参数，那么不能用这种方式
    elif ENCODING_TYPE == 'embedding':
        for residue in seq:
            if residue not in aa.keys():
                encoded_residue = embedding_dict['X']
            else:
                encoded_residue = embedding_dict[residue]
            encoded_seq.append(encoded_residue) # [len(residue), 6]
    else:
        print("wrong ENCODING_TYPE!")
    return encoded_seq


# 将序列列表中的字母编码后返回
def encode_seq_list(seq_list, ENCODING_TYPE):
    encoded_seq_list = []
    for seq in seq_list:
        encoded_seq_list.append(encode_seq(seq, ENCODING_TYPE))
    return encoded_seq_list


class MyDataSet_distribute(tud.Dataset):
    def __init__(self, train_peps, train_labels):
        super(MyDataSet_distribute, self).__init__()

        ENCODING_TYPE_PEP3 = 'AAfea_phy'

        encoded_train_peps3 = encode_seq_list(train_peps, ENCODING_TYPE_PEP3)
        self.encoded_peps3 = torch.Tensor(encoded_train_peps3).float()
        self.labels = torch.Tensor(train_labels).reshape(-1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.encoded_peps3[index], self.labels[index]


# 独立测试集
def test_EL_distribute_indep_test(model_test, independent_dataloader, fold, best_model_name, USE_CUDA, threshold):
    model_test.load_state_dict(torch.load(best_model_name))
    real_labels = []
    pred_pros = []
    predict_labels = []

    for i, (X1, test_labels) in enumerate(independent_dataloader):
        if USE_CUDA:
            X1 = X1.cuda()
            test_labels = test_labels.cuda()

        model_test.eval()
        output = model_test(X1)
        output_list = output.cpu().detach().numpy().tolist()
        output_class = []
        for item in output_list:
            if float(item[0]) > threshold:
                output_class.append([1])
            else:
                output_class.append([0])

        real_labels += test_labels.cpu().tolist()
        pred_pros += output.cpu().tolist()
        predict_labels += output_class

    return pred_pros
