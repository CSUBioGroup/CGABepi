import os
import csv
import re
import random
from utils import *

def get_epitope1d_train_data():
    lines = []
    in_f = open('../ref_tools_data/epitope1D/epitope1d_dataset_cdhit95_training.csv', 'r')
    csv_reader = csv.reader(in_f)
    line_num = 0
    for row in csv_reader:
        line_num += 1
        if line_num == 1:
            continue
        label = '1' if row[1] == 'epitope' else '0'
        if len(row[0]) < 5 or len(row[0]) > 25:
            continue
        lines.append(row[0] + '\t' + label + '\n')
    in_f.close()

    random.seed(0)
    random.shuffle(lines)

    of = open('../datasets/train_epitope1d.txt', 'w')
    for line in lines:
        of.write(line)
    of.close()

def get_NetBCE_train_data():
    lines = []
    sequences = read_fasta('../ref_tools_data/NetBCE/data/training dataset.txt')
    # print(sequences[0])
    for item in sequences:
        pep_all = item['sequence']
        fasta_name = item['id']
        pep = re.split('[-]', pep_all)[0]
        label = re.split('[_]', fasta_name)[3]
        if len(pep) < 5 or len(pep) > 25:
            continue
        lines.append(pep + '\t' + label + '\n')

    random.seed(0)
    random.shuffle(lines)

    of = open('../datasets/train_NetBCE.txt', 'w')
    for line in lines:
        of.write(line)
    of.close()

if __name__ == "__main__":
    # get_epitope1d_train_data()
    get_NetBCE_train_data()