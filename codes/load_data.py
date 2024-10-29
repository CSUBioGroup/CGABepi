import re
import os



def get_data_from_file(data_file_path):
    epi_seq_list, epi_seq_list_tmp, label_list = [], [], []

    in_f = open(data_file_path, 'r')
    for line in in_f:
        cols = re.split('[\t\n]', line)
        epi_seq_list_tmp.append(cols[0])
        label_list.append(1)
    in_f.close()

    # 裁剪cdr3
    for item in epi_seq_list_tmp:
        if len(item) >= 25:
            pseq_cdr3_seq = item[0:25]
        else:
            pseq_cdr3_seq = item + 'X' * (25 - len(item))
        epi_seq_list.append(pseq_cdr3_seq)

    return epi_seq_list, label_list


