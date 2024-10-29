import re

import numpy as np
from load_data import *
from CGABepi1 import * #configure

from sklearn.metrics import roc_auc_score, average_precision_score
np.seterr(divide='ignore', invalid='ignore')  #

USE_CUDA = torch.cuda.is_available()
random_seed = 0
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if USE_CUDA:
    torch.cuda.manual_seed(0)

def independent_test(independent_data_file_path, model_save_dir_name):
    MODEL_SAVE_PATH = '../models/' + model_save_dir_name + '/random_seed_0/'

    files = os.listdir(MODEL_SAVE_PATH)  # trans from 200 epoch
    # print('files', files)
    latest_files = []
    for i in range(5):
        fold_num = i + 1
        matched_file = []
        matched_file_tmp = []
        for file in files:
            matched_file_tmp.append(re.findall('validate_param_fold' + str(fold_num) + r'epoch.*', file))
        for item in matched_file_tmp:
            if item == []:
                continue
            else:
                matched_file.append(item[0])
        if len(matched_file) >= 2:
            mtime = os.path.getmtime(MODEL_SAVE_PATH + matched_file[0])
            latest_file_idx = 0
            for idx in range(1, len(matched_file)):
                if os.path.getmtime(MODEL_SAVE_PATH + matched_file[idx]) > mtime:
                    mtime = os.path.getmtime(MODEL_SAVE_PATH + matched_file[idx])
                    latest_file_idx = idx
            latest_files.append(MODEL_SAVE_PATH + matched_file[latest_file_idx])
        elif len(matched_file) == 1:
            latest_files.append(MODEL_SAVE_PATH + matched_file[0])
        else:
            continue

    # print('latest_files', latest_files)
    all_pred_pros_list = []

    ori_epi_seq_list = []
    in_f = open(independent_data_file_path, 'r')
    for line in in_f:
        ori_epi_seq_list.append(re.split('[\t\n]', line)[0])
    in_f.close()
    epi_seq_list, label_list = get_data_from_file(independent_data_file_path)
    for item in latest_files:
        independent_dataset = MyDataSet_distribute(epi_seq_list, label_list)
        independent_dataloader = tud.DataLoader(independent_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                num_workers=0)

        model_test = Network_conn()
        # print("CPU")
        if USE_CUDA:
            print('using cuda')
            model_test = model_test.cuda()

        item_name = re.split('[/]', item)[-1]
        fold = int(item_name[len('validate_param_fold')])
        pred_pros = test_EL_distribute_indep_test(model_test, independent_dataloader, fold, item, USE_CUDA, threshold)

        all_pred_pros_list.append(pred_pros)

    all_pred_pros_list_new = []
    for item_fold in all_pred_pros_list:
        for i, item in enumerate(item_fold):
            if len(all_pred_pros_list_new) <= i:
                all_pred_pros_list_new.append([item[0]])
            else:
                all_pred_pros_list_new[i].append(item[0])
    print(np.array(all_pred_pros_list_new).shape)

    pred_pros_mean = []
    for item in all_pred_pros_list_new:
        pred_pros_mean.append(sum(item) / len(item))

    return ori_epi_seq_list, pred_pros_mean



if __name__ == "__main__":

    # configure
    input_data_file_path = '../datasets/test2.txt'
    output_data_file_path = '../for_pred_output/test2_CGABepi1_netbce.txt'
    model_save_dir_name = 'CGABepi1_netbce'  # select models


    ori_epi_seq_list, pred_pros_mean = independent_test(input_data_file_path, model_save_dir_name)

    of = open(output_data_file_path, 'w')
    for i, seq in enumerate(ori_epi_seq_list):
        of.write(seq + '\t' + str(pred_pros_mean[i]) + '\n')
    of.close()
