import os
import json

def cal_num_ood_acc(test_data_num, acc):
    true_test_num = 0
    total_num = 0
    line_acc = []
    for i, line in enumerate(acc):
        true_test_num_line = 0
        total_num_line = 0
        for j, number in enumerate(line):
            if i != j:
                true_test_num += int(number * test_data_num[i])
                total_num += 100 * test_data_num[i]
                true_test_num_line += int(number * test_data_num[i])
                total_num_line += 100 * test_data_num[i]
        line_acc.append(true_test_num_line / total_num_line)
    total_acc = true_test_num / total_num
    return line_acc, total_acc

def cal_num_ind_acc(test_data_num, acc):
    true_test_num = 0
    total_num = 0
    line_acc = []
    for i, line in enumerate(acc):
        true_test_num_line = 0
        total_num_line = 0
        for j, number in enumerate(line):
            if i == j:
                true_test_num += int(number * test_data_num[i])
                total_num += 100 * test_data_num[i]
                true_test_num_line += int(number * test_data_num[i])
                total_num_line += 100 * test_data_num[i]
        line_acc.append(true_test_num_line / total_num_line)
    total_acc = true_test_num / total_num
    return line_acc, total_acc

def read_results(file_name, s=3, line_num=-1, data=None):
    print(file_name)
    data_list = []
    with open(file_name) as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)
    result = data_list[line_num]
    if s == 3:
        test_data_num = [1118, 1118, 1118, 2189, 2189, 2189, 4566, 4566, 4566, 1944, 1944, 1944, 4680, 4680, 4680, 1306, 1306, 1306]
    elif s == 4:
        test_data_num = [844, 844, 844, 844, 1647, 1647, 1647, 1647, 3432, 3432, 3432, 3432, 1464, 1464, 1464, 1464, 3520, 3520, 3520, 3520, 983, 983, 983, 983]
    elif s == 5:
        test_data_num = [677, 677, 677, 677, 677, 1320, 1320, 1320, 1320, 1320, 2748, 2748, 2748, 2748, 2748, 1172, 1172, 1172, 1172, 1172, 2800, 2800, 2800, 2800, 2800, 792, 792, 792, 792, 792]


    total_test_data_num = sum(test_data_num)

    acc = result['acc']
    # print('accuracy:')
    # print(acc)

    line_acc, total_num_ood_acc = cal_num_ood_acc(test_data_num, acc)
    # print('the ood accuracy of each class(by data number):')
    # print(line_acc)
    # print('the total ood accuracy(by data number):')
    # print(total_num_ood_acc)

    line_acc, total_num_ind_acc = cal_num_ind_acc(test_data_num, acc)
    # print('the ind accuracy of each class(by data number):')
    # print(line_acc)
    # print('the total ind accuracy(by data number):')
    # print(total_num_ind_acc)
    rounds = result['round']
    # print('round:')
    # print(rounds)
    times = result['total_train_time']
    # print('times:')
    # print(times)
    return total_num_ood_acc, total_num_ind_acc, rounds, times

import pandas as pd
methods = ['local','fedavg','fedprox', 'fedditto','fedmoon','fedproto','fedavgDBE']
latex_methods = ['local','FedAvg','FedProx','Ditto','MOON','FedProto','DBE']
split = 3

ood_accs, ind_accs, rounds = [], [], []
for m in methods:
    file_name = f'./{m}/ViT-B-32_domainnet_sub50_split{split}.json'
    ood_acc, ind_acc, round_, time = read_results(file_name, s=split, line_num=-2)
    ood_accs.append(round(ood_acc,4))
    ind_accs.append(round(ind_acc,4))
    rounds.append(round_)
table3 = pd.DataFrame({'method':latex_methods, 'ood_acc_3':ood_accs, 'ind_acc_3':ind_accs, 'rounds_3':rounds})

split = 4
ood_accs, ind_accs, rounds = [], [], []
for m in methods:
    file_name = f'./{m}/ViT-B-32_domainnet_sub50_split{split}.json'
    ood_acc, ind_acc, round_, time = read_results(file_name, s=split, line_num=-2)
    ood_accs.append(round(ood_acc,4))
    ind_accs.append(round(ind_acc,4))
    rounds.append(round_)
table4 = pd.DataFrame({'ood_acc_4':ood_accs, 'ind_acc_4':ind_accs, 'rounds_4':rounds})

split = 5
ood_accs, ind_accs, rounds = [], [], []
for m in methods:
    file_name = f'./{m}/ViT-B-32_domainnet_sub50_split{split}.json'
    ood_acc, ind_acc, round_, time = read_results(file_name, s=split, line_num=-2)
    ood_accs.append(round(ood_acc,4))
    ind_accs.append(round(ind_acc,4))
    rounds.append(round_)
table5 = pd.DataFrame({'ood_acc_5':ood_accs, 'ind_acc_5':ind_accs, 'rounds_5':rounds})

# horizontal concat tables, merge same columns with method
table = pd.concat([table3, table4, table5], axis=1)
# 只保留四位小数
table.to_latex('domainnet_sub50.tex', index=False, float_format="%.4f")