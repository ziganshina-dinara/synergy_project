import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, rankdata
import json
import sys
import argparse
from statistics import mean
import time
from multiprocessing import Pool
import math
import rpy2.robjects as robjects
r = robjects.r
from rpy2.robjects import StrVector, FloatVector

#setting the expected parameters
def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument('-source', '--source_type_cell', type=str)
    parser.add_argument('-target', '--target_type_cell', type=str)
    #parser.add_argument('-signatures', '--path_to_file_with_signatures',default = 'DATA/CD_signatures_binary_42809.gmt', type = argparse.FileType())
    parser.add_argument('-dir_results', '--path_to_dir_save_results', default = 'DATA', type = str)
    parser.add_argument('-cd_signature_metadata', '--path_to_file_with_cd_signature_metadata',
                        default = 'DATA/CD_signature_metadata.csv', type = str)
    parser.add_argument('-drugs_metadata', '--path_to_file_with_drugs_metadata',
                        default='DATA/Drugs_metadata.csv', type=str)
    parser.add_argument('-intersect_cfm_l1000fwd', '--path_to_file_with_intersect_cfm_l1000fwd',
                        default='DATA/table_of_cell_conversion_and_chemicals_1.csv', type=str)

    return parser


def modernization(data_intersect_CFM_L1000FWD, data_Drugs_metadata, data_CD_signature_metadata):
    #для малых молекул из L1000FWD по cid найдем pert_id
    data_intersect_CFM_L1000FWD['pert_id of chemicals in L1000FWD'] = str()
    for i in range(data_intersect_CFM_L1000FWD.shape[0]):
        if data_intersect_CFM_L1000FWD.iloc[i, 8] != 'not molecules':
            for cid in str(data_intersect_CFM_L1000FWD.iloc[i, 8]).split(';'):
                if data_intersect_CFM_L1000FWD.iloc[i, 9] == str():
                    data_intersect_CFM_L1000FWD.iloc[i, 9] = ';'.join(list(data_Drugs_metadata[data_Drugs_metadata['pubchem_cid'] == cid].index))
                else:
                    data_intersect_CFM_L1000FWD.iloc[i, 9] = data_intersect_CFM_L1000FWD.iloc[i, 9] + ';' +  ';'.join(list(data_Drugs_metadata[data_Drugs_metadata['pubchem_cid'] == cid].index))
    #для малых молекул из L1000FWD по pert_id найдем sign_id
    data_intersect_CFM_L1000FWD['sign_id'] = str()
    for i in range(data_intersect_CFM_L1000FWD.shape[0]):
        if data_intersect_CFM_L1000FWD.iloc[i, 9] != str():
            for pert_id in data_intersect_CFM_L1000FWD.iloc[i, 9].split(';'):
                if data_intersect_CFM_L1000FWD.iloc[i, 10] == str():
                    data_intersect_CFM_L1000FWD.iloc[i, 10] = ';'.join(list(data_CD_signature_metadata[data_CD_signature_metadata['pert_id'] == pert_id].index))
                else:
                    data_intersect_CFM_L1000FWD.iloc[i, 10] = data_intersect_CFM_L1000FWD.iloc[i, 10] + ';' + ';'.join(list(data_CD_signature_metadata[data_CD_signature_metadata['pert_id'] == pert_id].index))
    return data_intersect_CFM_L1000FWD


def filter(str_source_type_cell, str_target_type_cell, data_intersect_CFM_L1000FWD):
    data_intersect_CFM_L1000FWD = data_intersect_CFM_L1000FWD[(data_intersect_CFM_L1000FWD['Source Cell Type'] == str_source_type_cell)
                                            & (data_intersect_CFM_L1000FWD['Target Cell Type'] == str_target_type_cell)]
    data_intersect_CFM_L1000FWD = data_intersect_CFM_L1000FWD[data_intersect_CFM_L1000FWD['cid of chemicals in L1000FWD'] !=
                                                              'not molecules']
    return data_intersect_CFM_L1000FWD


def select_pair_sign_id_from_same_protocol(number_protocol, data_intersect_CFM_L1000FWD):
    pair_sign_id_from_same_protocol = []
    list_sign_id_in_same_protocol = data_intersect_CFM_L1000FWD.iloc[number_protocol, 10].strip().split(';')
    list_sign_id_in_same_protocol = list(set(list_sign_id_in_same_protocol))
    for i in range(len(list_sign_id_in_same_protocol) - 1):
        for j in range(i + 1, len(list_sign_id_in_same_protocol)):
            list_pair = [list_sign_id_in_same_protocol[i], list_sign_id_in_same_protocol[j]]
            list_pair.sort()
            pair_sign_id_from_same_protocol.append(tuple(list_pair))
    return pair_sign_id_from_same_protocol


def select_pair_sign_id_from_different_protocol(first_number_protocol, second_number_protocol, data_intersect_CFM_L1000FWD, list_syn_pair_sign_id):
    list_pair_sign_id_from_different_protocol = []
    list_sign_id_in_first_protocol = list(set(data_intersect_CFM_L1000FWD.iloc[first_number_protocol, 10].split(';')))
    list_sign_id_in_second_protocol = list(set(data_intersect_CFM_L1000FWD.iloc[second_number_protocol, 10].split(';')))
    for sign_id_1 in list_sign_id_in_first_protocol:
        for sign_id_2 in list_sign_id_in_second_protocol:
            if sign_id_1 != sign_id_2 :
                if (not (sign_id_1, sign_id_2) in list_syn_pair_sign_id) & (not (sign_id_2, sign_id_1) in list_syn_pair_sign_id) & (not (sign_id_1, sign_id_2) in list_pair_sign_id_from_different_protocol) & (not (sign_id_2, sign_id_1) in list_pair_sign_id_from_different_protocol):
                    list_pair_sign_id_from_different_protocol.append((sign_id_1, sign_id_2))
    return list_pair_sign_id_from_different_protocol


def select_sign_id_from_protocol(number_protocol, data_intersect_CFM_L1000FWD):
    list_sign_id = list(set(data_intersect_CFM_L1000FWD.iloc[number_protocol, 10].strip().split(';')))
    return list_sign_id


def select_sign_id(data_intersect_CFM_L1000FWD, number_processes):
    # select all synegy pair of signature ids
    with Pool(processes=number_processes) as pool:
        results = pool.starmap(select_pair_sign_id_from_same_protocol,
                               [(i, data_intersect_CFM_L1000FWD) for i in range(data_intersect_CFM_L1000FWD.shape[0])])
    list_syn_pair_sign_id = []
    for sub_list_pair_sign_id in results:
        print(len(sub_list_pair_sign_id))
        list_syn_pair_sign_id += sub_list_pair_sign_id
    print('all length', len(list_syn_pair_sign_id))
    list_syn_pair_sign_id = list(set(list_syn_pair_sign_id))
    print('all length after remove', len(list_syn_pair_sign_id))

    # select all signature ids
    with Pool(processes=number_processes) as pool:
        results = pool.starmap(select_sign_id_from_protocol,
                               [(i, data_intersect_CFM_L1000FWD) for i in range(data_intersect_CFM_L1000FWD.shape[0])])
    list_sign_id = []
    for sub_list_sign_id in results:
        list_sign_id += sub_list_sign_id
    list_sign_id = list(set(list_sign_id))
    if str() in list_sign_id:
        list_sign_id.remove(str())
    print('all length all sign id', len(list_sign_id))

    d = {}
    d['sign_id_1'] = []
    d['sign_id_2'] = []
    d['class_label'] = []
    list_not_syn_pair_sign_id = []
    for i in range(len(list_sign_id)):
        for j in range(i):
            list_pair = [list_sign_id[i], list_sign_id[j]]
            list_pair.sort()
            list_pair = tuple(list_pair)
            d['sign_id_1'].append(list_pair[0])
            d['sign_id_2'].append(list_pair[1])
            if list_pair in list_syn_pair_sign_id:
                d['class_label'].append(1)
            else:
                d['class_label'].append(0)
                list_not_syn_pair_sign_id.append(list_pair)
    df = pd.DataFrame(d)

    return (list_sign_id, df)


def split_by_synergy(df_sign_id_pairs_with_labels, df_matrix, name_metric):
    list_score_synergy_pair = []
    list_score_not_synergy_pair = []
    df_sign_id_pairs_with_labels_metric = df_sign_id_pairs_with_labels.copy()
    df_sign_id_pairs_with_labels_metric[name_metric] = np.nan
    for i in range(df_sign_id_pairs_with_labels_metric.shape[0]):
        if list(df_matrix.index).index(df_sign_id_pairs_with_labels_metric.loc[i, 'sign_id_1']) < list(df_matrix.index).index(df_sign_id_pairs_with_labels_metric.loc[i, 'sign_id_2']):
            df_sign_id_pairs_with_labels_metric.loc[i, name_metric] = df_matrix.loc[df_sign_id_pairs_with_labels_metric.loc[i, 'sign_id_1'], df_sign_id_pairs_with_labels_metric.loc[i, 'sign_id_2']]
        else:
            df_sign_id_pairs_with_labels_metric.loc[i, name_metric] = df_matrix.loc[
                df_sign_id_pairs_with_labels_metric.loc[i, 'sign_id_2'], df_sign_id_pairs_with_labels_metric.loc[i, 'sign_id_1']]

        if df_sign_id_pairs_with_labels_metric.loc[i, 'class_label'] == 1:
            list_score_synergy_pair.append(df_sign_id_pairs_with_labels_metric.loc[i, name_metric])
        else:
            list_score_not_synergy_pair.append(df_sign_id_pairs_with_labels_metric.loc[i, name_metric])

    return (list_score_synergy_pair, list_score_not_synergy_pair, df_sign_id_pairs_with_labels_metric)


def draw(list_cos_dist_synergy_pair, list_cos_dist_not_synergy_pair, path_to_figure):
    plt.figure(figsize=(15, 10))
    snsplot = sns.distplot(list_cos_dist_not_synergy_pair, hist=True, kde=False, color='b', label='not synergy')
    snsplot = sns.distplot(list_cos_dist_synergy_pair, hist=True, kde=False, color = 'green', label = 'synergy')
    plt.legend()
    snsplot.set_xlabel('score')
    plt.show()
    fig = snsplot.get_figure()
    fig.savefig(path_to_figure)


def number_sign_in_protocol(data_intersect_CFM_L1000FWD):
    number_sign = []
    list_sign = []
    for i in range(data_intersect_CFM_L1000FWD.shape[0]):
        number_sign.append(len(set(data_intersect_CFM_L1000FWD.iloc[i, 10].split(';'))))
        list_sign = list_sign + list(set(data_intersect_CFM_L1000FWD.iloc[i, 10].split(';')))
    number_all_sign = len(set(list_sign))
    d = {'number_sign_protocol': number_sign, 'number_all_sign': number_all_sign }
    return d


def split_signatures(str_source_type_cell, str_target_type_cell, data_intersect_CFM_L1000FWD, data_Drugs_metadata, data_CD_signature_metadata, number_processes):
    data = modernization(data_intersect_CFM_L1000FWD, data_Drugs_metadata, data_CD_signature_metadata)
    data = filter(str_source_type_cell, str_target_type_cell, data)
    all_s, df_sign_id_pairs_with_labels_scores = select_sign_id(data, number_processes)
    return (all_s, df_sign_id_pairs_with_labels_scores)


def rank_pair_based_syn_score(df_sign_id_pairs_with_labels_scores, metric_name, ascending_type):
    df_sign_id_pairs_with_labels_scores = df_sign_id_pairs_with_labels_scores.sort_values(by=metric_name, ascending=ascending_type)
    df_sign_id_pairs_with_labels_scores = df_sign_id_pairs_with_labels_scores.reset_index()
    return df_sign_id_pairs_with_labels_scores


def count_synergy_pair_in_top50(df_sign_id_pairs_with_labels_scores_sorted):
    return sum(df_sign_id_pairs_with_labels_scores_sorted.loc[:49, 'class_label'])


def count_synergy_pair_in_top_5percent(df_sign_id_pairs_with_labels_scores_sorted):
    len_list_pair_signatures_5_percent = round(len(df_sign_id_pairs_with_labels_scores_sorted) * 0.05)
    number_syn_pair_in_top_5percent = sum(df_sign_id_pairs_with_labels_scores_sorted.loc[:len_list_pair_signatures_5_percent-1, 'class_label'])
    return (number_syn_pair_in_top_5percent, len_list_pair_signatures_5_percent,
            number_syn_pair_in_top_5percent / len_list_pair_signatures_5_percent)


def count_pairs(df_sign_id_pairs_with_labels_scores):
    return df_sign_id_pairs_with_labels_scores.shape[0]

def calculate_PSEA_metric(df, metric_name, path):

    df['score_for_fgsea'] = df[metric_name] - df[metric_name].mean()
    syn_pair_number_list = list(df[df['class_label'] == 1].index)
    syn_pair_number_list = [str(x + 1) for x in syn_pair_number_list]  # сдвиг  для индексации в R
    not_syn_pair_number_list = list(df[df['class_label'] == 0].index)
    not_syn_pair_number_list = [str(x + 1) for x in not_syn_pair_number_list]
    list_rank = list(df['score_for_fgsea'])
    r['source']('PSEA.R')
    function_r = robjects.globalenv['metric_PSEA']
    res = function_r(StrVector(syn_pair_number_list), StrVector(not_syn_pair_number_list),
                     FloatVector(list_rank))
    dict_res = {}
    dict_res['syn_pair'] = {}
    for el in res[1:4]:
        if type(el.split('_')[1]) == str:
            dict_res['syn_pair'][el.split('_')[0]] = el.split('_')[1]
        else:
            dict_res['syn_pair'][el.split('_')[0]] = float(el.split('_')[1])

    dict_res['not_syn_pair'] = {}
    for el in res[5:]:
        if type(el.split('_')[1]) == str:
            dict_res['not_syn_pair'][el.split('_')[0]] = el.split('_')[1]
        else:
            dict_res['not_syn_pair'][el.split('_')[0]] = float(el.split('_')[1])
    if path:
        with open(path, 'w') as file:
            json.dump(dict_res, file)
    return dict_res



def statistic_analys_results(set_one, set_two, name_set_one, name_set_two):
    if len(set_one) > len(set_two) :
        set_big = set_one
        set_small = set_two
    else:
        set_big = set_two
        set_small = set_one
    n = len(set_big) // len(set_small) + 1
    p_value_list = np.empty(n, dtype=float)
    log_p_value_list = np.empty(n, dtype=float)
    log_10_p_value_list = np.empty(n, dtype=float)
    p_list = []
    stat_list = []
    for i in range(n):
        set_big_split = set_big[i * len(set_small): (i + 1) * len(set_small)]
        if len(set_big_split) != len(set_small):
            set_big_split = set_big_split + set_big[0: (len(set_small) - len(set_big_split))]
        (stat, p_value) = ks_2samp(set_big_split, set_small)
        p = np.float64(ks_2samp(set_big_split, set_small)[1])
        #print(ks_2samp(set_big_split, set_small)[1])
        #print(math.log(ks_2samp(set_big_split, set_small)[1]))

        #log_p_value = math.log(ks_2samp(set_big_split, set_small)[1])
        #log_10_p_value = math.log10(ks_2samp(set_big_split, set_small)[1])
        #p_value_list[i] = p_value
        #log_p_value_list[i] = log_p_value
        #log_10_p_value_list[i] = log_10_p_value
        stat_list.append(stat)
        p_list.append(p)
    #print(log_p_value_list)
    #print(log_10_p_value_list)


    average_p_value = mean(p_list)
    average_stat = mean(stat_list)
    dict_statistics= {}
    dict_statistics['average statistic'] = average_stat
    dict_statistics['average pvalue'] = average_p_value
    #dict_statistics['log average pvalue'] = math.log(average_p_value)
    #dict_statistics['log_10 average pvalue'] = math.log10(average_p_value)
    dict_statistics['statistic values'] = stat_list
    dict_statistics['pvalue values'] = p_list
    #dict_statistics['log pvalue values'] = log_p_value_list
    #dict_statistics['log_10 pvalue values'] = log_10_p_value_list
    dict_statistics['mean ' + name_set_one] = mean(set_one)
    dict_statistics['mean ' + name_set_two] = mean(set_two)
    dict_statistics['difference of averagу'] = mean(set_one) - mean(set_two)
    return dict_statistics


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    print(' '.join(namespace.target_type_cell.split('_')))
    data_CD_signature_metadata = pd.read_csv(namespace.path_to_file_with_cd_signature_metadata, index_col=0)
    data_Drugs_metadata = pd.read_csv(namespace.path_to_file_with_drugs_metadata, index_col=0)
    data_intersect_CFM_L1000FWD = pd.read_csv(namespace.path_to_file_with_intersect_cfm_l1000fwd, index_col=0)

    syn, not_syn, all_s = split_signatures(namespace.source_type_cell, ' '.join(namespace.target_type_cell.split('_')), data_intersect_CFM_L1000FWD, data_Drugs_metadata,
                     data_CD_signature_metadata)
    print(len(syn), len(not_syn), len(all_s))
    print(len(set(syn)), len(set(not_syn)), len(set(all_s)))
    with open(namespace.path_to_dir_save_results + '/' + namespace.source_type_cell + '_' + namespace.target_type_cell + '/' + 'list_signatures' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.txt', "w") as file:
        file.write('\n'.join(all_s))

    df_cosine_dist_matrix = pd.read_csv(namespace.path_to_dir_save_results + '/' + namespace.source_type_cell + '_' + namespace.target_type_cell + '/' +
    'cosine_dist_matrix' + '_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.csv', index_col=0)
    syn_split, not_syn_split = split_by_synergy(df_cosine_dist_matrix, syn, not_syn)
    print(len(syn_split), len(not_syn_split))
    start = time.time()
    d = statistic_analys_results(syn_split, not_syn_split, "synergy", 'not synergy')
    print(time.time() - start)
    print(d)
