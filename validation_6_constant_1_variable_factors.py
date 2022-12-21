import time  # to calculate the time
import argparse  # read arguments from the command line
import sys
import os
from search_signatures_by_id import create_list_needed_signatures
from PPI_v1 import calculate_inf_score, func_inf_score_v1, \
    concat_df_log_FC_topo_score_normalize
from CMap_dict import cosine_similarity
import pandas as pd
import json
from validation import split_signatures, split_by_synergy, statistic_analys_results, draw


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-number_pair', '--number_pair_signatures', default=50, type=int)
    parser.add_argument('-p', '--number_processes', default=10, type=int)
    parser.add_argument('-i', '--number_iteration', type=int)

    parser.add_argument('-source', '--source_type_cell', type=str)
    parser.add_argument('-target', '--target_type_cell', type=str)
    parser.add_argument('-desc', '--description', type=str)
    parser.add_argument('-dir_results', '--path_to_dir_save_results', default='DATA', type=str)
    #parser.add_argument('-description', '--description_validation', type=str)

    parser.add_argument('-CD_signature_metadata', '--path_to_file_with_CD_signature_metadata',
                        default='DATA/CD_signature_metadata.csv', type=str)
    parser.add_argument('-drugs_metadata', '--path_to_file_with_drugs_metadata',
                        default='DATA/Drugs_metadata.csv', type=str)
    parser.add_argument('-intersect_cfm_l1000fwd', '--path_to_file_with_intersect_cfm_l1000fwd',
                        default='DATA/table_of_cell_conversion_and_chemicals_1.csv', type=str)

    return parser


def synergy(coeff_logFC, coeff_betweenness, coeff_pagerank, coeff_closeness, coeff_katz, coeff_eigenvector,
            coeff_eigentrust):

    dict_multiplication_factor = {'logFC': coeff_logFC, 'betweenness': coeff_betweenness, 'pagerank': coeff_pagerank,
                                  'closeness': coeff_closeness,
                                  'katz': coeff_katz, 'eigenvector': coeff_eigenvector, 'eigentrust': coeff_eigentrust}
    dict_additive_factor = {'logFC': 1, 'betweenness': 1, 'pagerank': 1, 'closeness': 1, 'katz': 1, 'eigenvector': 1,
                            'eigentrust': 1}

    global i
    global source_type_cell
    global target_type_cell
    global path_to_folder_results
    global data_CD_signature_metadata
    global data_Drugs_metadata
    global data_intersect_CFM_L1000FWD
    global df_search_parameters
    global df_up_topo_score
    global df_down_topo_score
    global list_needed_signatures
    global description
    global path_to_folder_results_single_parameters

    # combined up and down and normalized metrics
    df_topo_score = concat_df_log_FC_topo_score_normalize(df_up_topo_score, df_down_topo_score)
    print(df_topo_score)

    """
    df_topo_score.to_csv(
        path_to_folder_results + '/df_topo_score_' + source_type_cell + '_' + target_type_cell + '.csv',
        columns=df_topo_score.columns, index=True)
    """

    # calculate influence score
    df_inf_score = calculate_inf_score(df_topo_score, func_inf_score_v1, dict_multiplication_factor,
                                       dict_additive_factor)
    print(df_inf_score)
    print('максимум inf_score: {}\n среднее inf_score: {}'.format(df_inf_score['inf_score'].max(),
                                                                  df_inf_score['inf_score'].mean()))
    """
    df_inf_score.to_csv(
        path_to_folder_results_single_parameters + '/df_inf_' + source_type_cell + '_' + target_type_cell + str(i) + '.csv', 
        columns=df_inf_score.columns, index=True)
    print(df_inf_score)
    """

    start_time = time.time()
    df_cosine_dist_matrix = cosine_similarity(list_needed_signatures, df_inf_score, namespace.number_processes)
    print('время подсчета synergy_score для всех пар:', '--- %s seconds ---' % (time.time() - start_time))
    """
    df_cosine_dist_matrix.to_csv(path_to_folder_results_single_parameters + '/df_cosine_dict_matrix_' + source_type_cell + '_' +
                                target_type_cell + str(i) + '.csv', columns=df_cosine_dist_matrix.columns, index=True)
    """

    # see results
    global syn_sign_id
    global not_syn_sign_id

    syn_split, not_syn_split = split_by_synergy(df_cosine_dist_matrix, syn_sign_id, not_syn_sign_id)
    print(len(syn_split), len(not_syn_split))
    d = statistic_analys_results(syn_split, not_syn_split, 'synergy', 'not synergy')
    d['dict_additive_factor'] = dict_additive_factor
    d['dict_multiplication_factor'] = dict_multiplication_factor

    """
    with open(path_to_folder_results_single_parameters + '/dict_results_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '_' + str(i) + '.json',"w") as write_file:
        json.dump(d, write_file)
    """
    draw(syn_split, not_syn_split,
         path_to_folder_results_single_parameters + '/fig_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '_'
         + str(i) + '.png')

    df_search_parameters.loc[i] = [i, i, dict_additive_factor, dict_multiplication_factor, d['average statistic'],
                                   d['average pvalue'],
                                   d['mean synergy'], d['mean not synergy'], d['mean not synergy'] - d['mean synergy'],
                                   description]
    i += 1
    return d['mean not synergy'] - d['mean synergy']


if __name__ == '__main__':
    total_start_time = time.time()
    print("начали")
    # read arguments from the command line
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    # create folder for results
    source_type_cell = namespace.source_type_cell
    target_type_cell = namespace.target_type_cell
    path_to_folder_results = namespace.path_to_dir_save_results + '/' + source_type_cell + '_' + target_type_cell

    # read needed files
    data_CD_signature_metadata = pd.read_csv(namespace.path_to_file_with_CD_signature_metadata, index_col=0)
    data_Drugs_metadata = pd.read_csv(namespace.path_to_file_with_drugs_metadata, index_col=0)
    data_intersect_CFM_L1000FWD = pd.read_csv(namespace.path_to_file_with_intersect_cfm_l1000fwd, index_col=0)

    # find sets of signature id by collecting the signature id corresponding to the small molecules from the protocols for this transition
    # (syn_sign_id, not_syn_sign_id, all_sign_id) = split_signatures(namespace.source_type_cell, ' '.join(namespace.target_type_cell.split('_')),
    #         data_intersect_CFM_L1000FWD, data_Drugs_metadata, data_CD_signature_metadata)
    """
    with open(path_to_folder_results + '/' + 'list_signature_id_syn_' + \
                         '_' + source_type_cell + '_' + target_type_cell + '.json', 'w') as file:
        json.dump(syn_sign_id, file)
    with open(path_to_folder_results + '/' + 'list_signature_id_not_syn_' + \
              '_' + source_type_cell + '_' + target_type_cell + '.json', 'r') as file:
        json.dump(not_syn_sign_id, file)
    """
    with open(
            path_to_folder_results + '/list_signature_id_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.txt',
            "r") as file:
        all_sign_id = file.read().split('\n')
        print(all_sign_id[:10])
    with open('DATA/CD_signatures_binary_42809.gmt', "r") as file:
        file_with_signatures_42809 = file.read()

    # list_needed_signatures = create_list_needed_signatures(file_with_signatures_42809, all_sign_id)
    # with open( path_to_folder_results + '/list_signatures_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.txt',
    #       "w") as file:
    #  file.write('\n'.join(list_needed_signatures))
    with open(
            path_to_folder_results + '/list_signatures_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.txt',
            "r") as file:
        list_needed_signatures = file.read()
    print(list_needed_signatures[:3])

    with open(
            path_to_folder_results + '/' + 'list_signature_id_syn_' + source_type_cell + "_" + target_type_cell + '.json',
            'r') as file:
        syn_sign_id = json.load(file)
    print('число синергетических пар:', len(syn_sign_id))
    with open(
            path_to_folder_results + '/' + 'list_signature_id_not_syn_' + source_type_cell + '_' + target_type_cell + '.json',
            'r') as file:
        not_syn_sign_id = json.load(file)
    print('число несинергетических пар:', len(not_syn_sign_id))
    df_up_topo_score = pd.read_csv(
        path_to_folder_results + '/df_topo_up_' + source_type_cell + '_' + target_type_cell + '.csv',
        index_col=0)
    #print(df_up_topo_score)
    df_down_topo_score = pd.read_csv(
        path_to_folder_results + '/df_topo_down_' + source_type_cell + '_' + target_type_cell + '.csv',
        index_col=0)
    #print(df_down_topo_score)
    """
    df_search_parameters = pd.DataFrame(
        list(zip([0], [0], [0], [0],
                 [0], [0], [0], [0])),
        columns=['id_folder', 'dict_additive_factor', 'dict_multiplication_factor',
                 'average_statistic', 'average_p_value', 'mean_synergy', 'mean_not_synergy', 'dif_mean'])
    """

    df_search_parameters = pd.read_csv(
        path_to_folder_results + '/df_search_parameters_' + source_type_cell + '_' + target_type_cell + '.csv',
        index_col=0)
    #print(df_search_parameters)


    i = namespace.number_iteration



    list_metric = ['logFC', 'betweenness', 'pagerank', 'closeness', 'katz', 'eigenvector','eigentrust']
    dict_factor = {'logFC': 5, 'betweenness': 5, 'pagerank': 5, 'closeness': 5,'katz': 5, 'eigenvector': 5, 'eigentrust': 5}
    for metric in list_metric:
        description = '6_constant_factors_1_variable_' + metric + namespace.description
        path_to_folder_results_single_parameters = path_to_folder_results + '/Validation_results_' + source_type_cell + '_' + target_type_cell \
                                                   + '/Validation_results_' + source_type_cell + '_' + target_type_cell + '_' + description
        os.mkdir(path_to_folder_results_single_parameters)
        for coeff in range(10, 110, 2):
            print(coeff/10)
            dict_factor[metric] = coeff/10
            print(dict_factor)
            synergy(dict_factor['logFC'], dict_factor['betweenness'], dict_factor['pagerank'], dict_factor['closeness'], dict_factor['katz'],
                    dict_factor['eigenvector'], dict_factor['eigentrust'])
        df_search_parameters.to_csv(
            path_to_folder_results + '/df_search_parameters_' + source_type_cell + '_' + target_type_cell + '.csv',
            columns=df_search_parameters.columns)
        dict_factor = {'logFC': 5, 'betweenness': 5, 'pagerank': 5, 'closeness': 5, 'katz': 5, 'eigenvector': 5,'eigentrust': 5}

    df_search_parameters.to_csv(
        path_to_folder_results + '/df_search_parameters_' + source_type_cell + '_' + target_type_cell + '.csv',
        columns=df_search_parameters.columns)












