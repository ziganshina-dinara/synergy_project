import time  # to calculate the time
import argparse  # read arguments from the command line
import sys
import os
import pandas as pd
import numpy as np
import json

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from bayes_opt import UtilityFunction

from search_signatures_by_id import create_list_needed_signatures
from PPI_v1 import calculate_inf_score, func_inf_score_v1, concat_df_log_FC_topo_score_normalize
from CMap_dict import find_similarity
from validation import split_signatures, split_by_synergy, statistic_analys_results, draw, count_synergy_pair_in_top50, count_synergy_pair_in_top_5percent, count_pairs, rank_pair_based_syn_score, calculate_PSEA_metric


def createParser ():
    """
    script parameters parser

    Return
    ------
    instance of the class ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--number_processes', default=10, type=int)
    parser.add_argument('-i', '--number_iteration', type=int)

    parser.add_argument('-source', '--source_type_cell', type=str)
    parser.add_argument('-target', '--target_type_cell', type=str)
    parser.add_argument('-dir_results', '--path_to_dir_save_results', default='DATA', type=str)
    parser.add_argument('-desc', '--description', type=str)
    parser.add_argument('-list_metrics', '--list_metrics_for_pair', default='cosine_dist', type=str)
    parser.add_argument('-file_query_terms', '--presence_file_with_query_terms', default='no', type=str)
    parser.add_argument('-file_terms', '--presence_file_with_terms', default='no', type=str)

    parser.add_argument('-CD_signature_metadata', '--path_to_file_with_CD_signature_metadata',
                        default='DATA/CD_signature_metadata.csv', type=str)
    parser.add_argument('-drugs_metadata', '--path_to_file_with_drugs_metadata',
                        default='DATA/Drugs_metadata.csv', type=str)
    parser.add_argument('-intersect_cfm_l1000fwd', '--path_to_file_with_intersect_cfm_l1000fwd',
                        default='DATA/table_of_cell_conversion_and_chemicals_1.csv', type=str)
    return parser


def synergy(coeff_logFC, coeff_betweenness, coeff_pagerank, coeff_closeness, coeff_katz, coeff_eigenvector, coeff_eigentrust):
    """
    Calculate synergy scores and evaluates the effectiveness of the method.  This function is written for bayes
    optimization

    Parameters
    ---------
    coeff_logFC : float
        coefficient by which the logFC metric is multiplied in the expression inf_score (1 < coeff_logFC < 10)
    coeff_betweenness : float
        coefficient by which the betweenness metric is multiplied in the expression inf_score
        (1 < coeff_betweenness < 10)
    coeff_pagerank : float
        coefficient by which the pagerank metric is multiplied in the expression inf_score (1 < coeff_pagerank < 10)
    coeff_closeness : float
        coefficient by which the closeness metric is multiplied in the expression inf_score (1 < coeff_closeness < 10)
    coeff_katz : float
        coefficient by which the katz metric is multiplied in the expression inf_score (1 < coeff_katz < 10)
    coeff_eigenvector : float
        coefficient by which the eigenvector metric is multiplied in the expression inf_score
        (1 < coeff_eigenvector < 10)
    coeff_eigentrust : float
        coefficient by which the eigentrust metric is multiplied in the expression inf_score (1 < coeff_eigentrust < 10)

    Return
    ------
    DataFrame with signatures ids, whose signatures induced small molecules used for conversion in protocols
    """
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
    global path_to_file_with_query_terms
    global path_to_file_with_terms
    global presence_file_with_query_terms
    global presence_file_with_terms
    global list_metrics_for_pair

    dict_multiplication_factor = {'logFC': coeff_logFC, 'betweenness': coeff_betweenness, 'pagerank': coeff_pagerank,
                                  'closeness': coeff_closeness,
                                  'katz': coeff_katz, 'eigenvector': coeff_eigenvector, 'eigentrust': coeff_eigentrust}
    dict_additive_factor = {'logFC': 1, 'betweenness': 1, 'pagerank': 1, 'closeness': 1, 'katz': 1, 'eigenvector': 1,
                            'eigentrust': 1}

    # combined topological metrics for up  and down  genes, normalized metrics
    df_topo_score = concat_df_log_FC_topo_score_normalize(df_up_topo_score, df_down_topo_score)
    #print(df_topo_score)

    """
    df_topo_score.to_csv(
        path_to_folder_results + '/df_topo_score_' + source_type_cell + '_' + target_type_cell + '.csv',
        columns=df_topo_score.columns, index=True)
    """

    # calculate influence score
    df_inf_score = calculate_inf_score(df_topo_score, func_inf_score_v1, dict_multiplication_factor,
                                         dict_additive_factor)
    print(df_inf_score)
    print('максимум inf_score: {}\n среднее inf_score: {}'.format(df_inf_score['inf_score'].max(), df_inf_score['inf_score'].mean()))
    """
    df_inf_score.to_csv(
        path_to_folder_results_single_parameters + '/df_inf_' + source_type_cell + '_' + target_type_cell + str(i) + '.csv', 
        columns=df_inf_score.columns, index=True)
    print(df_inf_score)
    """

    start_time = time.time()
    # calculate synergy score
    list_metric_name_with_matrix = find_similarity(list_needed_signatures, df_inf_score, namespace.number_processes,
                                                   path_to_file_with_query_terms, path_to_file_with_terms,
                                                   presence_file_with_query_terms,
                                                   presence_file_with_terms, list_metrics_for_pair)

    print('время подсчета synergy_score для всех пар:', '--- %s seconds ---' % (time.time() - start_time))

    # see results
    for (metric_name, matrix) in list_metric_name_with_matrix:
        print('делим посчитанные скоры на 2 выборки: синерг и несинерг')

        syn_split, not_syn_split, df_sign_id_pairs_with_labels_scores = split_by_synergy(df_sign_id_pairs_with_labels,
                                                                                         matrix, metric_name)

        print('считаем статистику')
        d = statistic_analys_results(syn_split, not_syn_split, 'synergy', 'not synergy')
        d['dict_additive_factor'] = dict_additive_factor
        d['dict_multiplication_factor'] = dict_multiplication_factor
        dict_ascending_value = {'cosine_dist': False, 'tanimoto_coeff': False, 'mutual_info_coeff': False,
                                'intersection_terms_of_pair_query': False, 'intersection_terms_of_pair': False}
        print('сортируем датафрейм')
        df_sign_id_pairs_with_labels_scores_sorted = rank_pair_based_syn_score(df_sign_id_pairs_with_labels_scores,
                                                                               metric_name,
                                                                               dict_ascending_value[metric_name])
        print('смотрим на топы')
        d['count_synergy_pair_in_top50'] = count_synergy_pair_in_top50(df_sign_id_pairs_with_labels_scores_sorted)
        d['number_pairs'] = count_pairs(df_sign_id_pairs_with_labels_scores_sorted)
        number_syn_pair_in_top_5percent, len_list_pair_signatures_5_percent, fraction_syn_pairs = count_synergy_pair_in_top_5percent(
            df_sign_id_pairs_with_labels_scores_sorted)
        d['count_synergy_pair_in_top_5_percent'] = number_syn_pair_in_top_5percent
        d['number_pairs_in_top_5_percent'] = len_list_pair_signatures_5_percent
        d['fraction_synergy_pair_in_top_5_percent'] = fraction_syn_pairs
        if metric_name == 'cosine_dist':
            dict_res = calculate_PSEA_metric(df_sign_id_pairs_with_labels_scores_sorted, None)
            d['PSEA'] = np.float(dict_res['syn_pair']['NES'])


        draw(syn_split, not_syn_split,
             path_to_folder_results_single_parameters + '/fig_' + metric_name + '_' + str(
                 i) + '_' + source_type_cell + '_' + \
             target_type_cell + '.png')
        df_search_parameters.loc[i] = [i, dict_additive_factor, dict_multiplication_factor, d['average statistic'],
                                       d['average pvalue'], d['mean synergy'], d['mean not synergy'],
                                       d['mean not synergy'] - d['mean synergy'],
                                       description, d['count_synergy_pair_in_top50'], d['number_pairs'],
                                       d['count_synergy_pair_in_top_5_percent'],
                                       d['number_pairs_in_top_5_percent'], d['fraction_synergy_pair_in_top_5_percent'],
                                       d['PSEA']]
        print(df_search_parameters.loc[i])
        i += 1
    return d['PSEA']
    #return d['mean not synergy'] - d['mean synergy']

    """
    df_cosine_dist_matrix.to_csv(path_to_folder_results_single_parameters + '/df_cosine_dict_matrix_' + source_type_cell + '_' +
                                target_type_cell + str(i) + '.csv', columns=df_cosine_dist_matrix.columns, index=True)
 
    with open(path_to_folder_results_single_parameters + '/dict_results_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '_' + str(i) + '.json',"w") as write_file:
        json.dump(d, write_file)
    """


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

    # find sets of signature id by collecting the signature id corresponding to the small molecules from the protocols
    # for this transition
    with open(
            path_to_folder_results + '/list_signatures_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.txt',
            "r") as file:
        list_needed_signatures = file.read()

    # read type pair
    df_sign_id_pairs_with_labels = pd.read_csv(path_to_folder_results + '/' + 'df_pair_with_class_labels_' + \
                                 source_type_cell + '_' + target_type_cell + '.csv', index_col=0)

    # read topological metrics
    df_up_topo_score = pd.read_csv(
        path_to_folder_results + '/df_topo_up_' + source_type_cell + '_' + target_type_cell + '.csv',
        index_col=0)
    print(df_up_topo_score)
    df_down_topo_score = pd.read_csv(
        path_to_folder_results + '/df_topo_down_' + source_type_cell + '_' + target_type_cell + '.csv',
        index_col=0)
    print(df_down_topo_score)

    # read terms
    presence_file_with_query_terms = namespace.presence_file_with_query_terms
    path_to_file_with_query_terms = path_to_folder_results + '/query_terms_' + namespace.source_type_cell + '_' + \
                                    namespace.target_type_cell + '.txt'

    presence_file_with_terms = namespace.presence_file_with_terms
    path_to_file_with_terms = path_to_folder_results + '/terms_' + namespace.source_type_cell + '_' + \
                              namespace.target_type_cell + '.txt'
    list_metrics_for_pair = namespace.list_metrics_for_pair.split(';')

    """
    df_search_parameters = pd.DataFrame(
        list(zip([0], [0], [0], [0],
                 [0], [0], [0], [0])),
        columns=['id_folder', 'dict_additive_factor', 'dict_multiplication_factor',
                 'average_statistic', 'average_p_value', 'mean_synergy', 'mean_not_synergy', 'dif_mean'])
    """
    # read df with results of validation
    df_search_parameters = pd.read_csv(path_to_folder_results + '/df_search_parameters_' + source_type_cell + '_' + target_type_cell + '.csv', index_col = 0)
    print(df_search_parameters)
    

    i = namespace.number_iteration
    description = namespace.description
    print(description)
    path_to_folder_results_single_parameters = path_to_folder_results + '/Validation_results_' + source_type_cell + '_' + target_type_cell \
                                                   + '/Validation_results_' + source_type_cell + '_' + target_type_cell + '_' + description
    os.mkdir(path_to_folder_results_single_parameters)


    try:
        optimizer = BayesianOptimization(f=synergy,pbounds={'coeff_logFC': (1, 10), 'coeff_betweenness': (1, 10), 'coeff_pagerank': (1, 10),
                    'coeff_closeness': (1, 10), 'coeff_katz': (1, 10),
                     'coeff_eigenvector': (1, 10), 'coeff_eigentrust':(1,10)}, verbose=9, random_state=1)

        optimizer.set_gp_params(alpha=0.001) #0.00001
        optimizer.maximize(init_points=50, n_iter=30) #50,30
        print(optimizer.max)
        with open(path_to_folder_results + '/' + '_max_bayes' + '_' + source_type_cell + '_' + target_type_cell + '_' + description + '.json', 'w') as file:
            json.dump(optimizer.max, file)

        with open(path_to_folder_results + '/' + '_res_bayes' + '_' + source_type_cell + '_' + target_type_cell + '_' + description + '.json', 'w') as file:
            json.dump(optimizer.res, file)

        logger = JSONLogger(path = path_to_folder_results +'/' + "logs.json")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        df_search_parameters.to_csv(path_to_folder_results + '/df_search_parameters_' + source_type_cell + '_' + target_type_cell + '.csv',
            columns=df_search_parameters.columns)

        print('i:', i)


    except OSError:
        df_search_parameters.to_csv(path_to_folder_results + '/df_search_parameters_' + source_type_cell + '_' + target_type_cell + '.csv',
            columns = df_search_parameters.columns)

        with open(path_to_folder_results + '/' +  '_max_bayes' + '_' + source_type_cell + '_' + target_type_cell + '.json', 'w') as file:
            json.dump(optimizer.max, file)

        with open(path_to_folder_results + '/' +  '_res_bayes' + '_' + source_type_cell + '_' + target_type_cell + '.json', 'w') as file:
            json.dump(optimizer.res, file)

        logger = JSONLogger(path = path_to_folder_results +'/' + "logs.json")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)



    '''
    start = time.time()
    print("значение функции:", synergy(1, 1, 1, 1, 1, 1, 1))
    print("время работы одной итерации :", time.time() - start)
    '''




