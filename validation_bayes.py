import time #to calculate the time
import argparse #read arguments from the command line
import sys
import os
from PPI_v1 import create_df_gene_logFC_topo_score, calculate_inf_score, func_inf_score_v1, concat_df_log_FC_topo_score_normalize
from CMap_dict import cosine_similarity, find_near_signatures
import pandas as pd
import json
from validation import split_signatures, split_by_synergy, statistic_analys_results, draw
from create_list_dict_parameters import create_list_additive_multiplication_dicts
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from bayes_opt import UtilityFunction
def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument('-number_pair', '--number_pair_signatures', default = 50, type=int)
    parser.add_argument('-p', '--number_processes', default =10, type=int)
    parser.add_argument('-i', '--number_iteration', type=int)

    parser.add_argument('-source', '--source_type_cell', type=str)
    parser.add_argument('-target', '--target_type_cell', type=str)
    parser.add_argument('-dir_results', '--path_to_dir_save_results', default = 'DATA', type = str)

    parser.add_argument('-CD_signature_metadata', '--path_to_file_with_CD_signature_metadata',
                        default='DATA/CD_signature_metadata.csv', type=str)
    parser.add_argument('-drugs_metadata', '--path_to_file_with_drugs_metadata',
                        default='DATA/Drugs_metadata.csv', type=str)
    parser.add_argument('-intersect_cfm_l1000fwd', '--path_to_file_with_intersect_cfm_l1000fwd',
                        default='DATA/table_of_cell_conversion_and_chemicals_1.csv', type=str)


    return parser



def synergy(coeff_logFC, coeff_betweenness, coeff_pagerank, coeff_closeness, coeff_katz, coeff_hits_authority, coeff_hits_hub, coeff_eigenvector, coeff_eigentrust):

    list_metric = ['logFC', 'betweenness', 'pagerank', 'closeness', 'katz', 'hits_authority', 'hits_hub', 'eigenvector',
                   'eigentrust']
    dict_multiplication_factor = {'logFC': coeff_logFC, 'betweenness':coeff_betweenness, 'pagerank': coeff_pagerank, 'closeness': coeff_closeness, 'katz': coeff_katz, 'hits_authority':
        coeff_hits_authority, 'hits_hub': coeff_hits_hub, 'eigenvector': coeff_eigenvector, 'eigentrust': coeff_eigentrust}
    dict_additive_factor = {'logFC': 1, 'betweenness': 1, 'pagerank': 1, 'closeness': 1, 'katz': 1, 'hits_authority':
        1, 'hits_hub': 1, 'eigenvector': 1, 'eigentrust': 1}

    global list_average_statistic
    global list_average_p_value
    global list_mean_synergy
    global list_mean_not_synergy
    global list_mean_dif
    global list_id_folder
    global i
    global source_type_cell
    global target_type_cell
    global path_to_folder_results
    global data_CD_signature_metadata
    global data_Drugs_metadata
    global data_intersect_CFM_L1000FWD
    global df_search_parameters
    df_up_topo_score = pd.read_csv(
        path_to_folder_results + '/df_topo_up_' + source_type_cell + '_' + target_type_cell + '.csv',
        index_col=0)
    df_down_topo_score = pd.read_csv(
        path_to_folder_results + '/df_topo_down_' + source_type_cell + '_' + target_type_cell + '.csv',
        index_col=0)
    df_topo_score = concat_df_log_FC_topo_score_normalize(df_up_topo_score, df_down_topo_score)

    df_topo_score.to_csv(
        path_to_folder_results + '/df_topo_score_' + source_type_cell + '_' + target_type_cell + '.csv',
        columns=df_topo_score.columns, index=True)
    path_to_folder_results_single_parameters = path_to_folder_results + '/Validation_results_' + source_type_cell + '_' + target_type_cell + '/' + source_type_cell\
                                               + '_' + target_type_cell + '_' + str(i)
    os.mkdir(path_to_folder_results_single_parameters)


    # calculate influence score
    df_inf_score = calculate_inf_score(df_topo_score, func_inf_score_v1, dict_multiplication_factor,
                                              dict_additive_factor)
    df_inf_score.to_csv(
        path_to_folder_results_single_parameters + '/df_inf_' + source_type_cell + '_' + target_type_cell + '.csv',
        columns=df_inf_score.columns, index=True)
    print(df_inf_score)


    start_time = time.time()
    global list_needed_signatures
    df_cosine_dist_matrix = cosine_similarity(list_needed_signatures, df_inf_score,
                                                  namespace.number_processes)
    df_cosine_dist_matrix.to_csv(path_to_folder_results_single_parameters + '/df_cosine_dict_matrix_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.csv',
            columns=df_cosine_dist_matrix.columns, index=True)
    print('время подсчета synergy_score для всех пар:', '--- %s seconds ---' % (time.time() - start_time))

    start_time = time.time()
    df_with_signatures_pert_id = find_near_signatures(list_needed_signatures, df_cosine_dist_matrix,
                                                          namespace.number_pair_signatures, data_CD_signature_metadata)
    print('время отбора пар сигнатур:', '--- %s seconds ---' % (time.time() - start_time))
    df_with_signatures_pert_id.to_csv(
            path_to_folder_results_single_parameters + '/closest_pair_sign_id_pert_id_pert_name_score_' + namespace.source_type_cell + '_'
            + namespace.target_type_cell + '.csv', columns=df_with_signatures_pert_id.columns)
    print('полное время работы:', '--- %s seconds ---' % (time.time() - total_start_time))

    # see results
    global syn_sign_id
    global not_syn_sign_id

    syn_split, not_syn_split = split_by_synergy(df_cosine_dist_matrix, syn_sign_id, not_syn_sign_id)
    print(len(syn_split), len(not_syn_split))
    d = statistic_analys_results(syn_split, not_syn_split, 'synergy', 'not synergy', path_to_folder_results_single_parameters +  '/' +
            'statistics' + '_' + source_type_cell + '_' + target_type_cell + '.json')
    draw(syn_split, not_syn_split,
        path_to_folder_results_single_parameters + '/fig_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.png')

    df_search_parameters.loc[i] = [i, dict_additive_factor, dict_multiplication_factor, d['average statistic'], d['average pvalue'], d['mean synergy'], d['mean not synergy'],d['mean not synergy'] - d['mean synergy'] ]
    i += 1
    return 1/d['average pvalue']



if __name__ == '__main__':
    total_start_time = time.time()
    print("начали")
    # read arguments from the command line
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    #create folder for results
    source_type_cell = namespace.source_type_cell
    target_type_cell = namespace.target_type_cell
    path_to_folder_results = namespace.path_to_dir_save_results + '/' + source_type_cell + '_' + target_type_cell

    #read needed files
    data_CD_signature_metadata = pd.read_csv(namespace.path_to_file_with_CD_signature_metadata, index_col=0)
    data_Drugs_metadata = pd.read_csv(namespace.path_to_file_with_drugs_metadata, index_col=0)
    data_intersect_CFM_L1000FWD = pd.read_csv(namespace.path_to_file_with_intersect_cfm_l1000fwd, index_col=0)


    # find sets of signature id by collecting the signature id corresponding to the small molecules from the protocols for this transition
    #(syn_sign_id, not_syn_sign_id, all_sign_id) = split_signatures(namespace.source_type_cell, ' '.join(namespace.target_type_cell.split('_')), data_intersect_CFM_L1000FWD, data_Drugs_metadata, data_CD_signature_metadata)


    with open(path_to_folder_results + '/' + 'list_signature_id_syn_' + \
                         '_' + source_type_cell + '_' + target_type_cell + '.json', 'r') as file:
        syn_sign_id = json.load(file)
    print('число синергетических пар:', len(syn_sign_id))
    with open(path_to_folder_results + '/' + 'list_signature_id_not_syn_' + \
                         '_' + source_type_cell + '_' + target_type_cell + '.json', 'r') as file:
        not_syn_sign_id = json.load(file)

    df_search_parameters = pd.read_csv(path_to_folder_results + '/df_search_parameters_' + source_type_cell + '_' + target_type_cell + '.csv', index_col = 0)

    print('число несинергетических пар:', len(not_syn_sign_id))



    with open(
            path_to_folder_results + '/list_signatures_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.txt',
            "r") as file:
        list_needed_signatures = file.read()
    i = namespace.number_iteration
    '''

    optimizer = BayesianOptimization(f=synergy,pbounds={'coeff_logFC': (0, 1), 'coeff_betweenness': (0, 1), 'coeff_pagerank': (0, 1), 'coeff_closeness': (0, 1),
                 'coeff_katz': (0, 1), 'coeff_hits_authority': (0, 1), 'coeff_hits_hub': (0, 1),
                 'coeff_eigenvector': (0, 1), 'coeff_eigentrust':(0,1)}, verbose = 9,random_state = 1)

    optimizer.set_gp_params(alpha=0.01)
    optimizer.maximize(
        init_points=2,
        n_iter=3,
    )
    print(optimizer.max)
    with open(path_to_folder_results + '/' +  '_max_bayes' + \
                         '_' + source_type_cell + '_' + target_type_cell + '.json', 'w') as file:
        json.dump(optimizer.max, file)

    with open(path_to_folder_results + '/' +  '_res_bayes' + \
                         '_' + source_type_cell + '_' + target_type_cell + '.json', 'w') as file:
        json.dump(optimizer.res, file)


    


    logger = JSONLogger(path = path_to_folder_results +'/' + "bayes_logs_" + namespace.source_type_cell + '_' + namespace.target_type_cell  + ".json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    '''


    print("значение функции:", synergy(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5))

    """
    df_search_parameters = pd.DataFrame(
        list(zip(list_id_folder, list_dict_additive_factor, list_dict_multiplication_factor, list_average_statistic,
                 list_average_p_value, list_mean_synergy, list_mean_not_synergy, list_mean_dif)),
        columns=['id_folder', 'dict_additive_factor', 'dict_multiplication_factor',
                 'average_statistic', 'average_p_value', 'mean_synergy', 'mean_not_synergy', 'dif_mean'])
    
    """
    """
    optimizer = BayesianOptimization(f=synergy, pbounds={'coeff_logFC': (0, 5), 'coeff_betweenness': (0, 5), 'coeff_pagerank': (0, 5), 'coeff_closeness': (0, 5),
                 'coeff_katz': (0, 5), 'coeff_hits_authority': (0, 5), 'coeff_hits_hub': (0, 5),
                 'coeff_eigenvector': (0, 5), 'coeff_eigentrust':(0,5)},
                                     verbose=9, random_state=1)

    #load_logs(optimizer, logs=[path_to_folder_results +'/' + "bayes_logs_" + namespace.source_type_cell + '_' + namespace.target_type_cell  + ".json"]);
    #optimizer.set_bounds(new_bounds={'coeff_logFC': (0, 5), 'coeff_betweenness': (0, 5), 'coeff_pagerank': (0, 5), 'coeff_closeness': (0, 5),
                 #'coeff_katz': (0, 5), 'coeff_hits_authority': (0, 5), 'coeff_hits_hub': (0, 5),
                 #'coeff_eigenvector': (0, 5), 'coeff_eigentrust':(0,5)})
    optimizer.set_gp_params(alpha=0.0001)
    optimizer.maximize(
        init_points=2,
        n_iter=1,
    )
    print(optimizer.max)
    with open(path_to_folder_results + '/' + '_max_bayes_0_5' + \
              '_' + source_type_cell + '_' + target_type_cell + '.json', 'w') as file:
        json.dump(optimizer.max, file)

    with open(path_to_folder_results + '/' + '_res_bayes_0_5' + \
              '_' + source_type_cell + '_' + target_type_cell + '.json', 'w') as file:
        json.dump(optimizer.res, file)

    logger = JSONLogger(
        path=path_to_folder_results + '/' + "bayes_0_5_" + namespace.source_type_cell + '_' + namespace.target_type_cell + "logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    """
    df_search_parameters.to_csv(
        path_to_folder_results + '/df_search_parameters_' + source_type_cell + '_' + target_type_cell + '.csv',
        columns=df_search_parameters.columns, )
    print('i:', i)