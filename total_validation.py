import time #to calculate the time
import argparse #read arguments from the command line
import sys
import os
from function_signature_from_DE_v1 import get_signature_for_request_in_STRING, make_signature_from_DE
from PPI_v1 import create_df_gene_logFC_topo_score, calculate_inf_score, func_inf_score_v1
from CMap_dict import cosine_similarity, find_near_signatures
import pandas as pd
from validation import split_signatures, split_by_synergy, statistic_analys_results, draw
from search_signatures_by_id import create_list_needed_signatures
from create_list_dict_parameters import create_list_additive_multiplication_dicts

def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument('-start_step', '--start_step_of_pipeline', type=str)
    parser.add_argument('-way_get_topo_metric', '--way_to_get_df_with_topolog_metrics', type=str)
    parser.add_argument('-availability_file_signatures', '--availability_file_with_required_signatures', type=str)

    parser.add_argument('-DE', '--path_to_file_with_DE', type = str)
    parser.add_argument('-logFC', '--logFC_threshold', default = 1.5, type = float)
    parser.add_argument('-pvalue', '--pvalue_threshold', default = 0.01, type = float)
    parser.add_argument('-sp', '--species', default=9606, type=int)
    parser.add_argument('-exp_thr', '--experimental_score_threshold', default=0.4, type=float)
    parser.add_argument('-signatures', '--path_to_file_with_signatures',
                        default='DATA/CD_signatures_binary_42809.gmt', type=argparse.FileType())
    parser.add_argument('-number_pair', '--number_pair_signatures', default = 50, type=int)
    parser.add_argument('-p', '--number_processes', default =10, type=int)

    parser.add_argument('-source', '--source_type_cell', type=str)
    parser.add_argument('-target', '--target_type_cell', type=str)
    parser.add_argument('-dir_results', '--path_to_dir_save_results', default = 'DATA', type = str)

    parser.add_argument('-CD_signature_metadata', '--path_to_file_with_CD_signature_metadata',
                        default='DATA/CD_signature_metadata.csv', type=str)
    parser.add_argument('-drugs_metadata', '--path_to_file_with_drugs_metadata',
                        default='DATA/Drugs_metadata.csv', type=str)
    parser.add_argument('-intersect_cfm_l1000fwd', '--path_to_file_with_intersect_cfm_l1000fwd',
                        default='DATA/table_of_cell_conversion_and_chemicals_1.csv', type=str)

    parser.add_argument('-lower_additive_factor', '--lower_bound_additive_factor_values', default = 1, type = int)
    parser.add_argument('-upper_additive_factor', '--upper_bound_additive_factor_values', default = 1, type=int)
    parser.add_argument('-lower_multiplication_factor', '--lower_bound_multiplication_factor_values', default = 3, type=int)
    parser.add_argument('-upper_multiplication_factor', '--upper_bound_multiplication_factor_values', default = 3, type=int)

    return parser

if __name__ == '__main__':
    total_start_time = time.time()
    print("начали")
    # read arguments from the command line
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    #create folder for results
    path_to_folder_results = namespace.path_to_dir_save_results + '/' + namespace.source_type_cell + '_' + namespace.target_type_cell

    #read needed files
    data_CD_signature_metadata = pd.read_csv(namespace.path_to_file_with_CD_signature_metadata, index_col=0)
    data_Drugs_metadata = pd.read_csv(namespace.path_to_file_with_drugs_metadata, index_col=0)
    data_intersect_CFM_L1000FWD = pd.read_csv(namespace.path_to_file_with_intersect_cfm_l1000fwd, index_col=0)
    file_with_signatures_42809 = namespace.path_to_file_with_signatures.read()

    # find sets of signature id by collecting the signature id corresponding to the small molecules from the protocols for this transition
    syn_sign_id, not_syn_sign_id, all_sign_id = split_signatures(namespace.source_type_cell,
                                                                 ' '.join(namespace.target_type_cell.split('_')),
                                                                 data_intersect_CFM_L1000FWD, data_Drugs_metadata,
                                                                 data_CD_signature_metadata)

    #creating list of signatures
    if namespace.availability_file_with_required_signatures == 'not exist':
        print('number of synergy signatures :', len(syn_sign_id), 'number of not synergy signatures :', len(not_syn_sign_id), 'number of all signatures :', len(all_sign_id))
        with open(path_to_folder_results + '/list_signature_id_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.txt', "w") as file:
            file.write('\n'.join(all_sign_id))

        list_needed_signatures = create_list_needed_signatures(file_with_signatures_42809, all_sign_id)
        with open(path_to_folder_results + '/list_signatures_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.txt', "w") as file:
            file.write('\n'.join(list_needed_signatures))
    else:
        with open(path_to_folder_results + '/list_signatures_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.txt', "r") as file:
            list_needed_signatures = file.read()

    # creating list of signatures

    #big step of calculating topological scores
    if namespace.way_to_get_df_with_topolog_metrics == 'create topo_df':
        if namespace.start_step_of_pipeline == 'check for genes in the string database':
            start_time = time.time()
            up, down = get_signature_for_request_in_STRING(namespace.path_to_file_with_DE,
                                                                        namespace.logFC_threshold,
                                                                        namespace.pvalue_threshold, number = 2000, species = namespace.species)
            with open(path_to_folder_results + '/list_genes_in_string_up_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.txt',
                    "w") as file:
                file.write('\n'.join(up))
            print('время работы создания сигнатуры')
            print(up)
            with open(path_to_folder_results + '/list_genes_in_string_down_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.txt', "w") \
                    as file:
                file.write('\n'.join(down))
            print('время работы создания сигнатуры запроса с учетом проверки наличия генов в string:', '--- %s seconds ---' % (time.time() - start_time))

        elif namespace.start_step_of_pipeline == 'use ready-made list of genes that are in the string database':

            with open(path_to_folder_results + '/list_genes_in_string_up_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.txt', "r") as file:
                up = file.read().split("\n")
            print(up)
            with open(path_to_folder_results + '/list_genes_in_string_down_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.txt', "r") as file:
                down = file.read().split("\n")
        # make signature
        start_time = time.time()
        series_up_genes, series_down_genes = make_signature_from_DE(namespace.path_to_file_with_DE,
                                                                        namespace.logFC_threshold,
                                                                        namespace.pvalue_threshold)
        print('время работы создания сигнатуры запроса:', '--- %s seconds ---' % (time.time() - start_time))

        # calculate topological metrics and inf_score
        start_time = time.time()
        df_up_topo_score = create_df_gene_logFC_topo_score(up, namespace.species, namespace.pvalue_threshold,
                                                               series_up_genes)
        df_down_topo_score = create_df_gene_logFC_topo_score(down, namespace.species,
                                                                 namespace.pvalue_threshold, series_down_genes)
        df_up_topo_score.to_csv(path_to_folder_results + '/df_topo_up_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.csv',
                                columns = df_up_topo_score.columns, index = True)
        df_down_topo_score.to_csv(path_to_folder_results + '/df_topo_down_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.csv',
                                columns = df_down_topo_score.columns, index = True)
        print('время работы вычисления топологических метрик:', '--- %s seconds ---' % (time.time() - start_time))

    elif namespace.way_to_get_df_with_topolog_metrics == 'use prepared topo_df':
        df_up_topo_score = pd.read_csv(path_to_folder_results + '/df_topo_up_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.csv', index_col=0)
        df_down_topo_score= pd.read_csv(path_to_folder_results + '/df_topo_down_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.csv', index_col=0)

    # create parameter dicts
    list_metric = ['betweenness', 'pagerank', 'closeness', 'katz', 'hits_authority', 'hits_hub', 'eigenvector',
                   'eigentrust']

    (list_dict_additive_factor, list_dict_multiplication_factor) = create_list_additive_multiplication_dicts(
        namespace.lower_bound_additive_factor_values,
        namespace.upper_bound_additive_factor_values, namespace.lower_bound_multiplication_factor_values,
        namespace.upper_bound_multiplication_factor_values,
        list_metric, namespace.source_type_cell, namespace.target_type_cell, path_to_folder_results)
    list_average_statistic = []
    list_average_p_value = []
    list_mean_synergy = []
    list_mean_not_synergy = []
    list_mean_dif = []
    list_id_folder = []

    for (dict_additive_factor, dict_multiplication_factor, i) in zip(list_dict_additive_factor, list_dict_multiplication_factor,
                                                                     range(len(list_dict_additive_factor) * len(list_dict_multiplication_factor))):
        path_to_folder_results_single_parameters = path_to_folder_results + '/' + namespace.source_type_cell + '_' + namespace.target_type_cell + '_' \
                                                   + str(i)
        os.mkdir(path_to_folder_results_single_parameters)

        list_id_folder.append(i)
        # calculate influence score
        df_up_inf_score = calculate_inf_score(df_up_topo_score , func_inf_score_v1, dict_multiplication_factor, dict_additive_factor)
        df_down_inf_score = calculate_inf_score( df_down_topo_score, func_inf_score_v1, dict_multiplication_factor, dict_additive_factor)



        start_time = time.time()
        df_cosine_dist_matrix = cosine_similarity(list_needed_signatures, df_up_inf_score, df_down_inf_score, namespace.number_processes)
        df_cosine_dist_matrix.to_csv(path_to_folder_results_single_parameters + '/df_cosine_dict_matrix_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.csv',
            columns=df_cosine_dist_matrix.columns, index = True)
        print('время подсчета synergy_score для всех пар:', '--- %s seconds ---' % (time.time() - start_time))

        start_time = time.time()
        df_with_signatures_pert_id = find_near_signatures( list_needed_signatures, df_cosine_dist_matrix,
                                        namespace.number_pair_signatures, data_CD_signature_metadata)
        print('время отбора пар сигнатур:', '--- %s seconds ---' % (time.time() - start_time))
        df_with_signatures_pert_id.to_csv( path_to_folder_results_single_parameters + '/closest_pair_sign_id_pert_id_pert_name_score_' + namespace.source_type_cell + '_'
                                           + namespace.target_type_cell  + '.csv', columns=df_with_signatures_pert_id.columns)
        print('полное время работы:', '--- %s seconds ---' % (time.time() - total_start_time))

        #see results
        syn_split, not_syn_split = split_by_synergy(df_cosine_dist_matrix, syn_sign_id, not_syn_sign_id)
        print(len(syn_split), len(not_syn_split))
        d = statistic_analys_results(syn_split, not_syn_split, 'synergy', 'not synergy')
        draw(syn_split, not_syn_split, path_to_folder_results_single_parameters + '/fig_' + namespace.source_type_cell + '_' + namespace.target_type_cell +'.png' )
        list_average_statistic.append(d['average statistic'])
        list_average_p_value.append(d['average pvalue'])
        list_mean_synergy.append(d['mean synergy'])
        list_mean_not_synergy.append(d['mean not synergy'])
        list_mean_dif.append(d['mean not synergy'] - d['mean synergy'])
    df_search_parameters = pd.DataFrame(list(zip(list_id_folder, list_dict_additive_factor, list_dict_multiplication_factor, list_average_statistic, list_average_p_value,
                                   list_mean_synergy, list_mean_not_synergy, list_mean_dif)), columns=['id_folder', 'dict_additive_factor',  'dict_multiplication_factor',
                             'average_statistic', 'average_p_value','mean_synergy', 'mean_not_synergy', 'dif_mean'])
    df_search_parameters.to_csv(path_to_folder_results + '/df_search_parameters_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.csv',
                                columns = df_search_parameters.columns)

