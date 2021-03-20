import time #to calculate the time
import argparse #read arguments from the command line
import sys
from function_signature_from_DE_v1 import get_signature_for_request_in_STRING, make_signature_from_DE
from PPI_v1 import create_df_gene_logFC_topol_inf_score
from CMap_dict import cosine_similarity
import pandas as pd

def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument('-DE', '--path_to_file_with_DE', type = str)
    parser.add_argument('-logFC', '--logFC_threshold', default = 1.5, type = float)
    parser.add_argument('-pvalue', '--pvalue_threshold', default = 0.01, type = float)
    parser.add_argument('-signatures', '--path_to_file_with_signatures',
                        default='DATA/CD_signatures_binary_42809.gmt', type=argparse.FileType())
    parser.add_argument('-dir_results', '--path_to_dir_save_results', default = 'DATA/', type = str)
    parser.add_argument('-sp', '--species', default = 9606, type = int)
    parser.add_argument('-exp_thr', '--experimental_score_threshold', default = 0.4, type = float)
    parser.add_argument('-conv', '--conversion', type=str)
    parser.add_argument('-CD_signature_metadata', '--path_to_file_with_CD_signature_metadata',
                        default='DATA/CD_signature_metadata.csv', type=str)
    parser.add_argument('-number_pair', '--number_pair_signatures', type=int)
    return parser

if __name__ == '__main__':
    total_start_time = time.time()
    print("начали")
    # read arguments from the command line
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    # read the genes with logFC selected by logFC, pvalue
    start_time = time.time()
    up, down = get_signature_for_request_in_STRING(namespace.path_to_file_with_DE,
                                                                namespace.logFC_threshold,
                                                                namespace.pvalue_threshold, number = 2000, species = namespace.species)
    print('время работы создания сигнатуры запроса с учетом проверки наличия генов в string:', '--- %s seconds ---' % (time.time() - start_time))

    '''
    with open("DATA/protein_network/list_proteins_up_in_STRING_cheart_fibroblast.txt", "r") as file:
        up = file.read().split("\n")
    print(up)
    with open("DATA/protein_network/list_proteins_down_in_STRING_cheart_fibroblast.txt", "r") as file:
        down = file.read().split("\n")
    '''
    # make signature
    start_time = time.time()
    series_up_genes, series_down_genes = make_signature_from_DE(namespace.path_to_file_with_DE, namespace.logFC_threshold, namespace.pvalue_threshold)
    print('время работы создания сигнатуры запроса:', '--- %s seconds ---' % (time.time() - start_time))

    # calculate topological metrics and inf_score
    start_time = time.time()
    df_up_inf_score = create_df_gene_logFC_topol_inf_score(up, namespace.species, namespace.pvalue_threshold, series_up_genes)
    df_down_inf_score = create_df_gene_logFC_topol_inf_score(down, namespace.species,
                                                                   namespace.pvalue_threshold, series_down_genes)

    print('время работы вычисления топологических метрик:', '--- %s seconds ---' % (time.time() - start_time))



    out_of_file_with_signatures = namespace.path_to_file_with_signatures.read()
    df_CD_signature_metadata = pd.read_csv(namespace.path_to_file_with_CD_signature_metadata, index_col=0)

    start_time = time.time()
    df_cosine_dist_matrix = cosine_similarity(out_of_file_with_signatures, df_up_inf_score, df_down_inf_score)
    df_cosine_dist_matrix.to_csv(namespace.path_to_dir_save_results + '/cosine_dist_matrix_' + namespace.conversion + '.csv',
        columns=df_cosine_dist_matrix.columns, index=True)
    print('время подсчета synergy_score для всех пар:', '--- %s seconds ---' % (time.time() - start_time))

    start_time = time.time()
    df_with_signatures_pert_id = find_near_signatures(out_of_file_with_signatures, df_cosine_dist_matrix,
                                                      namespace.number_pair_signatures,
                                                      df_CD_signature_metadata)
    print('время отбора пар сигнатур:', '--- %s seconds ---' % (time.time() - start_time))
    df_with_signatures_pert_id.to_csv(
        namespace.path_to_dir_save_results + '/closest_pair_sign_id_pert_id_pert_name_score_' + namespace.conversion + '.csv',
        columns=df_with_signatures_pert_id.columns)

    print('полное время работы:', '--- %s seconds ---' % (time.time() - total_start_time))