import time #to calculate the time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import rankdata
from sklearn.metrics import mutual_info_score
import argparse #read arguments from the command line
import sys
from joblib import Parallel, delayed
from PPI_v1 import concat_df_log_FC_topo_score_normalize, func_inf_score_v1, calculate_inf_score
from enrichment_analysis import Enrich
from multiprocessing import Pool


#setting the expected parameters
def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument('-inf_score_up', '--path_to_file_with_inf_score_up', type = str)
    parser.add_argument('-inf_score_down', '--path_to_file_with_inf_score_down', type = str)
    parser.add_argument('-signatures', '--path_to_file_with_signatures',
                        default = 'DATA/CD_signatures_binary_42809.gmt', type = argparse.FileType())
    parser.add_argument('-dir_results', '--path_to_dir_save_results', default = 'DATA', type = str)
    parser.add_argument('-CD_signature_metadata', '--path_to_file_with_CD_signature_metadata',
                        default = 'DATA/CD_signature_metadata.csv', type = str)
    parser.add_argument('-number_pair', '--number_pair_signatures', default = 50, type = int)
    parser.add_argument('-p', '--number_processes', default = 10, type=int)
    parser.add_argument('-conv', '--conversion', type = str)


    return parser


class Signature:
    """
    class Signature is used to create a signature based on a list of genes with increased and decreased expression

    Attributes
    ----------
    id : str
        the ID of the signature
    up_genes : list
        list of genes with increased expression
    down_genes: list
        list of genes with decreased expression
    number_in_signature_list: int
        number of signature in list with signatures from database ([0, length of the list with signatures))

    """
    def __init__(self, id, up_genes, down_genes, up_terms=None, down_terms=None):
        self.id = id
        self.up_genes = up_genes
        self.down_genes = down_genes
        self.up_terms = up_terms
        self.down_terms = down_terms

    def set_up_terms(self):
        enrich_up = Enrich(self.up_genes, 0.05)
        self.up_terms = enrich_up.get_enrichment_results()

    def set_down_terms(self):
        enrich_down = Enrich(self.down_genes, 0.05)
        self.down_terms = enrich_down.get_enrichment_results()


class Signature_pair:
    """
    Signature_pair class is intended for combining lists of genes with increased and decreased expression
    of a pair of signatures.

    Attributes
    ----------
    signature_1 : instance of the class Signature
        first signature of pair
    signature_2 : instance of the class Signature
        second signature of pair

    Methods
    -------
    get_id_signatures()
        return a tuple of IDs of signature of pair
    get_number_signatures()
        return a tuple of number of signature of pair in list with signatures from database
    get_up_down()
        combines lists of genes with increased expression of a pair of signatures into one list and combines lists
        of genes with reduced expression of a pair of signatures into one list.
        This method returns a tuple of these 2 lists.
    get_up()
        returns lists of genes with increased expression resulting from combining lists of genes with
         increased expression of a pair of signatures
    get_down()
        returns lists of genes with decreased expression resulting from combining lists of genes with
         decreased expression of a pair of signatures
    """
    def __init__(self, signature_1, signature_2):
        self.signature_1 = signature_1
        self.signature_2 = signature_2

    def get_id_signatures(self):
        return (self.signature_1.id, self.signature_2.id)

    @staticmethod
    def combine_sets(list_up_1, list_up_2, list_down_1, list_down_2):
        set_up_1 = set(list_up_1)
        set_up_2 = set(list_up_2)
        set_down_1 = set(list_down_1)
        set_down_2 = set(list_down_2)
        up = set_up_1 | set_up_2
        down = set_down_1 | set_down_2
        if (up & down):
            for term in (up & down):
                up.discard(term)
                down.discard(term)
        return (list(up), list(down))

    def get_up_down_genes(self):
        return self.combine_sets(self.signature_1.up_genes, self.signature_2.up_genes, self.signature_1.down_genes, self.signature_2.down_genes)

    def get_up_down_terms(self):
        return self.combine_sets(self.signature_1.up_terms, self.signature_2.up_terms, self.signature_1.down_terms, self.signature_2.down_terms)




def create_signature_list(out_from_file_with_signatures, path_to_file_with_terms, presence_file_with_terms):
    """
    creates a list of instances of the class Signature by signatures from the contents of the file with signatures
    of L1000FWD database

    Parametrs
    ---------
    out_from_file_with_signatures:  str
        contents of the file with signatures of L1000FWD database
    Return
    ------
    list of instances of the class Signature
    """
    signature_list = []
    if presence_file_with_terms == 'no':
        with open(path_to_file_with_terms, "w") as file:
            for i in range(0, len(out_from_file_with_signatures.split('\n'))-1, 2):
                start = time.time()
                signature_up_list = out_from_file_with_signatures.split('\n')[i].split('\t')
                signature_down_list = out_from_file_with_signatures.split('\n')[i+1].split('\t')
                signature = Signature(signature_up_list[0], signature_up_list[2:], signature_down_list[2:])
                signature.set_up_terms()
                signature.set_down_terms()
                signature_list.append(signature)


                file.write(signature_up_list[0] + '\t' + 'up_terms\t' + '\t'.join(signature.up_terms) + '\n')
                file.write(signature_down_list[0] + '\t' + 'down_terms\t' + '\t'.join(signature.down_terms) + '\n')
    else:
        with open(path_to_file_with_terms, "r") as file:
            out_from_file_with_terms = file.read()
            for (i_in_genes, i_in_terms) in zip(range(0, len(out_from_file_with_signatures.split('\n')) - 1, 2),
                                                range(0, len(out_from_file_with_terms.split('\n')) - 1, 2)):
                signature_up_list = out_from_file_with_signatures.split('\n')[i_in_genes].split('\t')
                signature_down_list = out_from_file_with_signatures.split('\n')[i_in_genes + 1].split('\t')
                terms_up_list = out_from_file_with_terms.split('\n')[i_in_terms].split('\t')
                #print(terms_up_list[1])
                #print(terms_up_list)
                terms_down_list = out_from_file_with_terms.split('\n')[i_in_terms + 1].split('\t')
                #print(terms_down_list[1])
                #print(terms_down_list)
                signature = Signature(signature_up_list[0], signature_up_list[2:], signature_down_list[2:],
                                      terms_up_list[2:], terms_down_list[2:])
                signature_list.append(signature)
    return signature_list


def cosine_distance(query_genes, pair_genes, list_inf_score):
    """
    Calculates a cosine_distance between vector corresponding genes for query signature and vector corresponding genes
    for pair signatures from L1000FWD database
    Parametrs
    ---------
    query_genes: np.array or list
        list genes from query signature (It can be a list of genes with increased expression or a list of genes with
        reduced expression)
    pair_genes: np.array or list
        list genes from pair signature (It can be a list of genes with increased expression or a list of genes with
        reduced expression)
    list_inf_score: np.array or list
        list of influence scores calculated for genes from the query signature
    Return
    ------
    cosine distance between vector corresponding genes for query signature and vector corresponding genes for pair
    signatures from L1000FWD database
    """
    set_intersecting_query_pair_genes = set(query_genes) & set(pair_genes)#find common genes
    list_genes_in_pair_not_in_query = list(set(pair_genes) - set_intersecting_query_pair_genes)#find genes that are in pair signature, not in query signature
    # create gene space: there are query genes in the beginning of vecor, then genes that are in pair signature, but not in query signature
    # vector_space = np.hstack((query_genes, np.array(list_genes_in_pair_not_in_query)))

    # create vector of weights: list of inf_scores is corresponding to genes in query, list of ones is corresponding genes that are only in pair signatures
    vector_weights = np.hstack((list_inf_score, np.ones(len(list_genes_in_pair_not_in_query))))

    # find vector query in gene space: there are only query genes, so there are ones in the beginning of the vector, corresponding to space of query genes
    vector_query = np.hstack((np.ones(len(query_genes)), np.zeros(len(list_genes_in_pair_not_in_query))))
    # find pair vector in gene space:
    first_part_of_vector_pair = np.zeros(len(query_genes))# first check the presence of query genes in pair signature
    for i in range(len(query_genes)):
        if query_genes[i] in pair_genes:
            first_part_of_vector_pair[i] = 1
    # join the first part of vector, corresponding to space of query genes and second part of vector, corresponding space of genes, that are in pair signature, but not in query signature
    vector_pair = np.hstack((first_part_of_vector_pair, np.ones(len(list_genes_in_pair_not_in_query))))
    cosine_distance = 1 - cosine(vector_pair, vector_query, vector_weights)
    return cosine_distance


def find_cosine_dist(pair, query_signature, list_inf_score_up, list_inf_score_down):
    """
    Calculates score for query signature and pair of signatures from L1000FWD database based on cosine distance

    Parametrs
    ---------
    pair : instance of the class Signature_pair
        pair of signatures from L1000FWD database
    query_signature : instance of the class Signature
        request signature
    list_inf_score_up : np.array
        list of influence scores calculated for genes with increased expression from the query signature
    list_inf_score_down : np.array
        ist of influence scores calculated for genes with reduced expression from the query signature

    Return
    ------
    cosine distance calculated as the average between:
    1)cosine distance between vector corresponding genes for decreased expression of the request signature and vector
    corresponding genes for increased expression of the pair of signatures from L1000FWD database
    2)cosine distance between vector corresponding genes for increased expression of the request signature and vector
    corresponding genes for decreased expression of the pair of signatures from L1000FWD database
    """
    pair_up_genes, pair_down_genes = pair.get_up_down_genes()
    #print('up_genes', pair_up_genes[:10])
    cosine_distance_1 = cosine_distance(query_signature.down_genes, pair_up_genes, list_inf_score_down)
    cosine_distance_2 = cosine_distance(query_signature.up_genes, pair_down_genes, list_inf_score_up)
    mean_cosine_distance = (cosine_distance_1 + cosine_distance_2) / 2


    return mean_cosine_distance


def tanimoto_coeff(list_1, list_2):
    set_1 = set(list_1)
    set_2 = set(list_2)
    intersection = set_1 & set_2
    return len(intersection) / (len(set_1) + len(set_2) - len(intersection))


def find_tanimoto_coeff(pair, query_signature):
    pair_up_terms, pair_down_terms = pair.get_up_down_terms()
    mean_tanimoto_coeff = (tanimoto_coeff(query_signature.down_terms, pair_up_terms) +
                           tanimoto_coeff(query_signature.up_terms, pair_down_terms)) / 2
    return mean_tanimoto_coeff


def find_intersection_entire_pair_query_for_up_or_down(sign_1_terms, sign_2_terms, query_signature_terms):
    intersection_sign_1 = set(sign_1_terms) & set(query_signature_terms)
    intersection_sign_2 = set(sign_2_terms) & set(query_signature_terms)
    intersection = intersection_sign_1 & intersection_sign_2
    return len(intersection)

"""
def find_intersection_entire_pair_query(sign_1, sign_2, query_signature):
    number_intersection_down = find_intersection_entire_pair_query_for_up_or_down(sign_1.down_terms, sign_2.down_terms, query_signature.up_terms)
    number_intersection_up = find_intersection_entire_pair_query_for_up_or_down(sign_1.up_terms, sign_2.up_terms, query_signature.down_terms)
    return (number_intersection_down + number_intersection_up) / 2
"""

def find_intersection_entire_pair_for_up_or_down(sign_1_terms, sign_2_terms):
    intersection = set(sign_1_terms) & set(sign_2_terms)
    return len(intersection)

"""
def find_intersection_entire_pair(sign_1, sign_2):
    number_intersection_down = find_intersection_entire_pair_for_up_or_down(sign_1.down_terms, sign_2.down_terms)
    number_intersection_up = find_intersection_entire_pair_for_up_or_down(sign_1.up_terms, sign_2.up_terms)
    return (number_intersection_down + number_intersection_up) / 2
"""
def calculate_mutual_info_score(sign_1_terms, sign_2_terms, query_signature_terms):
    intersection_sign_1 = set(sign_1_terms) & set(query_signature_terms)
    intersection_sign_2 = set(sign_2_terms) & set(query_signature_terms)
    if (len(intersection_sign_1) != 0) & (len(intersection_sign_2) != 0):
        min_length = min(len(intersection_sign_1), len(intersection_sign_2))
        intersection = set(intersection_sign_1) & set(intersection_sign_2)
        intersection_sign_1_same_length = list(intersection)[:] + list(intersection_sign_1 - intersection)[:min_length - len(intersection)]
        intersection_sign_2_same_length = list(intersection)[:] + list(intersection_sign_2 - intersection)[:min_length - len(intersection)]
        mutual_info_score_value = mutual_info_score(intersection_sign_1_same_length, intersection_sign_2_same_length)
    else:
        mutual_info_score_value = np.nan
    return mutual_info_score_value

def find_mutual_info_score(sign_1, sign_2, query_signature):
    mutual_info_score_down = calculate_mutual_info_score(sign_1.down_terms, sign_2.down_terms, query_signature.up_terms)
    mutual_info_score_up = calculate_mutual_info_score(sign_1.up_terms, sign_2.up_terms, query_signature.down_terms)
    if (not np.isnan(mutual_info_score_up)) & (not np.isnan(mutual_info_score_down)):
        mean_mutual_info_score = (mutual_info_score_up + mutual_info_score_down) / 2
    elif (not np.isnan(mutual_info_score_up)) & (np.isnan(mutual_info_score_down)):
        mean_mutual_info_score = mutual_info_score_up
    elif (np.isnan(mutual_info_score_up)) & (not np.isnan(mutual_info_score_down)):
        mean_mutual_info_score = mutual_info_score_down
    else :
        mean_mutual_info_score = np.nan
    return mean_mutual_info_score


#write func for multiprocessing
def compare_multiprocessing(i, j, query_signature, list_inf_score_up, list_inf_score_down, signature_list, list_metrics):
    """
    Return score for query signature and pair of signatures from L1000FWD database based on cosine distance

    Parametrs
    ---------
    i: int
        the number of the first signature of the pair in the signature list
    j: int
        the number of the second signature of the pair in the signature list
    query_signature: instance of the class Signature
        query signature
    list_inf_score_up : np.array
            list of influence scores calculated for genes with increased expression from the query signature
    list_inf_score_down : np.array
            ist of influence scores calculated for genes with reduced expression from the query signature
    signature_list: list
        list of signatures from L1000FWD(instances of the class Signature)
    Return
    Return synergy score for query signature and pair of signatures from L1000FWD database based on cosine distance
    """
    start_time = time.time()

    pair = Signature_pair(signature_list[i], signature_list[j])
    return_list = [i, j]
    if 'cosine_dist' in list_metrics:
        cosine_dist = find_cosine_dist(pair, query_signature, list_inf_score_up, list_inf_score_down)
        return_list.append(('cosine_dist', cosine_dist))
    if 'tanimoto_coeff' in list_metrics:
        tanimoto_coeff = find_tanimoto_coeff(pair, query_signature)
        return_list.append(('tanimoto_coeff', tanimoto_coeff))
    if 'mutual_info_coeff' in list_metrics:
        mutual_info_coeff = find_mutual_info_score(signature_list[i], signature_list[j], query_signature)
        return_list.append(('mutual_info_coeff', mutual_info_coeff))
    if 'intersection_terms_of_pair_query' in list_metrics:
        intersection_entire_pair_query = find_intersection_entire_pair_query(signature_list[i], signature_list[j], query_signature)
        return_list.append(('intersection_terms_of_pair_query', intersection_entire_pair_query))
    if 'intersection_terms_of_pair' in list_metrics:
        intersection_entire_pair = find_intersection_entire_pair(signature_list[i], signature_list[j])
        return_list.append(('intersection_terms_of_pair', intersection_entire_pair))
    #print('косинусное расстояние :', find_cosine_dist(pair, query_signature, list_inf_score_up, list_inf_score_down))
    #print('время работы поиска косинусного расстояния для одной пары:',     '--- %s seconds ---' % (time.time() - start_time))
    return tuple(return_list)


def find_similarity(content_of_file_with_signatures, df_inf_score, number_processes, path_to_file_with_query_terms, path_to_file_with_terms,
                    presence_file_with_query_terms, presence_file_with_terms, list_metrics):
    """
    Сounts the score based on cosine distance for request signature and pair of signatures
    running through all possible pairs of signatures from L1000FWD database

    Parametrs
    ---------
    content_of_file_with_signatures: str
        content of the file with signatures of L1000FWD database
    df_inf_score: DataFrame
        DataFrame that lists the genes from the request signature
        The dataframe contains influence score of gene calculated by topological metrics and logFC .
    number_processes: int
        desired number of processes for parallelization
    Return
    ------
    DataFrame whose column names and row indexes are signature IDs. The each element of dataframe represents
    score based on cosine distance for request signature and pair of signatures.
    """
    list_signature_up_genes = np.array(df_inf_score.loc['up'].index)
    list_inf_score_up = np.array(df_inf_score.loc['up']['inf_score'])
    list_signature_down_genes = np.array(df_inf_score.loc
                                         ['down'].index)
    list_inf_score_down = np.array(df_inf_score.loc['down']['inf_score'])

    if presence_file_with_query_terms != 'no':
        with open(path_to_file_with_query_terms, "r") as file:
            list_up_down_terms = file.read().split('\n')
            up_terms = list_up_down_terms[0].split('\t')[2:]
            down_terms = list_up_down_terms[1].split('\t')[2:]
        query_signature = Signature('query', list_signature_up_genes, list_signature_down_genes, up_terms, down_terms)
    else:
        query_signature = Signature('query', list_signature_up_genes, list_signature_down_genes)
        query_signature.set_up_terms()
        query_signature.set_down_terms()
        with open(path_to_file_with_query_terms, "w") as file:
            file.write('query\t' + 'up_terms\t' + '\t'.join(query_signature.up_terms) + '\n')
            file.write('query\t' + 'down_terms\t' + '\t'.join(query_signature.down_terms) + '\n')
    print("создали сигнатуру запроса")
    signature_list = create_signature_list(content_of_file_with_signatures, path_to_file_with_terms, presence_file_with_terms)
    signature_id_list = [signature.id for signature in signature_list]

    dict_metics_matrix = {}
    for name in list_metrics:
        zeros_array = np.ones(shape=(len(signature_list), len(signature_list)))
        matrix = pd.DataFrame(zeros_array)
        matrix.index = signature_id_list
        matrix.columns = signature_id_list
        dict_metics_matrix[name] = matrix

    list_metric_name_with_matrix = []
    print("приступаем к распараллеливанию")
    """
    results = Parallel(n_jobs = number_processes)(delayed(cosine_dist_for_multiprocessing)(i, j, query_signature,
                            list_inf_score_up, list_inf_score_down, signature_list) for i in range(len(signature_list))
        for j in range(len(signature_list)) if i < j)
    """
    with Pool(processes=number_processes) as pool:
        results = pool.starmap(compare_multiprocessing,
                               [(i, j, query_signature, list_inf_score_up, list_inf_score_down, signature_list, list_metrics) for
                                i in range(len(signature_list)) for j in range(len(signature_list)) if i < j])

    for res in results:
        i = res[0]
        j = res[1]
        for (metric_name, metric_value) in res[2:]:
            dict_metics_matrix[metric_name].iloc[i,j] = metric_value
    for metric_name in list_metrics:
        list_metric_name_with_matrix.append((metric_name, dict_metics_matrix[metric_name]))
    return tuple(list_metric_name_with_matrix)




def find_near_signatures(content_of_file_with_signatures, cosine_dist_matrix, n, df_with_signature_id_pert_id):
    """
    finds the closest pair of signature

    Parametrs
    ---------
    content_of_file_with_signatures : str
        content of the file with signatures of L1000FWD database
    cosine_dist_matrix : DataFrame
        DataFrame whose column names and row indexes are signature IDs. The each element of dataframe represents
        score based on cosine distance for request signature and pair of signatures.
    n : int
        the number of closest pairs of signatures
    df_with_signature_id_pert_id: DataFrame
        DataFrame that contains the signature ID and perturbation ID

    Return
    ------
    DataFrame that contains the signature ID of closest pair of signature (their score) and their corresponding perturbation ID, name
    """
    signature_list = create_signature_list(content_of_file_with_signatures)

    signature_id_list = []
    for signature in signature_list:
        signature_id_list.append(signature.id)
    list_pair_signatures_id = []
    list_pair_pert_id = []
    list_pair_pert_desc = []
    list_score = []
    rank_array = rankdata(cosine_dist_matrix, method='dense')
    for i in range(np.min(rank_array), np.min(rank_array) + n, 1):
        for j in range(len(rank_array)):
            if rank_array [j] == i:
                list_pair_signatures_id.append(signature_id_list[j//len(signature_id_list)] + ';' +
                                               signature_id_list[j%len(signature_id_list)])

                list_pair_pert_id.append(df_with_signature_id_pert_id.loc[signature_id_list[j//len(signature_id_list)],
                'pert_id']+ ";" + df_with_signature_id_pert_id.loc[signature_id_list[j%len(signature_id_list)], 'pert_id'])

                list_pair_pert_desc.append(df_with_signature_id_pert_id.loc[signature_id_list[j//len(signature_id_list)],
                'pert_desc']+ ";" + df_with_signature_id_pert_id.loc[signature_id_list[j%len(signature_id_list)], 'pert_desc'])

                list_score.append(cosine_dist_matrix.loc[signature_id_list[j//len(signature_id_list)],
                                                         signature_id_list[j%len(signature_id_list)]])
    dict_with_signatures_pert_id = {}
    dict_with_signatures_pert_id['sign_id'] = list_pair_signatures_id
    dict_with_signatures_pert_id['pert_id'] = list_pair_pert_id
    dict_with_signatures_pert_id['pert_desc'] = list_pair_pert_desc
    dict_with_signatures_pert_id['score'] = list_score
    df_with_signatures_pert_id = pd.DataFrame(dict_with_signatures_pert_id)
    return df_with_signatures_pert_id



if __name__ == '__main__':

    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    out_of_file_with_signatures = namespace.path_to_file_with_signatures.read()
    score_up = pd.read_csv(namespace.path_to_file_with_inf_score_up, index_col = 0)
    score_down = pd.read_csv(namespace.path_to_file_with_inf_score_down, index_col = 0)
    df_topo_score = concat_df_log_FC_topo_score_normalize(score_up, score_down)
    df_CD_signature_metadata = pd.read_csv(namespace.path_to_file_with_CD_signature_metadata, index_col = 0)
    dict_multiplication_factor = {'logFC': 1, 'betweenness': 1, 'pagerank': 1, 'closeness': 1, 'katz': 1, 'hits_authority':
        1, 'hits_hub': 1, 'eigenvector': 1, 'eigentrust': 1}
    dict_additive_factor = {'logFC': 1, 'betweenness': 1, 'pagerank': 1, 'closeness': 1, 'katz': 1, 'hits_authority':
        1, 'hits_hub': 1, 'eigenvector': 1, 'eigentrust': 1}
    df_inf_score = calculate_inf_score(df_topo_score, func_inf_score_v1, dict_multiplication_factor,
                                       dict_additive_factor)
    print("Все прочитали и начинаем работать")
    start_time = time.time()
    total_start_time = time.time()
    df_cosine_dist_matrix = cosine_similarity(out_of_file_with_signatures, df_inf_score, namespace.number_processes)
    print('время работы функции поиска косинусного расстояния:', '--- %s seconds ---' % (time.time() - start_time))
    df_cosine_dist_matrix.to_csv(namespace.path_to_dir_save_results + '/cosine_dist_matrix_' + namespace.conversion + '.csv', columns = df_cosine_dist_matrix.columns, index=True)
    start_time = time.time()
    df_with_signatures_pert_id = find_near_signatures(out_of_file_with_signatures, df_cosine_dist_matrix, namespace.number_pair_signatures,
                                                      df_CD_signature_metadata)
    print('время отбора пар сигнатур:', '--- %s seconds ---' % (time.time() - start_time))
    df_with_signatures_pert_id.to_csv(namespace.path_to_dir_save_results + '/closest_pair_sign_id_pert_id_pert_name_score_' + namespace.conversion + '.csv', columns = df_with_signatures_pert_id.columns)
    print('полное время работы:', '--- %s seconds ---' % (time.time() - total_start_time))
    #we are in rework