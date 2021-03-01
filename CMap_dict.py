import time #to calculate the time
import numpy as np
import pandas as pd
#from collections import defaultdict
from scipy.spatial.distance import cosine
from scipy.stats import rankdata
import argparse #read arguments from the command line
import sys
from multiprocessing import Pool
#from numba import njit
#from numba.typed import Dict
#from numba.core import types
#from create_weight_vector import test_f
#import dill
#from cython.parallel import prange


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
    def __init__(self, id, up_genes, down_genes):
        self.id = id
        self.up_genes = up_genes
        self.down_genes = down_genes


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

    def get_number_signatures(self):
        return (self.signature_1.number_in_signature_list, self.signature_2.number_in_signature_list)

    def get_up_down(self):
        up_1 = set(self.signature_1.up_genes)
        up_2 = set(self.signature_2.up_genes)
        down_1 = set(self.signature_1.down_genes)
        down_2 = set(self.signature_2.down_genes)
        up = up_1 | up_2
        down = down_1 | down_2
        if (up & down):
            for gene in (up & down):
                up.discard(gene)
                down.discard(gene)
        return (list(up), list(down))

    def get_up(self):
        up_down = self.get_up_down()
        return up_down[0]

    def get_down(self):
        up_down = self.get_up_down()
        return up_down[1]

class Gene_vector:
    """
    Gene_vector class is used to create a vector based on a list of genes and a gene space.

    Attributes
    ----------
    gene_list : list
        list of genes in signature
    gene_space : list
        list of genes that define the gene space

    Methods
    -------
    coordinates()
        returns the coordinates of the vector corresponding to the list of genes in the gene space
    """

    def __init__(self, gene_list, gene_space):
        self.gene_list = gene_list
        self.gene_space = gene_space
    def coordinates(self):
        coordinate = []
        for gene in self.gene_space:
            if gene in self.gene_list:
                coordinate.append(1)
            else:
                coordinate.append(0)
        return coordinate

def create_signature_list(out_from_file_with_signatures):
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
    for i in range(0, len(out_from_file_with_signatures.split('\n'))-1, 2):
        signature_up_list = out_from_file_with_signatures.split('\n')[i].split('\t')
        signature_down_list = out_from_file_with_signatures.split('\n')[i+1].split('\t')
        signature = Signature(signature_up_list[0], signature_up_list[2:], signature_down_list[2:])
        signature_list.append(signature)
    return signature_list

def create_gene_space_and_weight_vector(signature_genes, pair_genes, list_inf_score):
    set_intersecting_signature_pair_genes = set(signature_genes) & set(pair_genes)
    genes_in_pair_not_in_signature = set(pair_genes) - set_intersecting_signature_pair_genes
    space = signature_genes + list(genes_in_pair_not_in_signature)
    vector_weights = list_inf_score + [1] * len(genes_in_pair_not_in_signature)
    return (space, vector_weights)


def find_cosine_dist(pair, query_signature, list_inf_score_up, list_inf_score_down):
    """
    Calculates score for request signature and pair of signatures from L1000FWD database based on cosine distance

    Parametrs
    ---------
    pair : instance of the class Signature_pair
        pair of signatures from L1000FWD database
    query_signature : instance of the class Signature
        request signature
    df_inf_score_up : DataFrame
        DataFrame that lists the overexpression genes from the request signature
        The dataframe contains influence score of gene calculated by topological metrics and logFC .
    df_inf_score_down : DataFrame
        DataFrame that lists the genes with reduced expression from the request signature
        The dataframe contains influence score of gene calculated by topological metrics and logFC.

    Return
    ------
    cosine distance calculated as the average between:
    1)cosine distance between vector corresponding genes for decreased expression of the request signature and vector
    corresponding genes for increased expression of the pair of signatures from L1000FWD database
    2)cosine distance between vector corresponding genes for increased expression of the request signature and vector
    corresponding genes for decreased expression of the pair of signatures from L1000FWD database
    """
    start = time.time()
    (space_1, vector_inf_score_space_1) = create_gene_space_and_weight_vector(query_signature.down_genes, pair.get_up(), list_inf_score_down)
    print('время выполнения создания пространства и вектора весов: ', time.time() -start)
    start = time.time()
    pair_up_vector = Gene_vector(pair.get_up(), space_1)
    print('время создания вектора пары: ', time.time() - start)
    start = time.time()
    query_down_vector = Gene_vector(query_signature.down_genes, space_1)
    print('время создания вектора сигнатуры: ', time.time() - start)
    start = time.time()
    cosine_distance_1 = cosine(pair_up_vector.coordinates(), query_down_vector.coordinates(), vector_inf_score_space_1)
    print('время вычисления  косинусного расстояния для одной половинки: ', time.time() - start)


    (space_2, vector_inf_score_space_2) = create_gene_space_and_weight_vector(query_signature.up_genes, pair.get_down(), list_inf_score_up)
    pair_down_vector = Gene_vector(pair.get_down(), space_2)
    query_up_vector = Gene_vector(query_signature.up_genes, space_2)
    cosine_distance_2 = cosine(pair_down_vector.coordinates(), query_up_vector.coordinates(), vector_inf_score_space_2)
    return (cosine_distance_1 + cosine_distance_2) / 2


#write func for multiprocessing
def cosine_dist_for_multiprocessing(i, j, query_signature, list_inf_score_up, list_inf_score_down, signature_list):
    start_time = time.time()
    pair = Signature_pair(signature_list[i], signature_list[j])
    print('косинусное расстояние :', find_cosine_dist(pair, query_signature, list_inf_score_up, list_inf_score_down))
    print('время работы поиска косинусного расстояния для одной пары:',     '--- %s seconds ---' % (time.time() - start_time))
    return (i, j, find_cosine_dist(pair, query_signature, list_inf_score_up, list_inf_score_down))

    #return matrix

#def f(): return 1

def cosine_similarity(content_of_file_with_signatures, df_inf_score, number_processes):
    """
    Сounts the score based on cosine distance for request signature and pair of signatures
    running through all possible pairs of signatures from L1000FWD database

    Parametrs
    ---------
    content_of_file_with_signatures : str
        content of the file with signatures of L1000FWD database
    df_inf_score_up : DataFrame
        DataFrame that lists the overexpression genes from the request signature
        The dataframe contains influence score of gene calculated by topological metrics and logFC .
    df_inf_score_down : DataFrame
        DataFrame that lists the genes with reduced expression from the request signature
        The dataframe contains influence score of gene calculated by topological metrics and logFC.

    Return
    ------
    DataFrame whose column names and row indexes are signature IDs. The each element of dataframe represents
    score based on cosine distance for request signature and pair of signatures.
    """
    print(df_inf_score)
    list_signature_up_genes = list(df_inf_score.loc['up'].index)
    list_inf_score_up = list(df_inf_score.loc['up']['inf_score'])
    list_signature_down_genes = list(df_inf_score.loc['down'].index)
    list_inf_score_down = list(df_inf_score.loc['down']['inf_score'])


    query_signature = Signature('query', list_signature_up_genes, list_signature_down_genes)
    print("создали сигнатуру запроса")
    signature_list = create_signature_list(content_of_file_with_signatures)
    signature_id_list = [signature.id for signature in signature_list]


    zeros_array = np.ones(shape=(len(signature_list), len(signature_list)))
    cosine_dist_matrix = pd.DataFrame(zeros_array)
    cosine_dist_matrix.index = signature_id_list
    cosine_dist_matrix.columns = signature_id_list

    print("приступаем к распараллеливанию")
    pool = Pool(processes = number_processes)
    results = pool.starmap(cosine_dist_for_multiprocessing, [(i, j, query_signature, list_inf_score_up, list_inf_score_down, signature_list) for i in range(len(signature_list)) for j in range(len(signature_list)) if i < j])
    for (i,j, cos_distance) in results:
        cosine_dist_matrix.iloc[i,j] = cos_distance
    return cosine_dist_matrix

    """
    zeros_array = np.zeros(shape=(len(signature_list), len(signature_list)))


    print("приступаем к распераллеливанию")
    pool = Pool(processes=5)
    results = pool.starmap(cosine_dist_for_multiprocessing, [
        (i, j, query_signature, df_inf_score_up, df_inf_score_down, np.zeros(shape=(len(signature_list),
        len(signature_list))), signature_list) for i in range(len(signature_list)) for j in range(len(signature_list)) if i < j])
    for matrix in results:
        #print(matrix)
        zeros_array = zeros_array + matrix
    zeros_array = zeros_array + np.tril(np.ones(cosine_dist_matrix.shape), 0)
    cosine_dist_matrix = pd.DataFrame(zeros_array)
    cosine_dist_matrix.index = signature_id_list
    cosine_dist_matrix.columns = signature_id_list
    pool.close()
    print(cosine_dist_matrix.shape)
    print(cosine_dist_matrix)
    return cosine_dist_matrix
    """
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
    inf_up = pd.read_csv(namespace.path_to_file_with_inf_score_up, index_col = 0)
    inf_down = pd.read_csv(namespace.path_to_file_with_inf_score_down, index_col = 0)
    df_CD_signature_metadata = pd.read_csv(namespace.path_to_file_with_CD_signature_metadata, index_col = 0)
    print("Все прочитали и начинаем работать")
    start_time = time.time()
    total_start_time = time.time()
    df_cosine_dist_matrix = cosine_similarity(out_of_file_with_signatures, inf_up, inf_down, namespace.number_processes)
    print('время работы функции поиска косинусного расстояния:', '--- %s seconds ---' % (time.time() - start_time))
    df_cosine_dist_matrix.to_csv(namespace.path_to_dir_save_results + '/cosine_dist_matrix_' + namespace.conversion + '.csv', columns = df_cosine_dist_matrix.columns, index=True)
    start_time = time.time()
    df_with_signatures_pert_id = find_near_signatures(out_of_fgiile_with_signatures, df_cosine_dist_matrix, namespace.number_pair_signatures,
                                                      df_CD_signature_metadata)
    print('время отбора пар сигнатур:', '--- %s seconds ---' % (time.time() - start_time))
    df_with_signatures_pert_id.to_csv(namespace.path_to_dir_save_results + '/closest_pair_sign_id_pert_id_pert_name_score_' + namespace.conversion + '.csv', columns = df_with_signatures_pert_id.columns)
    print('полное время работы:', '--- %s seconds ---' % (time.time() - total_start_time))
    #we are in rework