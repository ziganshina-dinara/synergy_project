import time #to calculate the time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import rankdata
import argparse #read arguments from the command line
import sys

def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument('-topolog_metrics_up', '--path_to_file_with_df_topolog_metrics_up', type = str)
    parser.add_argument('-topolog_metrics_down', '--path_to_file_with_df_topolog_metrics_down', type=str)
    parser.add_argument('-signatures', '--path_to_file_with_signatures', type=argparse.FileType())
    parser.add_argument('-cosine_dist_matrix', '--path_to_file_with_cosine_dist_matrix', type=str)
    parser.add_argument('-list_signatures_pairs', '--path_to_file_with_list_signatures_pairs', type=str)
    return parser


class Inf_scores_signature:
    list_metric = ['betweenness', 'pagerank', 'closeness', 'katz', 'hits_authority', 'hits_hub', 'eigenvector']
    dict_coefficient = {}
    for metric in list_metric:
        dict_coefficient[metric] = 1

    def __init__(self, df_topolog_metrics_up, df_topolog_metrics_down):
        self.df_topolog_metrics_up = df_topolog_metrics_up
        self.df_topolog_metrics_down = df_topolog_metrics_down
        self.df_topolog_metrics = pd.concat([df_topolog_metrics_up, df_topolog_metrics_down])
        self.genes = pd.concat([df_topolog_metrics_up, df_topolog_metrics_down]).index

    def get_inf_score(self, gene):
        df_with_topolog_metrics = self.df_topolog_metrics
        if gene in self.genes:
            inf_score = 0
            for metric in self.list_metric:
                if np.isnan(df_with_topolog_metrics.loc[gene, metric]):
                    inf_score += self.dict_coefficient[metric] * df_with_topolog_metrics.loc[gene, metric]
                else:
                    return 1
            return inf_score
        else:
            return 0

class Signature:
    def __init__(self, id, up_genes, down_genes ):
        self.id = id
        self.up_genes = up_genes
        self.down_genes = down_genes


class Signature_pair:

    def __init__(self, signature_1, signature_2):
        self.signature_1 = signature_1
        self.signature_2 = signature_2

    def get_id_signatures(self):
        return (self.signature_1.id, self.signature_2.id)

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
    signature_list = []
    for i in range(0, 50, 2): #range(len(out_from_file_with_signatures.split('\n'))-1):
        signature_up_list = out_from_file_with_signatures.split('\n')[i].split('\t')
        signature_down_list = out_from_file_with_signatures.split('\n')[i+1].split('\t')
        signature = Signature(signature_up_list[0], signature_up_list[2:], signature_down_list[2:])
        signature_list.append(signature)
    return signature_list

def create_inf_score_as_weights_vector(df_topolog_metrics_up, df_topolog_metrics_down, space):
    inf_scores = Inf_scores_signature(df_topolog_metrics_up, df_topolog_metrics_down)
    inf_score_as_weights_vector = []
    for gene in space:
        inf_score_as_weights_vector.append(inf_scores.get_inf_score(gene))
    return inf_score_as_weights_vector

def find_cosine_dist(pair,query_signature, df_topolog_metrics_up, df_topolog_metrics_down):

    space_1 = set(query_signature.down_genes) | set(pair.get_up())
    pair_up_vector = Gene_vector(pair.get_up(), space_1)
    query_down_vector = Gene_vector(query_signature.down_genes, space_1)
    vector_topol_metrics_space_1 = create_inf_score_as_weights_vector(df_topolog_metrics_up,
                                                                      df_topolog_metrics_down, space_1)
    cosine_distance_1 = cosine(pair_up_vector.coordinates(), query_down_vector.coordinates(), vector_topol_metrics_space_1)
    #print(pair_up_vector.coordinates())
    #print(query_down_vector.coordinates())

    space_2 = set(query_signature.down_genes) | set(pair.get_down())
    pair_down_vector = Gene_vector(pair.get_down(), space_2)
    query_up_vector = Gene_vector(query_signature.up_genes, space_2)
    vector_topol_metrics_space_2 = create_inf_score_as_weights_vector(df_topolog_metrics_up,
                                                                      df_topolog_metrics_down, space_2)
    cosine_distance_2 = cosine(pair_down_vector.coordinates(), query_up_vector.coordinates(), vector_topol_metrics_space_2)
    return (cosine_distance_1 + cosine_distance_2) / 2

def cosine_similarity(content_of_file_with_signatures, df_topolog_metrics_up, df_topolog_metrics_down):
    query_signature = Signature('query', df_topolog_metrics_up.index, df_topolog_metrics_down.index)
    signature_list = create_signature_list(content_of_file_with_signatures)
    signature_id_list = []
    for signature in signature_list:
        signature_id_list.append(signature.id)


    zero_array = np.zeros(shape=(len(signature_list), len(signature_list)))
    cosine_dist_matrix = pd.DataFrame(zero_array)
    cosine_dist_matrix.index = signature_id_list
    cosine_dist_matrix.columns = signature_id_list

    for i in range(len(signature_list)-1):
        for j in range(i,len(signature_list)):
            pair = Signature_pair(signature_list[i], signature_list[j])
            cosine_dist = find_cosine_dist(pair, query_signature, df_topolog_metrics_up, df_topolog_metrics_down)
            cosine_dist_matrix.loc[signature_list[i].id, signature_list[j].id] = cosine_dist
    return cosine_dist_matrix

def find_near_signatures(content_of_file_with_signatures, cosine_dist_matrix, n):
    signature_list = create_signature_list(content_of_file_with_signatures)
    signature_id_list = []
    for signature in signature_list:
        signature_id_list.append(signature.id)
    list_pair_signatures_id = []
    rank_array = rankdata(cosine_dist_matrix, method='dense')
    for i in range(np.max(rank_array), np.max(rank_array) - n, -1):
        for j in range(len(rank_array)):
            if rank_array [j] == i:
                list_pair_signatures_id.append(signature_id_list[j//len(signature_id_list)] + '\t' + signature_id_list[j%len(signature_id_list)])
    return list_pair_signatures_id



if __name__ == '__main__':

    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    out_of_file_with_signatures = namespace.path_to_file_with_signatures.read()
    topolog_metrics_up = pd.read_csv(namespace.path_to_file_with_df_topolog_metrics_up, index_col = 0)
    topolog_metrics_down = pd.read_csv(namespace.path_to_file_with_df_topolog_metrics_down, index_col = 0)
    df_cosine_dist_matrix = cosine_similarity(out_of_file_with_signatures, topolog_metrics_up, topolog_metrics_down)
    df_cosine_dist_matrix.to_csv(namespace.path_to_file_with_cosine_dist_matrix, columns=df_cosine_dist_matrix.columns, index=True)
    with open(namespace.path_to_file_with_list_signatures_pairs, "w") as file:
        file.write('\n'.join(find_near_signatures(out_of_file_with_signatures,df_cosine_dist_matrix, 50)))