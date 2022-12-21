import sys
import time
import argparse  # read arguments from the command line
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
import requests  # for make API requests
from requests.exceptions import HTTPError  # to handle the responses
import graph_tool as gt  # for make protein networks
from graph_tool import centrality as ct
from graph_tool.draw import graph_draw

from function_signature_from_DE_v1 import make_signature_from_DE


def createParser ():
    """
    script parameters parser

    Return
    ------
    instance of the class ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-up', '--path_to_file_with_up_genes', type=argparse.FileType())
    parser.add_argument('-down', '--path_to_file_with_down_genes', type=argparse.FileType())
    parser.add_argument('-DE', '--path_to_file_with_DE', type=str)
    parser.add_argument('-logFC', '--logFC_threshold', default=1.5, type=float)
    parser.add_argument('-pvalue', '--pvalue_threshold', default=0.01, type=float)
    parser.add_argument('-dir_results', '--path_to_dir_save_results', default='DATA', type=str)
    parser.add_argument('-sp', '--species', default = 9606, type=int)
    parser.add_argument('-exp_thr', '--experimental_score_threshold', default=0.4, type=float)
    parser.add_argument('-source', '--source_type_cell', type=str)
    parser.add_argument('-target', '--target_type_cell', type=str)
    return parser


class PPI_numpy_array:
    """
    class PPI_numpy_array is designed to represent STRING protein-protein interaction network in adjacency matrix

    Attributes
    ----------
    gene_set: a list of genes with increased expression or list of genes with decreased expression
    species: NCBI taxon identifiers (e.g. Human is 9606)
    experimental_score_threshold: threshold for combined score. It's computed by combining the probabilities from
    the different evidence channels and corrected for the probability of randomly observing an interaction.

    Methods
    -------
    API_request()
        requests STRING interaction network for multiple proteins

    get_dict_genes()
        returns a dictionary where the keys are the name of the genes and the values are gene numbers

    get_dict_number_genes()
        returns a dictionary where the keys are gene numbers and the values are the name of the genes

    get_interactions_as_adjacency_matrix()
        builds an adjacency matrix for the interaction network

    """

    def __init__(self, gene_set, species, experimental_score_threshold):
        self.gene_set = gene_set
        self.species = species
        self.experimental_score_threshold = experimental_score_threshold

    def API_request(self):

        string_api_url = 'https://string-db.org/api'
        output_format = 'tsv-no-header'
        method = 'network'

        request_url = '/'.join([string_api_url, output_format, method])
        params = {

            'identifiers': '%0d'.join(self.gene_set),  # your protein
            'species': self.species,  # species NCBI identifier
            'caller_identity': 'www.awesome_app.org'  # your app name

        }

        response = requests.post(request_url, data=params)
        return response

    def get_dict_genes(self):
        array_numbers = range(len(self.gene_set))
        dict_genes = dict(zip(self.gene_set, array_numbers))
        return dict_genes

    def get_dict_number_genes(self):
        array_numbers = range(len(self.gene_set))
        dict_number_genes = dict(zip(array_numbers, self.gene_set))
        return dict_number_genes

    def get_interactions_as_adjacency_matrix(self):

        np_array_as_adjacency_matrix = np.zeros(shape=(len(self.gene_set), len(self.gene_set)))

        response = self.API_request()
        try:
            response.raise_for_status()
            for line in response.text.strip().split("\n"):

                l = line.strip().split("\t")
                p1, p2 = l[2], l[3]

                ## filter the interaction according to experimental score
                experimental_score = float(l[10])
                if experimental_score > self.experimental_score_threshold:
                    if ((p1 in self.gene_set) & (p2 in self.gene_set)):
                        np_array_as_adjacency_matrix[self.get_dict_genes()[p1], self.get_dict_genes()[p2]] = experimental_score
                        np_array_as_adjacency_matrix[self.get_dict_genes()[p2], self.get_dict_genes()[p1]] = experimental_score
            return np_array_as_adjacency_matrix
        except HTTPError as http_err:
            print(f'some problems with querying the STRING database. More precisely HTTP error occurred: {http_err}')
            print(f'answer database STRING: {response.text}')


# write class for build the PPI graph
class PPI_graph:
    """
    class PPI_graph is designed to represent STRING protein-protein interaction network as a graph

    Attributes
    ----------
    np_array_as_adjacency_matrix: adjacency_matrix of PPI network as np.array
    dict_number_genes: a dictionary where the keys are gene numbers and the values are the name of the genes

    Methods
    -------
    get_graph()
        return graph of PPI network

    draw_PPI_graph()
        draw graph

    save_graph(path_to_file)
        save graph in 'gt' format file

    save_image_graph(path_to_image)
        save image of the graph

    get_dataframe_all_topolog_metrics(path_to_file)
        returns a dataframe in which all topological metrics are calculated for each gene

    """

    def __init__(self, np_array_as_adjacency_matrix, dict_number_genes):
        self.adjacency_matrix = np_array_as_adjacency_matrix
        self.number_proteins = np_array_as_adjacency_matrix.shape[0]
        self.dict_number_genes = dict_number_genes
        self.dict_genes = dict(zip(self.dict_number_genes.values(), self.dict_number_genes.keys()))

    def get_graph(self):
        graph = gt.Graph(directed=False)
        vertex_list = graph.add_vertex(self.number_proteins)
        matrix = np.triu(self.adjacency_matrix)
        graph.add_edge_list((np.transpose(matrix.nonzero())))
        vprop_name_proteins = graph.new_vertex_property('string')
        for i in range(self.number_proteins):
            vprop_name_proteins[i] = self.dict_number_genes[i]
        graph.vertex_properties['name_proteins'] = vprop_name_proteins
        return graph

    def draw_PPI_graph(self):
        graph = self.get_graph()
        graph_draw(graph, vertex_text=graph.vertex_properties['name_proteins'])

    def save_graph(self, path_to_file):
        graph = self.get_graph()
        graph.save(path_to_file)

    def save_image_graph(self, path_to_image):
        graph = self.get_graph()
        graph_draw(graph, vertex_text=graph.vertex_properties['name_proteins'], output=path_to_image)

    def get_dataframe_all_topolog_metrics(self):
        graph = self.get_graph()
        eprop_trust = graph.new_edge_property('double')

        start_time = time.time()
        for e in graph.edges():
            v_name_s = graph.vertex_properties['name_proteins'][e.source()]
            v_number_s = self.dict_genes[v_name_s]
            v_name_t = graph.vertex_properties['name_proteins'][e.target()]
            v_number_t = self.dict_genes[v_name_t]
            eprop_trust[e] = self.adjacency_matrix[v_number_s, v_number_t]
        graph.edge_properties['trust'] = eprop_trust
        print('confidence score за :', '--- %s seconds ---' % (time.time() - start_time))

        list_metrics = ['betweenness', 'pagerank', 'closeness', 'katz',  'eigenvector',
                        'eigentrust'] # 'trust_transitivity', 'hits_authority', 'hits_hub',

        dict_map = {}
        start_time = time.time()
        dict_map['betweenness'] = ct.betweenness(graph)[0]
        dict_map['pagerank'] = ct.pagerank(graph)
        dict_map['closeness'] = ct.closeness(graph)
        dict_map['katz'] = ct.katz(graph)
        #dict_map['hits_authority'] = ct.hits(graph)[1]
        #dict_map['hits_hub'] = ct.hits(graph)[2]
        dict_map['eigenvector'] = ct.eigenvector(graph)[1]
        #print('trust_transitivity')
        #"dict_map['trust_transitivity'] = ct.trust_transitivity(graph,  graph.edge_properties["trust"])
        print('все метрики кроме eigentrust за :', '--- %s seconds ---' % (time.time() - start_time))
        start_time = time.time()
        dict_map['eigentrust'] = ct.eigentrust(graph, graph.edge_properties['trust'], max_iter = 10**6)
        print('eigentrust за :', '--- %s seconds ---' % (time.time() - start_time))
        start_time = time.time()
        dict_metrics = {}
        for key in list_metrics:
            dict_metrics[key] = []
        for v in graph.vertices():
            for metric in list_metrics:
                dict_metrics[metric].append(dict_map[metric][v])
        dataframe_all_topolog_metrics = pd.DataFrame(dict_metrics)
        dataframe_all_topolog_metrics.index = graph.vertex_properties['name_proteins']
        print('получила датафрейм с метриками за :', '--- %s seconds ---' % (time.time() - start_time))
        return dataframe_all_topolog_metrics


def create_df_gene_topo_scores_logFC(series_gene_logFC, dataframe_all_topolog_metrics_for_gene_STRING):
    """
        creates a dataframe that contains logFC, all topological metrics

        Parameters
        ---------
        series_gene_logFC : Series
            Series that contains logFC for each gene
        dataframe_all_topolog_metrics_for_gene_STRING : DataFrame
            dataframe in which all topological metrics are calculated for each gene
        Return
        ------
        DataFrame that contains logFC, all topological metrics  and influence score calculated by logFC
        and topological metrics
    """
    df_gene_logFC = pd.DataFrame(series_gene_logFC)
    df_logFC_topo_scores = df_gene_logFC.merge(dataframe_all_topolog_metrics_for_gene_STRING, how='left',
                                                                                left_index=True, right_index=True)

    return df_logFC_topo_scores


def create_df_gene_logFC_topo_score_from_beginning(gene_set, species, experimental_score_threshold, series_genes):
    """
        creates a dataframe that contains logFC, all topological metrics for differentially expressed genes

        Parameters
        ---------
        gene_set : list
            a list of genes with increased expression or list of genes with decreased expression
        species : int
            NCBI taxon identifiers (e.g. Human is 9606)
        experimental_score_threshold : float
            threshold for combined score. It's computed by combining the probabilities from
            the different evidence channels and corrected for the probability of randomly observing an interaction.
        series_genes : Series
            Series that contains logFC for each gene

        Return
        ------
        DataFrame that contains logFC, all topological metrics
    """
    array = PPI_numpy_array(gene_set, species, experimental_score_threshold)
    matrix = array.get_interactions_as_adjacency_matrix()
    interactions_graph = PPI_graph(matrix, array.get_dict_number_genes())
    df_topo = create_df_gene_topo_scores_logFC(series_genes, interactions_graph.get_dataframe_all_topolog_metrics())
    return df_topo

def concat_df_log_FC_topo_score_normalize(df_topo_up, df_topo_down):
    """
        Combine 2 dataframes that contain logFC, all topological metrics for genes with increased expression and genes
        with decreased expression, into one and normalizes it

        Parameters
        ---------
        df_topo_up : pandas.core.frame.DataFrame
            dataframe that contain logFC, all topological metrics for genes with increased expression
        df_topo_down : pandas.core.frame.DataFrame
            dataframe that contain logFC, all topological metrics for genes with decreased expression

        Return
        ------
        normalized DataFrame that contains logFC, all topological metrics for differentially expressed genes
    """
    df_topo_concated = pd.concat([df_topo_up, df_topo_down], keys = ['up','down'])
    df_topo_concated['logFC'] = abs(df_topo_concated['logFC'])
    min_max_scaler = MinMaxScaler()
    df_topo_concated_positive_logFC_MinMaxScaler = min_max_scaler.fit_transform(df_topo_concated)
    df_topo_concated_positive_logFC_MinMaxScaler = pd.DataFrame(df_topo_concated_positive_logFC_MinMaxScaler)
    df_topo_concated_positive_logFC_MinMaxScaler.columns = df_topo_concated.columns
    df_topo_concated_positive_logFC_MinMaxScaler.index = df_topo_concated.index
    return df_topo_concated_positive_logFC_MinMaxScaler


def calculate_inf_score(df_logFC_topo_scores, func_inf_score, dict_multiplication_factor, dict_additive_factor):
    """
        Сalculate inf_score values based on logFC and topological metrics for genes

        Parameters
        ---------
        df_logFC_topo_scores : pandas.core.frame.DataFrame
            normalized DataFrame that contains logFC, all topological metrics for differentially expressed genes
        func_inf_score : function
            function to calculate inf_score values
        dict_multiplication_factor : dict
            dictionary with the values of the coefficients by which the metrics are multiplied
            in the expression inf_score
        dict_additive_factor : dict
            a dictionary with coefficient values that are added to metrics in the expression inf_score

        Return
        ------
        DataFrame that contains logFC, all topological metrics and inf_scores for differentially expressed genes
    """
    df_logFC_topo_scores['inf_score'] = np.ones(df_logFC_topo_scores.shape[0])
    for key in ['up', 'down']:
        for gene in list(df_logFC_topo_scores.loc[key].index):
            list_multiplication_factor = []
            list_additive_factor = []
            list_topo_score = []
            for metric in list(df_logFC_topo_scores.columns)[:-1]:
                if (not np.isnan(df_logFC_topo_scores.loc[key].loc[gene, metric])):
                    list_multiplication_factor.append(dict_multiplication_factor[metric])
                    list_additive_factor.append(dict_additive_factor[metric])
                    list_topo_score.append(df_logFC_topo_scores.loc[key].loc[gene, metric])
                else:
                    pass
            d_arg = {}
            d_arg['multiplication factor'] = list_multiplication_factor
            d_arg['additive_factor'] = list_additive_factor
            d_arg['topo_score'] = list_topo_score
            df_logFC_topo_scores.loc[key].loc[gene,'inf_score'] = func_inf_score(**d_arg)
    return df_logFC_topo_scores


def func_inf_score_v1(**kwargs):
    """
         Сalculate inf_score values based on logFC and topological metrics for gene

         Parameters
         ---------
         dict with list of multiplication factor, additive factor, topological metrics values for gene

         Return
         ------
         inf_score
     """
    inf_score = 1
    for (multiplication_factor, additive_factor, topo_score) in zip(kwargs['multiplication factor'], kwargs['additive_factor'], kwargs['topo_score']):
        inf_score = inf_score * (multiplication_factor * topo_score + additive_factor)

    return inf_score


if __name__ == '__main__':

    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    #read the genes with logFC selected by logFC, pvalue
    series_up_genes, series_down_genes = make_signature_from_DE(namespace.path_to_file_with_DE, namespace.logFC_threshold,
                                                                namespace.pvalue_threshold)
    #read the genes that are in STRING
    up = namespace.path_to_file_with_up_genes.read()
    up = up.split('\n')
    down = namespace.path_to_file_with_down_genes.read()
    down = down.split('\n')

    path_to_dir_save_results_conv = namespace.path_to_dir_save_results + '/' + namespace.source_type_cell +'_' + namespace.target_type_cell

    #first let's look at genes with increased expression

    start_time = time.time()

    up_interactions_numpy_array = PPI_numpy_array(up, namespace.species, namespace.experimental_score_threshold)

    matrix = up_interactions_numpy_array.get_interactions_as_adjacency_matrix()
    up_interactions_graph = PPI_graph(up_interactions_numpy_array.get_interactions_as_adjacency_matrix(),
                                      up_interactions_numpy_array.get_dict_number_genes())

    #up_interactions_graph.draw_PPI_graph()
    up_interactions_graph.save_image_graph( path_to_dir_save_results_conv + '/PPI_grap_tool_up_' + namespace.source_type_cell+'_' + namespace.target_type_cell + '.png')
    up_interactions_graph.save_graph(path_to_dir_save_results_conv + '/PPI_graph_tool_up_' + namespace.source_type_cell +'_' + namespace.target_type_cell + '.gt')
    print('--- %s seconds ---' % (time.time() - start_time))

    q = 0
    up_g = up_interactions_graph.get_graph()
    for i in up_g.edges():
        q = q + 1
    print('number edges in up:', q)

    n=0
    for i in up_g.vertices():
        n = n + 1
    print('number vertices in up', n)

    start_time = time.time()
    df_topo_up = create_df_gene_topo_scores_logFC(series_up_genes, up_interactions_graph.get_dataframe_all_topolog_metrics())
    df_topo_up.to_csv(path_to_dir_save_results_conv + '/df_topo_up_' + namespace.source_type_cell +'_' + namespace.target_type_cell + '.csv', columns = df_topo_up.columns, index = True)
    print('получила датафрейм со скорами за :', '--- %s seconds ---' % (time.time() - start_time))


    #let's look at genes with decreased expression

    start_time = time.time()
    down_interactions_numpy_array = PPI_numpy_array(down, namespace.species, namespace.experimental_score_threshold)
    matrix_down = down_interactions_numpy_array.get_interactions_as_adjacency_matrix()
    down_interactions_graph = PPI_graph(down_interactions_numpy_array.get_interactions_as_adjacency_matrix(),
                                      down_interactions_numpy_array.get_dict_number_genes())
    # up_interactions_graph.draw_PPI_graph()
    down_interactions_graph.save_image_graph(path_to_dir_save_results_conv + '/PPI_grap_tool_down_' + namespace.source_type_cell +'_' + namespace.target_type_cell + '.png')
    down_interactions_graph.save_graph(path_to_dir_save_results_conv + '/PPI_graph_tool_down_' +  namespace.source_type_cell +'_' + namespace.target_type_cell + '.gt')
    print('--- %s seconds ---' % (time.time() - start_time))

    q = 0
    down_g = down_interactions_graph.get_graph()
    for i in down_g.edges():
        q = q + 1
    print('number edges in down:', q)

    n = 0
    for i in down_g.vertices():
        n = n + 1
    print('number vertices in down', n)

    start_time = time.time()
    df_topo_down = create_df_gene_topo_scores_logFC(series_down_genes, down_interactions_graph.get_dataframe_all_topolog_metrics())
    df_topo_down.to_csv(path_to_dir_save_results_conv + '/df_topo_down_' + namespace.source_type_cell +'_' + namespace.target_type_cell +
                        '.csv', columns=df_topo_down.columns, index=True)
    print('получила датафрейм со скорами :', '--- %s seconds ---' % (time.time() - start_time))

    df_topo = concat_df_log_FC_topo_score_normalize(df_topo_up, df_topo_down)
    df_topo.to_csv(
        path_to_dir_save_results_conv + '/df_topo_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.csv',
        columns=df_topo.columns, index=True)
    dict_multiplication_factor = {'betweenness': 9.460967341000304, 'closeness': 6.238127619523721, 'eigentrust':
        8.909487859706598, 'eigenvector': 8.602610008529997, 'katz': 5.917121344183359, 'logFC': 8.187432320368355,
                                  'pagerank': 3.571469665607328}
    dict_additive_factor = {'logFC': 1, 'betweenness': 1, 'pagerank': 1, 'closeness': 1, 'katz': 1, 'eigenvector': 1, 'eigentrust': 1}
    df_inf = calculate_inf_score(df_topo, func_inf_score_v1, dict_multiplication_factor,
                                       dict_additive_factor)
    df_inf.to_csv(path_to_dir_save_results_conv + '/df_inf_' + namespace.source_type_cell + '_' + namespace.target_type_cell + '.csv',
        columns=df_inf.columns, index=True)




