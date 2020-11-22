import numpy as np

import requests # for make API requests
from requests.exceptions import HTTPError# to handle the responses

import graph_tool as gt #for make protein networks
from graph_tool import centrality as ct
from graph_tool.draw import graph_draw

import time #to calculate the time


class PPI_numpy_array:
    """
    class Interactions is designed to represent STRING protein-protein interaction network in different forms: adjacency matrix, graph

    Attributes
    ----------
    gene_set: a list of genes with increased expression or list of genes with decreased expression
    species: NCBI taxon identifiers (e.g. Human is 9606)
    experimental_score_threshold: threshold for combined score. It's computed by combining the probabilities from
    the different evidence channels and corrected for the probability of randomly observing an interaction.

    Methods
    -------
    API_request(expression_change)
        requests STRING interaction network for multiple proteins

    get_interactions_as_adjacency_matrix(expression_change)
        builds an adjacency matrix for the interaction network

    """

    def __init__(self, gene_set, species, experimental_score_threshold):
        self.gene_set = gene_set
        self.species = species
        self.experimental_score_threshold = experimental_score_threshold

    def API_request(self):

        string_api_url = "https://string-db.org/api"
        output_format = "tsv-no-header"
        method = "network"

        request_url = "/".join([string_api_url, output_format, method])
        params = {

            "identifiers": "%0d".join(self.gene_set),  # your protein
            "species": self.species,  # species NCBI identifier
            "caller_identity": "www.awesome_app.org"  # your app name

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
                        np_array_as_adjacency_matrix[self.get_dict_genes()[p1], self.get_dict_genes()[p2]] = 1
                        np_array_as_adjacency_matrix[self.get_dict_genes()[p2], self.get_dict_genes()[p1]] = 1
            return np_array_as_adjacency_matrix
        except HTTPError as http_err:
            print(f'some problems with querying the STRING database. More precisely HTTP error occurred: {http_err}')
            print(f'answer database STRING: {response.text}')





"""
write class for build the PPI graph
"""

class PPI_graph:

    def __init__(self, np_array_as_adjacency_matrix, dict_number_genes):
        self.adjacency_matrix = np_array_as_adjacency_matrix
        self.number_proteins = np_array_as_adjacency_matrix.shape[0]
        self.dict_number_genes = dict_number_genes

    def get_graph(self):
        graph = gt.Graph(directed=False)
        vertex_list = graph.add_vertex(self.number_proteins)
        matrix = np.triu(self.adjacency_matrix)
        graph.add_edge_list((np.transpose(matrix.nonzero())))
        vprop_name_proteins = graph.new_vertex_property('string')
        for i in range(self.number_proteins):
            vprop_name_proteins[i] = self.dict_number_genes[i]
        graph.vertex_properties["name_proteins"] = vprop_name_proteins
        return graph

    def draw_PPI_graph(self):
        graph = self.get_graph()
        graph_draw(graph, vertex_text=graph.vertex_properties["name_proteins"])

    def save_graph(self, path_to_file):
        graph = self.get_graph()
        graph.save(path_to_file)

    def save_image_graph(self, path_to_image):
        graph = self.get_graph()
        graph_draw(graph, vertex_text=graph.vertex_properties["name_proteins"], output=path_to_image)

    def get_pagerank(self):
        graph = self.get_graph()
        graph.vertex_properties["pagerank"] = ct.pagerank(graph)
        return graph.vertex_properties["pagerank"]

    def get_betweenness(self):
        graph = self.get_graph()
        betweenness_turple = ct.betweenness(graph)
        graph.vertex_properties["betweenness"] = betweenness_turple[0]
        graph.edge_properties["betweenness"] = betweenness_turple[1]
        return betweenness_turple

    def get_central_point_dominance(self):
        graph = self.get_graph()
        graph.vertex_properties["betweenness"] = self.get_betweenness()[0]
        central_point_dominance = ct.central_point_dominance(graph, graph.vertex_properties["betweenness"])
        vprop_central_point_dominance = graph.new_graph_property('double')
        graph.graph_properties['central_point_dominance'] = vprop_central_point_dominance
        graph.graph_properties['central_point_dominance'] = central_point_dominance
        return central_point_dominance

    def get_closeness(self):
        graph = self.get_graph()
        graph.vertex_properties["closeness"] = ct.closeness(graph)
        return graph.vertex_properties["closeness"]

    def get_eigenvector(self):
        graph = self.get_graph()
        eigenvector_turple = ct.eigenvector(graph)
        vprop_largest_eigenvalue_adjacency_matrix = graph.new_graph_property('double')
        graph.graph_properties[
            'largest eigenvalue of the (weighted) adjacency matrix'] = vprop_largest_eigenvalue_adjacency_matrix
        graph.graph_properties['largest eigenvalue of the (weighted) adjacency matrix'] = eigenvector_turple[0]
        graph.vertex_properties["eigenvector"] = eigenvector_turple[1]
        return eigenvector_turple

    def get_katz(self):
        graph = self.get_graph()
        graph.vertex_properties["katz"] = ct.katz(graph)
        return graph.vertex_properties["katz"]

    def get_hits(self):
        graph = self.get_graph()
        turple_hits = ct.hits(graph)
        graph.vertex_properties["hits, authority centrality values"] = turple_hits[1]
        graph.vertex_properties["hits, hub centrality values"] = turple_hits[2]
        vprop_largest_eigenvalue_ocitation_matrix = graph.new_graph_property('double')
        graph.graph_properties['largest eigenvalue of the cocitation matrix'] = vprop_largest_eigenvalue_ocitation_matrix
        graph.graph_properties['largest eigenvalue of the cocitation matrix'] = turple_hits[0]
        return turple_hits


if __name__ == '__main__':
    from function_signature_from_DE import make_signature_from_DE  # from previous task

    with open("DATA/protein_network/list_proteins_up_in_STRING_2000.txt", "r") as file:
        up = file.read().split("%0d")
    with open("DATA/protein_network/list_proteins_down_in_STRING_2000.txt", "r") as file:
        down = file.read().split("%0d")


    """
    first let's look at genes with increased expression
    """
    start_time = time.time()

    up_interactions_numpy_array = PPI_numpy_array(up, 9606, 0.4)

    matrix = up_interactions_numpy_array.get_interactions_as_adjacency_matrix()
    up_interactions_graph = PPI_graph(up_interactions_numpy_array.get_interactions_as_adjacency_matrix(),
                                      up_interactions_numpy_array.get_dict_number_genes())
    #up_interactions_graph.draw_PPI_graph()
    up_interactions_graph.save_image_graph("DATA/protein_network/up_PPI_grap_tool.png")
    up_interactions_graph.save_graph("DATA/protein_network/up_PPI_graph_tool.gt")
    print("--- %s seconds ---" % (time.time() - start_time))

    q = 0
    up_g = up_interactions_graph.get_graph()
    for i in up_g.edges():
        q = q + 1
    print("number edges in up:", q)

    n=0
    for i in up_g.vertices():
        n = n + 1
    print("number vertices in up", n)

    start_time = time.time()
    up_interactions_graph.get_pagerank()
    up_interactions_graph.get_betweenness()
    #up_interactions_graph.get_central_point_dominance()
    up_interactions_graph.get_closeness()
    up_interactions_graph.get_eigenvector()
    up_interactions_graph.get_katz()
    up_interactions_graph.get_hits()
    print("--- %s seconds ---" % (time.time() - start_time))

    """
    let's look at genes with decreased expression
    """
    start_time = time.time()

    down_interactions_numpy_array = PPI_numpy_array(down, 9606, 0.4)

    matrix_down = down_interactions_numpy_array.get_interactions_as_adjacency_matrix()
    down_interactions_graph = PPI_graph(down_interactions_numpy_array.get_interactions_as_adjacency_matrix(),
                                      down_interactions_numpy_array.get_dict_number_genes())
    # up_interactions_graph.draw_PPI_graph()
    down_interactions_graph.save_image_graph("DATA/protein_network/down_PPI_grap_tool.png")
    down_interactions_graph.save_graph("DATA/protein_network/down_PPI_graph_tool.gt")
    print("--- %s seconds ---" % (time.time() - start_time))

    q = 0
    down_g = down_interactions_graph.get_graph()
    for i in down_g.edges():
        q = q + 1
    print("number edges in down:", q)

    n = 0
    for i in down_g.vertices():
        n = n + 1
    print("number vertices in down", n)

    start_time = time.time()
    down_interactions_graph.get_pagerank()
    down_interactions_graph.get_betweenness()
    # down_interactions_graph.get_central_point_dominance()
    down_interactions_graph.get_closeness()
    down_interactions_graph.get_eigenvector()
    down_interactions_graph.get_katz()
    down_interactions_graph.get_hits()
    print("--- %s seconds ---" % (time.time() - start_time))

