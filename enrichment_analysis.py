import json
import requests
import time
from function_signature_from_DE_v1 import make_signature_from_DE


class Enrich:
    """
    Сlass Enrich: is used to enrichment analysis

    Attributes
    ----------
    gene_list: list
        list of genes with increased or decreased expression
    threshold : int
        p_value threshold

    Methods
    -------
    analyze_gene_list()
        sends a request
    get_enrichment_results()
        get enrichment results
    """

    def __init__(self, gene_list, threshold):
        self.gene_list = gene_list
        self.threshold = threshold

    def analyze_gene_list(self):

        ENRICHR_URL = 'http://maayanlab.cloud/Enrichr/addList'
        genes_str = '\n'.join(self.gene_list)
        description = 'Example gene list'
        payload = {
            'list': (None, genes_str),
            'description': (None, description)
        }

        response = requests.post(ENRICHR_URL, files=payload)
        if not response.ok:
            raise Exception('Error analyzing gene list')

        dict_results = json.loads(response.text)
        return dict_results['userListId']

    def get_enrichment_results(self):

        ENRICHR_URL = 'http://maayanlab.cloud/Enrichr/enrich'
        query_string = '?userListId=%s&backgroundType=%s'
        user_list_id = self.analyze_gene_list()
        gene_set_library = 'KEGG_2019_Human'
        response = requests.get(
            ENRICHR_URL + query_string % (user_list_id, gene_set_library)
        )
        if not response.ok:
            raise Exception('Error fetching enrichment results')

        dict_results = json.loads(response.text)
        list_results = [i[1] for i in dict_results['KEGG_2019_Human'] if i[2] < self.threshold]
        return list_results


if __name__ == '__main__':

    #for example
    print('С сигнатурами из базы данных')
    with open('DATA/CD_signatures_binary_42809.gmt', "r") as file:
        out_from_file_with_signatures= file.read()
    signature_list = []
    for i in range(0, 20, 2):
        signature_up_list = out_from_file_with_signatures.split('\n')[i].split('\t')
        signature_down_list = out_from_file_with_signatures.split('\n')[i + 1].split('\t')
        up = signature_up_list[2:]
        down = signature_down_list[2:]
        start = time.time()
        analysis = Enrich(up, 0)
        print('количество термов для up: {}'.format(len(analysis.get_enrichment_results())))
        print("за : {}".format(time.time() - start))

        start = time.time()
        analysis = Enrich(down, 0)
        print('количество термов для down: {}'.format(len(analysis.get_enrichment_results())))
        print("за : {}".format(time.time() - start))

    print('С сигнатурой запроса')
    series_up_genes, series_down_genes = make_signature_from_DE('DATA/Fibroblasts_Induced_Cardiomyocytes/DE_edgeR_cheart_fibroblast.txt')
    print(series_up_genes)
    print(series_up_genes.index)
    list_gene_up = list(series_up_genes.index)
    start = time.time()
    analysis = Enrich(list_gene_up, 0)
    print('количество термов для up в сигнатуре запроса: {}'.format(len(analysis.get_enrichment_results())))
    print("за : {}".format(time.time() - start))

    list_gene_down = list(series_down_genes.index)
    start = time.time()
    analysis = Enrich(list_gene_down, 0)
    print('количество термов для down в сигнатуре запроса: {}'.format(len(analysis.get_enrichment_results())))
    print("за : {}".format(time.time() - start))
