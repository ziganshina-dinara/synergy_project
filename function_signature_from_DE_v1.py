import pandas as pd
import requests # for make API requests
from requests.exceptions import HTTPError# to handle the responses
import time


#Создадим функцию, которая будет определять сигнатуру по анализу DE

def make_signature_from_DE(file, logFC=1, pvalue=0.01):
    """
    find genes with increased and decreased expression

    Parametrs
    ---------
    file: the path to the file with differential expression results (DESeq2, EdgeR)
    logFC: a tuple with a lower and upper threshold for log2 fold change
    pvalue: threshold for pvalue

    Return
    ------
    a tuple with 2 lists: 1) list of genes with increased expression - genes with a logarithm of the fold change greater
                          than set upper threshold for logFC, pvalue less than threshold for pvalue
                          2) list of genes with reduced  expression - genes with a logarithm of the fold change less than
                            set lower threshold for logFC, pvalue less than threshold for pvalue
    Each list of genes is sorted decrease of the modulo the logarithm of the fold change and sorted by increasing pvalue.
    """
    Dif_exp = pd.read_table(file, sep='\t')

    if ('logFC' in Dif_exp.columns.tolist()) & ('PValue' in Dif_exp.columns.tolist()):  # названия столбцов 'logFC', PValue' характерны для edgeR
        Dif_exp_up = Dif_exp[(Dif_exp['logFC'] > logFC) & (Dif_exp['PValue'] < pvalue)]
        Dif_exp_up = Dif_exp_up.sort_values(by=['logFC', 'PValue'], ascending=[False, True])
        Dif_exp_down = Dif_exp[(Dif_exp['logFC'] < - logFC) & (Dif_exp['PValue'] < pvalue)]
        Dif_exp_down = Dif_exp_down.sort_values(by=['logFC', 'PValue'], ascending=[True, True])
        Dif_exp_up_genes = Dif_exp_up['logFC']
        Dif_exp_down_genes = Dif_exp_down['logFC']
    elif ('log2FoldChange' in Dif_exp.columns.tolist()) & ('pvalue' in Dif_exp.columns.tolist()):  # названия столбцов 'log2FoldChange', 'pvalue' характерны для DESeq2
        Dif_exp_up = Dif_exp[(Dif_exp['log2FoldChange'] > logFC) & (Dif_exp['pvalue'] < pvalue)]
        Dif_exp_up = Dif_exp_up.sort_values(by=['log2FoldChange', 'pvalue'], ascending=[False, True])
        Dif_exp_down = Dif_exp[(Dif_exp['log2FoldChange'] < - logFC) & (Dif_exp['pvalue'] < pvalue)]
        Dif_exp_down = Dif_exp_down.sort_values(by=['log2FoldChange', 'pvalue'], ascending=[True, True])
        Dif_exp_up_genes = Dif_exp_up['log2FoldChange']
        Dif_exp_down_genes = Dif_exp_down['log2FoldChange']


    return (Dif_exp_up_genes, Dif_exp_down_genes)

def get_list_protein_in_STRING(gene_set, number=2000, species=9606):
    """
    find a list of genes whose proteins are in the database STRING

    Parametrs
    ---------
    gene_set: set of genes
    number: the number of genes you want to get. These genes are selected from the beginning of the submitted list of genes
    with the condition that their protein must be in the database STRING
    species: NCBI taxon identifiers (e.g. Human is 9606)

    Return
    ------
    a tuple with 2 lists: 1) list of genes selected from the beginning of the submitted list of genes and
                            their proteins are in the database STRING
                          2) list of genes selected from the beginning of the submitted list of genes and
                            their proteins are not in the database STRING
    """

    string_api_url = "https://string-db.org/api"
    output_format = "tsv-no-header"
    method = "network"

    request_url = "/".join([string_api_url, output_format, method])

    list_proteins_in_STRING = []
    list_proteins_not_in_STRING = []
    i = 0
    number_proteins_in_STRING = 0
    while (number_proteins_in_STRING < number):
        gene = gene_set[i]
        i += 1
        params = {

            "identifiers": gene,  # your protein
            "species": species,  # species NCBI identifier
            "caller_identity": "www.awesome_app.org"  # your app name

        }

        response = requests.post(request_url, data=params)

        try:
            response.raise_for_status()
            list_proteins_in_STRING.append(gene)
            number_proteins_in_STRING = len(list_proteins_in_STRING)

        except ConnectionError:
            print('Connection Error')

        except HTTPError as http_err:
            list_proteins_not_in_STRING.append(gene)

        except gaierror as g_err:
            print(f'Error occurred: {g_err}')

        except MaxRetryError as MaxRetry_err:
            print(f'Error occurred: {MaxRetry_err}')

        except NewConnectionError as NewConnection_err:
            print(f'Error occurred: {NewConnection_err}')

        except Exception as err:
            print(f'Other error occurred: {err}')

    return (list_proteins_in_STRING, list_proteins_not_in_STRING)


def get_signature_for_request_in_STRING(file, logFC=1, pvalue=0.01, number=2000, species=9606):
    """
    Find genes with increased and decreased expression from file with differential expression results.
    First these genes are selected from file with DE results by logarithm of the fold change and pvalue.
    Second these genes sorted by decrease of the modulo the logarithm of the fold change and sorted by increasing pvalue.
    Third the desired number of genes with the highest modulo the logarithm of the fold change and lowest pvalue is selected
    with the condition that their protein must be in the database STRING.

    Parametrs
    ---------
    file: the path to the file with differential expression results (DESeq2, EdgeR)
    logFC: a tuple with a lower and upper threshold for log2 fold change
    pvalue: threshold for pvalue
    number: the number of genes with increased or decreased expression you want to get.
    species: NCBI taxon identifiers (e.g. Human is 9606)

    Return
    ------
    a tuple with 2 lists: 1) list of genes with increased expression - genes with a logarithm of the fold change greater than
                            set upper threshold for logFC, pvalue less than threshold for pvalue.
                            List consist of genes that have the highest the logarithm of the fold change and lowest pvalue.
                            List is sorted by decrease of the modulo the logarithm of the fold change and sorted by
                            increasing pvalue. Proteins of these genes are in the database STRING.
                          2) list of genes with decreased expression - genes with a logarithm of the fold change less than
                            set down threshold for logFC, pvalue less than threshold for pvalue.
                            List consist of genes that have the lowest the logarithm of the fold change and lowest pvalue.
                            List is sorted by decrease of the modulo the logarithm of the fold change and sorted by
                            increasing pvalue. Proteins of these genes are in the database STRING.
    """
    # find genes with increased and decreased expression from DE file by the threshold values
    data_up_genes, data_down_genes = make_signature_from_DE(file, logFC, pvalue)
    data_up_genes = list(data_up_genes.index)
    data_down_genes = list(data_down_genes.index)

    # select genes whose proteins are in the database STRING
    proteins_up_in_STRING = get_list_protein_in_STRING(data_up_genes, number, species)[0]
    proteins_down_in_STRING = get_list_protein_in_STRING(data_down_genes, number, species)[0]
    return (proteins_up_in_STRING, proteins_down_in_STRING)

if __name__ == '__main__':
    """
    Для примера посмотрим на результаты анализа дифференциальной экспрессии для образцов клеток сердца и фибробластов
    c помощью edgeR.
    """
    start_time = time.time()
    data_up_genes, data_down_genes = get_signature_for_request_in_STRING(file = './DATA/DE/DE_edgeR_cheart_fibroblast.txt', logFC=1, pvalue=0.01, number=2000, species=9606)
    print("up :", len(data_up_genes))
    print(data_up_genes)
    print("down :", len(data_down_genes))
    print(data_down_genes)
    print('время работы :', '--- %s seconds ---' % (time.time() - start_time))



