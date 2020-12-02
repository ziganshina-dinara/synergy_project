import pandas as pd
import numpy as np
import requests
import json

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

if __name__ == '__main__':
    """
    Для примера посмотрим на результаты анализа дифференциальной экспрессии для образцов клеток сердца и фибробластов
    c помощью edgeR.
    """
    data = pd.read_table('DATA/DE_heart_fibroblast/DE_with_edgeR_for_heart_fibroblast.txt', sep='\t')
    data.head()
    data_up_genes, data_down_genes = make_signature_from_DE('DATA/DE_heart_fibroblast/DE_with_edgeR_for_heart_fibroblast.txt')

    print("up :", data_up_genes.shape)
    print("down :", data_down_genes.shape)



