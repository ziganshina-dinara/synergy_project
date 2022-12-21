import pandas as pd
import time
from multiprocessing import Pool, cpu_count
import json
from intersecting_signatures_by_pert_id import split_list, TC, find_sid_by_sid_id

def create_pars_signatures_same_cell_id(data_CD_signature_metadata):
    list_cell_id_pair_signatures = []
    for cell_id in list(data_CD_signature_metadata['cell_id'].unique()):
        if len(list(data_CD_signature_metadata[data_CD_signature_metadata['cell_id'] == cell_id].index)) > 1:
            list_sig_id = list(data_CD_signature_metadata[data_CD_signature_metadata['cell_id'] == cell_id].index)
            for i in range(len(list_sig_id) - 1):
                for j in range(i + 1, len(list_sig_id)):
                    list_cell_id_pair_signatures.append((cell_id, list_sig_id[i], list_sig_id[j]))
    return list_cell_id_pair_signatures

def intersecting_signatures_by_cell_id(list_cell_id_pair_signatures, data_CD_signature_metadata,
                            data_Drugs_metadata, out_of_file_with_signatures):
    list_cell_id = []
    list_sig_id_1 = []
    list_sig_id_2 = []
    list_pert_id_for_sig_1 = []
    list_pert_id_for_sig_2 = []
    list_pert_name_for_sig_1 = []
    list_pert_name_for_sig_2 = []
    list_canonical_smiles_for_sig_1 = []
    list_canonical_smiles_for_sig_2 = []
    list_Up_intersecting_genes= []
    list_List_of_up_intersecting_genes = []
    list_Tc_up = []
    list_Down_intersecting_genes = []
    list_List_of_down_intersecting_genes = []
    list_Tc_down = []

    for (cell_id, sign_id_1, sign_id_2) in list_cell_id_pair_signatures:
        #print(cell_id, sign_id_1, sign_id_2)
        start = time.time()
        list_cell_id.append(cell_id)
        list_sig_id_1.append(sign_id_1)
        list_sig_id_2.append(sign_id_2)
        pert_id_1 = data_CD_signature_metadata.loc[sign_id_1, 'pert_id']
        pert_id_2 = data_CD_signature_metadata.loc[sign_id_2, 'pert_id']
        list_pert_id_for_sig_1.append(pert_id_1)
        list_pert_id_for_sig_2.append(pert_id_2)
        list_pert_name_for_sig_1.append(data_Drugs_metadata.loc[pert_id_1, 'pert_iname'])
        list_pert_name_for_sig_2.append(data_Drugs_metadata.loc[pert_id_2, 'pert_iname'])
        list_canonical_smiles_for_sig_1.append(data_Drugs_metadata.loc[pert_id_1, 'canonical_smiles'])
        list_canonical_smiles_for_sig_2.append(data_Drugs_metadata.loc[pert_id_2, 'canonical_smiles'])
        (up_1, down_1) = find_sid_by_sid_id(out_of_file_with_signatures, sign_id_1)
        (up_2, down_2) = find_sid_by_sid_id(out_of_file_with_signatures, sign_id_2)
        up = set(up_1) & set(up_2)
        down = set(down_1) & set(down_2)
        list_List_of_up_intersecting_genes.append(';'.join(list(up)))
        list_Up_intersecting_genes.append(len(up))
        list_Tc_up.append(TC(up_1, up_2, up))
        list_Down_intersecting_genes.append(len(down))
        list_List_of_down_intersecting_genes.append(';'.join(list(down)))
        list_Tc_down.append(TC(down_1, down_2, down))
        #print('time of iteration', time.time()-start)
    list_list = [list_cell_id, list_sig_id_1, list_sig_id_2, list_pert_id_for_sig_1, list_pert_id_for_sig_2, list_pert_name_for_sig_1,
                         list_pert_name_for_sig_2, list_canonical_smiles_for_sig_1, list_canonical_smiles_for_sig_2, list_Up_intersecting_genes,
                         list_List_of_up_intersecting_genes, list_Tc_up, list_Down_intersecting_genes, list_List_of_down_intersecting_genes, list_Tc_down]
    return list_list


def creating_df_intersecting_signatures_by_feature(function_intersecting_signatures, data_CD_signature_metadata,
data_Drugs_metadata, dict_signatures, list_name, path_to_file_with_df):
    start = time.time()
    list_pars_signatures_same_cell_id = create_pars_signatures_same_cell_id(data_CD_signature_metadata)
    list_pars_signatures_same_cell_id_splited = split_list(list_pars_signatures_same_cell_id, 1000000)
    print(len(list_pars_signatures_same_cell_id_splited))
    print([len(x) for x in list_pars_signatures_same_cell_id_splited][0:10])
    pool = Pool(processes=10)
    results = pool.starmap(function_intersecting_signatures, [(list_cell_id_pair_signatures, data_CD_signature_metadata,
                                                      data_Drugs_metadata, dict_signatures) for list_cell_id_pair_signatures in
                                                              list_pars_signatures_same_cell_id_splited if len(list_cell_id_pair_signatures) != 0])

    dict_intersecting_signatures = {}
    print("Приступили к результатам")
    print(len(results))
    print([len(results[i]) for i in range(len(results))])
    for name in list_name:
        dict_intersecting_signatures[name] = []
    for list_of_intersecting_signatures in results:
        for (name, list_name_of_intersecting_signatures) in zip(list_name, list_of_intersecting_signatures):
            dict_intersecting_signatures[name] += list_name_of_intersecting_signatures
    pool.close()
    for name in list_name:
        print(name, len(dict_intersecting_signatures[name]))
    #print(dict_intersecting_signatures)
    df_intersecting_signatures = pd.DataFrame(dict_intersecting_signatures)
    print(time.time() - start)
    df_intersecting_signatures.to_csv(path_to_file_with_df)
    print(df_intersecting_signatures.shape)
    print(df_intersecting_signatures)
    return df_intersecting_signatures


if __name__ == '__main__':
    print("начинаем читать функции и файлы")
    data_Drugs_metadata = pd.read_csv('DATA/Drugs_metadata.csv', index_col = 0)
    data_CD_signature_metadata = pd.read_csv('DATA/CD_signature_metadata.csv', index_col = 0)
    with open("DATA/CD_signatures_binary_42809.gmt", "r") as file:
        out_of_file_with_signatures =file.read()
    print("прочитали файлы")

    with open("intersecting_signatures_task/signatures_dict.json", "r") as read_file: #intersecting_signatures_task/
        dict_signatures = json.load(read_file)

    start = time.time()
    #intersect signatures perturbed on the same cell type
    print('intersect signatures perturbed on the same cell type')


    list_name_for_intersecting_signatures_by_cell_id = ['cell_id', 'sig_id_1', 'sig_id_2', 'pert_id for sig_1', 'pert_id for sig_2', 'pert_name for sig_1',
                    'pert_name for sig_2', 'canonical_smiles for pert of sig_1', 'canonical_smiles for pert of sig_2',
                    'Up intersecting genes', 'List of up intersecting genes', 'Tc up', 'Down intersecting genes', 'List of down intersecting genes', 'Tc down']
    df_intersecting_signatures_by_cell_id = creating_df_intersecting_signatures_by_feature(intersecting_signatures_by_cell_id, data_CD_signature_metadata,
                                                                                           data_Drugs_metadata, dict_signatures, list_name_for_intersecting_signatures_by_cell_id,
                                                                                           'intersecting_signatures_task/intersecting_signatures_by_cell_id_1.csv')

    print('время работы:',time.time() - start)

