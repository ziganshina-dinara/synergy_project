import pandas as pd
import time
from multiprocessing import Pool, cpu_count
import json
from math import ceil

def find_sid_by_sid_id(dict_signatures, sign_id):
    try:
        return dict_signatures[sign_id]
    except KeyError:
        return None

def TC(set_1, set_2, intersect): return len(intersect)/(len(set_1) + len(set_2) - len(intersect))



def intersecting_signatures_by_pert_id(list_pert_id_with_sign, data_CD_signature_metadata,
                            data_Drugs_metadata, out_of_file_with_signatures):
    list_pert_id= []
    list_pert_name = []
    list_sig_id_1 = []
    list_sig_id_2 = []
    list_canonical_smiles = []
    list_Up_intersecting_genes= []
    list_List_of_up_intersecting_genes = []
    list_Tc_up = []
    list_Down_intersecting_genes = []
    list_List_of_down_intersecting_genes = []
    list_Tc_down = []
    list_cell_id_sig_1 = []
    list_cell_id_sig_2 = []
    for pert_id in list_pert_id_with_sign:
        if data_CD_signature_metadata[data_CD_signature_metadata['pert_id'] == pert_id].shape[0] > 1:
            list_sig_id = list(data_CD_signature_metadata[data_CD_signature_metadata['pert_id'] == pert_id].index)
            for i in range(len(list_sig_id) - 1):
                for j in range(i + 1, len(list_sig_id)):
                    list_pert_id.append(pert_id)
                    list_pert_name.append(data_Drugs_metadata.loc[pert_id, 'pert_iname'])
                    list_sig_id_1.append(list_sig_id[i])
                    list_sig_id_2.append(list_sig_id[j])
                    list_cell_id_sig_1.append(data_CD_signature_metadata.loc[list_sig_id[i],'cell_id'])
                    list_cell_id_sig_2.append(data_CD_signature_metadata.loc[list_sig_id[j], 'cell_id'])
                    list_canonical_smiles.append(data_Drugs_metadata.loc[pert_id, 'canonical_smiles'])
                    (up_1, down_1) = find_sid_by_sid_id(out_of_file_with_signatures, list_sig_id[i])
                    (up_2, down_2) = find_sid_by_sid_id(out_of_file_with_signatures, list_sig_id[j])
                    up = set(up_1) & set(up_2)
                    down = set(down_1) & set(down_2)
                    list_List_of_up_intersecting_genes.append(';'.join(list(up)))
                    list_Up_intersecting_genes.append(len(up))
                    list_Tc_up.append(TC(up_1, up_2, up))
                    list_Down_intersecting_genes.append(len(down))
                    list_List_of_down_intersecting_genes.append(';'.join(list(down)))
                    list_Tc_down.append(TC(down_1, down_2, down))
    list_list = [list_pert_id, list_pert_name, list_sig_id_1, list_sig_id_2, list_cell_id_sig_1, list_cell_id_sig_2, list_canonical_smiles,
                 list_Up_intersecting_genes, list_List_of_up_intersecting_genes, list_Tc_up,
                 list_Down_intersecting_genes,list_List_of_down_intersecting_genes, list_Tc_down]
    return list_list




def split_list(list_for_split, n_parts):
    part_len = ceil(len(list_for_split) / n_parts)
    return [list_for_split[part_len * k:part_len * (k + 1)] for k in range(n_parts)]


def creating_df_intersecting_signatures_by_feature(function_intersecting_signatures, feature_list,  data_CD_signature_metadata,
data_Drugs_metadata, dict_signatures, list_name, path_to_file_with_df):
    start = time.time()
    list_feature_list_splited = split_list(feature_list, 10)
    print(len(list_feature_list_splited))
    print([len(x) for x in list_feature_list_splited])
    print([feature_list_splited for feature_list_splited in list_feature_list_splited if len(feature_list_splited) != 0])
    pool = Pool(processes=10)
    results = pool.starmap(function_intersecting_signatures, [(feature_list_splited, data_CD_signature_metadata,
                                                      data_Drugs_metadata, dict_signatures) for feature_list_splited in
                                                              list_feature_list_splited if len(feature_list_splited) !=0 ]) #cpu_count()

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

    #intersect signatures perturbed by the same small molecule
    print('intersect signatures perturbed by the same small molecule')
    list_pert_id_with_sign = []
    for pert_id in list(data_Drugs_metadata.index):
        if data_CD_signature_metadata[data_CD_signature_metadata['pert_id'] == pert_id].shape[0] > 1:
            list_pert_id_with_sign.append(pert_id)
    print("составили список соединений длиной:", len(list_pert_id_with_sign))

    list_name_for_intersecting_signatures_by_pert_id  = ['pert_id', 'pert_name', 'sig_id_1', 'sig_id_2', 'cell_id for sig_1', 'cell_id for sig_2',
                     'canonical_smiles', 'Up intersecting genes',
                     'List of up intersecting genes', 'Tc up', 'Down intersecting genes', 'List of down intersecting genes',
                     'Tc down']
    df_intersecting_signatures_by_pert_id = creating_df_intersecting_signatures_by_feature(intersecting_signatures_by_pert_id, list_pert_id_with_sign,  data_CD_signature_metadata,
    data_Drugs_metadata, dict_signatures, list_name_for_intersecting_signatures_by_pert_id , 'intersecting_signatures_task/intersecting_signatures_by_pert_id.csv')
