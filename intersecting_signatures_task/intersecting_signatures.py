import pandas as pd
import time
from multiprocessing import Pool, cpu_count
import json
print("начинаем читать функции и файлы")
data_Drugs_metadata = pd.read_csv('DATA/Drugs_metadata.csv', index_col = 0)
data_CD_signature_metadata = pd.read_csv('DATA/CD_signature_metadata.csv', index_col = 0)
with open("DATA/CD_signatures_binary_42809.gmt", "r") as file:
    out_of_file_with_signatures =file.read()
print("прочитали файлы")

with open("intersecting_signatures_task/signatures_dict.json", "r") as read_file:
    dict_signatures = json.load(read_file)

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

def intersecting_signatures_by_cell_id(list_needed_cell_id, data_CD_signature_metadata,
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

    for cell_id in list_needed_cell_id:
        list_sig_id = list(data_CD_signature_metadata[data_CD_signature_metadata['cell_id'] == cell_id].index)
        for i in range(len(list_sig_id) - 1):
            for j in range(i + 1, len(list_sig_id)):
                list_cell_id.append(cell_id)
                list_sig_id_1.append(list_sig_id[i])
                list_sig_id_2.append(list_sig_id[j])
                pert_id_1 = data_CD_signature_metadata.loc[list_sig_id[i],'pert_id']
                pert_id_2 = data_CD_signature_metadata.loc[list_sig_id[j], 'pert_id']
                list_pert_id_for_sig_1.append(pert_id_1)
                list_pert_id_for_sig_2.append(pert_id_2)
                list_pert_name_for_sig_1.append(data_Drugs_metadata.loc[pert_id_1, 'pert_iname'])
                list_pert_name_for_sig_2.append(data_Drugs_metadata.loc[pert_id_2, 'pert_iname'])
                list_canonical_smiles_for_sig_1.append(data_Drugs_metadata.loc[pert_id_1, 'canonical_smiles'])
                list_canonical_smiles_for_sig_2.append(data_Drugs_metadata.loc[pert_id_2, 'canonical_smiles'])
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
    list_list = [list_cell_id, list_sig_id_1, list_sig_id_2, list_pert_id_for_sig_1, list_pert_id_for_sig_2, list_pert_name_for_sig_1,
                 list_pert_name_for_sig_2, list_canonical_smiles_for_sig_1, list_canonical_smiles_for_sig_2, list_Up_intersecting_genes,
                 list_List_of_up_intersecting_genes, list_Tc_up, list_Down_intersecting_genes, list_List_of_down_intersecting_genes, list_Tc_down]
    return list_list


def split_list(list_for_split, n):
    list_sublist = []
    k = len(list_for_split)//n
    i = 0
    while(i + k < len(list_for_split) - 2):
        list_sublist.append(list_for_split[i: i+ k])
        i += k
    list_sublist[-1] += list_for_split[i: len(list_for_split)]
    return list_sublist

def creating_df_intersecting_signatures_by_feature(function_intersecting_signatures, feature_list,  data_CD_signature_metadata,
data_Drugs_metadata, dict_signatures, list_name, path_to_file_with_df):
    start = time.time()
    pool = Pool()
    results = pool.starmap(function_intersecting_signatures, [(feature_list_splited, data_CD_signature_metadata,
                                                      data_Drugs_metadata, dict_signatures) for feature_list_splited in
                                                     split_list(feature_list, cpu_count())])
    dict_intersecting_signatures = {}
    print("Приступили к распараллеливанию")


    for name in list_name:
        dict_intersecting_signatures[name] = []
    for list_of_intersecting_signatures in results:
        for (name, list_name_of_intersecting_signatures) in zip(list_name, list_of_intersecting_signatures):
            dict_intersecting_signatures[name] += list_name_of_intersecting_signatures
    pool.close()
    df_intersecting_signatures = pd.DataFrame(dict_intersecting_signatures)
    print(time.time() - start)
    df_intersecting_signatures.to_csv(path_to_file_with_df)
    print(df_intersecting_signatures.shape)
    print(df_intersecting_signatures)
    df_intersecting_signatures
    return df_intersecting_signatures

"""
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

"""

#intersect signatures perturbed on the same cell type
print('intersect signatures perturbed on the same cell type')
print(list(data_CD_signature_metadata['cell_id'].unique())[0:3])
list_name_for_intersecting_signatures_by_cell_id  = ['cell_id', 'sig_id_1', 'sig_id_2', 'pert_id for sig_1', 'pert_id for sig_2', 'pert_name for sig_1',
                'pert_name for sig_2', 'cell_id for sig_1', 'cell_id for sig_2', 'canonical_smiles for pert of sig_1', 'canonical_smiles for pert of sig_2',
                'Up intersecting genes', 'List of up intersecting genes', 'Tc up', 'Down intersecting genes', 'List of down intersecting genes', 'Tc down']
df_intersecting_signatures_by_cell_id = creating_df_intersecting_signatures_by_feature(intersecting_signatures_by_cell_id, list(data_CD_signature_metadata['cell_id'].unique())[0:3],  data_CD_signature_metadata,
data_Drugs_metadata, dict_signatures, list_name_for_intersecting_signatures_by_cell_id , 'intersecting_signatures_task/intersecting_signatures_by_cell_id.csv')

