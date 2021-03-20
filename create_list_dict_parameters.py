import json
import argparse, sys
def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lower_additive_factor', '--lower_bound_additive_factor_values', default = 1, type = int)
    parser.add_argument('-upper_additive_factor', '--upper_bound_additive_factor_values', default = 1, type=int)
    parser.add_argument('-lower_multiplication_factor', '--lower_bound_multiplication_factor_values', default = 1, type=int)
    parser.add_argument('-upper_multiplication_factor', '--upper_bound_multiplication_factor_values', default = 1, type=int)

    parser.add_argument('-dir_results', '--path_to_dir_save_results', default='DATA', type=str)
    parser.add_argument('-source', '--source_type_cell', type=str)
    parser.add_argument('-target', '--target_type_cell', type=str)
    return parser

def create_list_additive_multiplication_dicts(lower_additive, upper_additive, lower_multiplication, upper_multiplication, list_metric,
                                              source_type_cell, target_type_cell, path_to_dir_save_results ):
    list_dict_additive_factor = []
    list_dict_multiplication_factor = []
    for i in range(lower_additive, upper_additive, 1):
        for j in range(lower_multiplication, upper_multiplication, 1):
            dict_additive_factor = {}
            dict_multiplication_factor = {}
            for metric in list_metric:
                dict_additive_factor[metric] = i
                dict_multiplication_factor[metric] = j
            list_dict_additive_factor.append(dict_additive_factor)
            list_dict_multiplication_factor.append(dict_multiplication_factor)

    with open(path_to_dir_save_results + '/' + 'additive_factor_dicts' + \
                         '_' + source_type_cell + '_' + target_type_cell + '.json', 'w') as file:
        json.dump(list_dict_additive_factor, file)
    with open(path_to_dir_save_results + '/' + 'multiplication_factor_dicts' + \
                         '_' + source_type_cell + '_' + target_type_cell + '.json', 'w') as file:
        json.dump(list_dict_multiplication_factor, file)
    return (list_dict_additive_factor, list_dict_multiplication_factor)




list_metric = ['logFC', 'betweenness', 'pagerank', 'closeness', 'katz', 'hits_authority', 'hits_hub', 'eigenvector',
                        'eigentrust']

if __name__ == '__main__':
    # read arguments from the command line
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    (list_dict_additive_factor, list_dict_multiplication_factor) = create_list_additive_multiplication_dicts(namespace.lower_bound_additive_factor_values,
    namespace.upper_bound_additive_factor_values, namespace.lower_bound_multiplication_factor_values, namespace.upper_bound_multiplication_factor_values,
                                                            list_metric,  namespace.source_type_cell, namespace.target_type_cell, namespace.path_to_dir_save_results)
