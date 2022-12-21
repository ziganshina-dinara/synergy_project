import argparse  # read arguments from the command line
import sys


def createParser ():
    """
    script parameters parser

    Return
    ------
    instance of the class ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-signatures', '--path_to_file_with_signatures',
                        default = 'DATA/CD_signatures_binary_42809.gmt', type = argparse.FileType())
    parser.add_argument('-list_sign_id', '--path_to_file_with_list_sign_id', type = argparse.FileType())
    parser.add_argument('-path_to_sign', '--path_to_file_with_needed_signatures', type=str)

    return parser


def create_list_needed_signatures(out_from_file_with_signatures, list_sign_id):
    """
    create list of needed signatures by the list of sign_id and the content of the file with signatures from database

    Parameters
    ---------
    out_from_file_with_signatures : str
        content of the file with signatures from database
    list_sign_id : list
        list of sign_id for needed signatures

    Return
    ------
    list of needed signatures
    """
    l = out_from_file_with_signatures.strip().split('\n')
    list_sign_id_in_L1000FWD = []
    list_needed_signatures = []
    for sub_l in l:
        list_sign_id_in_L1000FWD.append(sub_l.split('\t')[0])
    for sign_id in list_sign_id:
        if sign_id != str():
            k = list_sign_id_in_L1000FWD.index(sign_id)
            list_needed_signatures.append(l[k])
            list_needed_signatures.append(l[k+1])
    return list_needed_signatures


if __name__ == '__main__':

    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    out_of_file_with_signatures = namespace.path_to_file_with_signatures.read()
    out_of_file_with_list_sign_id = namespace.path_to_file_with_list_sign_id.read()
    list_sign_id = out_of_file_with_list_sign_id.strip().split('\n')
    list_needed_signatures = create_list_needed_signatures(out_of_file_with_signatures, list_sign_id)
    with open(namespace.path_to_file_with_needed_signatures, "w") as file:
        file.write('\n'.join(list_needed_signatures))
