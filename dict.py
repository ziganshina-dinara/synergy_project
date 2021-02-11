import json
with open("DATA/CMap/CD_signatures_binary_42809.gmt", "r") as file:
    out_of_file_with_signatures = file.read()
    
dict_signatures = {}
for i in range(0, len(out_of_file_with_signatures.split('\n'))-1, 2):
    start = time.time()
    signature_up_list = out_of_file_with_signatures.split('\n')[i].split('\t')
    signature_down_list = out_of_file_with_signatures.split('\n')[i+1].split('\t')
    dict_signatures[signature_up_list[0]] = (signature_up_list[2:], signature_down_list[2:])
    print("работа с", i, "-ой сигнатурой для словаря за", time.time() - start)
    
with open("DATA/CMap/signatures_dict.json", "w") as write_file:
    json.dump(dict_signatures, write_file)

