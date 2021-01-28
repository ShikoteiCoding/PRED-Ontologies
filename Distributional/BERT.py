# NOT WORKING !


import subprocess


# path to the model
path_to_model = "C:\\Users\\yahuan\\Documents\\Polytech\\PRED\\Ontologies\\Model\\cased_L-12_H-768_A-12"


# start model service
def start_model_service() -> None:
    subprocess.check_output(['bert-serving-start', '-model_dir', "C:\\Users\\yahuan\\Documents\\Polytech\\PRED"
                                                                 "\\Ontologies\\Model\\cased_L-12_H-768_A-12\\",
                             "-num_worker=1"])


start_model_service()
# bert-serving-start -model_dir ./Model/cased_L-12_H-768_A-12/ -num_worker=1

