import os
import sys
import model
import numpy as np


def parameter_sweep(yaml_file, path="/storage/home/hcoda1/0/ajin40/p-mkemp6-0/ajin/cho_adhesion_model"):
    print(yaml_file + " ...")
    os.chdir(path + '/yaml_parameters')
    model.TestSimulation.start_sweep(path + '/outputs', yaml_file, f"{yaml_file[:-5]}", 0)
    print("Finished")


if __name__ == "__main__":
    directory = sys.argv[1]
    process_tag = int(sys.argv[2])
    os.chdir(directory)
    yaml_array = os.listdir(directory + "/yaml_parameters")
    for key in yaml_array:
        tag = int(key.split(sep='_')[0])
        if tag % 8 == process_tag:
            parameter_sweep(key)