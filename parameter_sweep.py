import os
import sys
import model
from multiprocessing import Pool


def parameter_sweep(yaml_file, path="/Users/andrew/PycharmProjects/CHO_adhesion_model"):
    print(yaml_file + " ...")
    os.chdir(path + '/yaml_parameters')
    model.TestSimulation.start_sweep(path + '/outputs', yaml_file, f"{yaml_file[:-5]}", 0)
    print("Finished")


if __name__ == "__main__":
    directory = sys.argv[1]
    num_processes = int(sys.argv[2])
    os.chdir(directory)
    yaml_array = os.listdir(directory + "/yaml_parameters")
    yaml_array = [s for s in yaml_array if ".yaml" in s]
    with Pool(processes=num_processes) as pool:
        pool.map(parameter_sweep, yaml_array)