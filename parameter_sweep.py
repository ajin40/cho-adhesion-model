import os
from pythonabm import Simulation
import model

def parameter_sweep(directory):
    os.chdir(directory + "/yaml_parameters")
    for file in os.listdir(directory + "/yaml_parameters"):
        if file.endswith(".yaml"):
            print(file + " ...")
            model.TestSimulation.start_sweep(directory + "/outputs", file, f"{file[:-5]}", 0)
            print("Finished")


if __name__ == "__main__":
    parameter_sweep("/Users/andrew/PycharmProjects/CHO_adhesion_model")