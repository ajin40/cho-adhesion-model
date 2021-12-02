import os
import csvtoxyz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def show_data(X, centroids, radius):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    for i in range(len(centroids)):
        alpha = np.linspace(0, 2 * np.pi)
        x = radius[i] * np.cos(alpha) + centroids[i,0]
        y = radius[i] * np.sin(alpha) + centroids[i,1]
        plt.plot(x, y, color='Red')
    plt.show()

def cluster(file_cluster, datafile):
    if os.stat(file_cluster).st_size != 0:
        cells = np.loadtxt(datafile, delimiter='\t', skiprows=2, usecols=(1,2,3))
        cell_type = np.loadtxt(datafile, delimiter='\t', skiprows=2, usecols=(0),dtype=str)

        centroids = np.loadtxt(file_cluster, delimiter=',', usecols=(0,1,2))
        radius = np.loadtxt(file_cluster, delimiter=',', usecols=(3))
        # look at the HEK cells
        hek_cells = cells[cell_type=='HEK']


        show_data(hek_cells, centroids, radius)


def directory_to_images(directory, name, ts=20):
    os.chdir(directory)
    csvtoxyz.directory_to_xyz(directory, ts)
    maxts = len(os.listdir(directory))
    ts_files = []
    for i in range(int(maxts/ts)+1):
        for file in os.listdir(directory):
            if file.endswith(f"_0_clusters.csv"):
                pass
            elif file.endswith(f"_{i*ts}_clusters.csv"):
                file_name_locs = f"{name}_values_{i*ts}.xyz"
                cluster(file, file_name_locs)

if __name__ == "__main__":
    directory_to_images("/Users/andrew/PycharmProjects/tordoff_model/Outputs/10282021_test/10282021_test_values/", "10282021_test")
