import os
import pandas as pd

def csv_to_xyz(datafile):
    # assume datafile name is output:
    df = pd.read_csv(datafile, sep=',')
    xyz_file_name = "{}.xyz".format(datafile[:-4])
    x = df['locations[0]']
    y = df['locations[1]']
    z = df['locations[2]']
    cell_type = df['cell_type']
    cell_type_name = ['CHO', 'HEK']
    xyz_file = open(xyz_file_name, 'w')
    xyz_file.write("{}\n".format(len(x)))
    xyz_file.write("\n")
    for i in range(len(x)):
        xyz = "{}\t {}\t {}\t {}\n".format(cell_type_name[int(cell_type[i])], x[i], y[i], z[i])
        xyz_file.write(xyz)
    xyz_file.close()


def directory_to_xyz(directory, ts):
    os.chdir(directory)
    maxts = len(os.listdir(directory))
    ts_files = []
    for i in range(int(maxts/ts)+1):
        for file in os.listdir(directory):
            if file.endswith(f"_{i*ts}.csv"):
                csv_to_xyz(file)
