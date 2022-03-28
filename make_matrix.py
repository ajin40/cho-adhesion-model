import cv2
import os
import numpy as np

def start(home_dir, time, sim_num, suff):
    os.chdir(home_dir)
    paths = os.listdir(home_dir)
    paths = sorted(paths)
    im_suffix = ['', '_0', '_1', '_2']
    suffix = ['', '_b' , '_r', '_y']
    v_counter = 0
    h_counter = 0
    temp = []
    img = []
    matrix = []
    zoom = 325
    for simulation in paths:
        if simulation.endswith('.png'):
            tags = simulation.split('_')
            if int(tags[3]) == sim_num and int(tags[7]) == time:
                if simulation.endswith(f'image{im_suffix[suff]}.png'):
                    temp = cv2.imread(simulation)
                    print(simulation)
                    temp = temp[1625-zoom:1625+zoom, 1625-zoom:1625+zoom]
                    if h_counter == 0:
                        img = temp
                    else:
                        img = cv2.hconcat([img, temp])
                    h_counter +=1
                    if h_counter > 7:
                        h_counter = 0
                        if v_counter > 0:
                            matrix = cv2.vconcat([matrix, img])
                        else:
                            matrix = img
                        v_counter+=1
        os.chdir(home_dir)
    cv2.imshow('image', matrix)
    cv2.waitKey(0)
    cv2.imwrite(f'../run2_replicate{sim_num}_matrix_96h{suffix[suff]}.png', matrix)

def lil_rename(home_dir):
    comparison = ['0','1','2','3','4','5','6','7','8','9']
    os.chdir(home_dir)
    paths = os.listdir(home_dir)
    for simulation_file in paths:
        if simulation_file.split('_')[0] in comparison:
                #os.chdir(simulation_file + '/')
                #for file in os.listdir():
            os.rename(simulation_file, '0'+simulation_file)
                #os.chdir(home_dir + '/')
                #os.rename(simulation, '0'+simulation)

def add_simulation_tag(home_dir):
    comparison = ['0','1','2','3','4','5','6','7','8','9']
    os.chdir(home_dir)
    paths = os.listdir(home_dir)
    for simulation_file in paths:
        reconstruct_name = simulation_file.split('_')
        new_name = ''.join([str(elem) + '_' for elem in reconstruct_name[0:3]])+'0_' + ''.join([str(elem) + '_' for elem in reconstruct_name[3:]])
        os.rename(simulation_file, new_name[:-1])

def move_images(home_dir, target_dir):
    os.chdir(home_dir)
    paths = os.listdir(home_dir)
    for simulation in paths:
        if os.path.isdir(simulation):
            os.chdir(simulation + '/')
            for file in os.listdir():
                if file.endswith('.png'):
                    os.rename(file, target_dir + file)
            os.chdir(home_dir + '/')

if __name__ == "__main__":
    directory = '/Users/andrew/PycharmProjects/pace_outputs/Backup/experiment1_run3/replicate0/'
    start(directory, 960, 0, 0)
    start(directory, 960, 0, 1)
    start(directory, 960, 0, 2)
    start(directory, 960, 0, 3)