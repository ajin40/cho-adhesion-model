import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import pandas as pd
from pythonabm.backend import check_direct


def zstack(name, directory, loc_x, loc_y, loc_z, colors, min_z, num_slices, z, size=[325, 325, 325], background=(0, 0, 0), origin_bottom=True, image_quality=3250, cell_rad=0.5):
    x_size = image_quality
    scale = x_size / size[0]
    y_size = math.ceil(scale * size[1])
    temp = np.zeros((y_size, x_size, 3), dtype=np.uint8)
    # fig, ax = plt.subplots(5, 3, sharex=True, sharey=True, squeeze=True)
    for i in range(num_slices):
        # get the size of the array used for imaging in addition to the scaling factor

        # create the agent space background image and apply background color
        image = np.zeros((y_size, x_size, 3), dtype=np.uint8)
        image[:, :] = background
        indices = []
        # flatten 3d image in slice
        for index in range(len(loc_z)):
            if min_z+(z*i)< loc_z[index] < min_z+(z*(i+1)):
                indices.append(index)
        zslice_x = np.array(loc_x[indices])
        zslice_y = np.array(loc_y[indices])
        colors_slice = colors[:, indices]

        for index in range(len(zslice_x)):
            # get xy coordinates, the axis lengths, and color of agent
            x, y = int(scale * zslice_x[index]), int(scale * zslice_y[index])
            major, minor = int(scale * cell_rad), int(scale * cell_rad)
            color = (int(colors_slice[2][index]), int(colors_slice[1][index]), int(colors_slice[0][index]))

            # draw the agent and a black outline to distinguish overlapping agents
            image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, color, -1)
            image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, (0, 0, 0), 1)

        # if the origin should be bottom-left flip it, otherwise it will be top-left
        if origin_bottom:
            image = cv2.flip(image, 0)
        image_compression = 4  # image compression of png (0: no compression, ..., 9: max compression)
        alpha = 1/(i+1)
        temp = cv2.addWeighted(temp, 1-alpha, image, alpha, 0)
        cv2.imshow('image', temp)
        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0)
        file_name = f"{name}_zstack_{i}.png"
        if not cv2.imwrite(directory + file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, image_compression]):
            raise Exception("Could not write image")
    file_name = f"{name}_image_zstack.png"
    if not cv2.imwrite(directory + file_name, temp, [cv2.IMWRITE_PNG_COMPRESSION, image_compression]):
        raise Exception("Could not write image")

def find_cluster(directory, csv,z=15):
    df = pd.read_csv(directory + csv, sep=',')

    locations_x = df['locations[0]']
    locations_y = df['locations[1]']
    locations_z = df['locations[2]']
    min_z = np.floor(np.amin(locations_z))
    max_z = np.ceil(np.amax(locations_z))
    num_slices = int(np.rint((max_z - min_z)/z))
    colors_0 = df['colors[0]']
    colors_1 = df['colors[1]']
    colors_2 = df['colors[2]']
    colors = np.vstack((colors_0, colors_1, colors_2))
    return locations_x, locations_y, locations_z, colors, min_z, num_slices, z
if __name__ == "__main__":
    directory = '/Users/andrew/PycharmProjects/CHO_adhesion_model/outputs/HIGHDOX_HIGHABA_v30_n25_urr30_uyy40'
    name = '/HIGHDOX_HIGHABA_v30_n25_urr30_uyy40_values/HIGHDOX_HIGHABA_v30_n25_urr30_uyy40_values_240.csv'
    [locations_x, locations_y, locations_z, colors, min_z, num_slices, z] = find_cluster(directory, name)
    zstack('HIGHDOX_HIGHABA_v30_n25_urr30_uyy40', directory, locations_x, locations_y, locations_z, colors, min_z, num_slices, z)