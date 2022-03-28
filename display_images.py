import cv2
import os


def start(home_dir):
    os.chdir(home_dir)
    print(os.listdir())
    num_datasets = 3
    times = [240, 240, 960]
    replication_type = ['Default', 'None', 'None']
    data = input(f"Name of Dataset: ")
    files = [data, data, data]
    image = gen_image(files, replication_type, times, home_dir)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    os.chdir(home_dir)
    cv2.imwrite(f'{data}_comparison.png', image)

def gen_image(files, replication_type, times, home_dir):
    img = []
    for j in range(len(files)):
        os.chdir(os.getcwd() + f'/{replication_type[j]}_replication/{files[j]}_{replication_type[j]}_replication/')
        temp = cv2.imread(f'{files[j]}_{replication_type[j]}_replication_values_{times[j]}_image.png')
        for k in range(3):
            temp = cv2.vconcat([temp, cv2.imread(f'{files[j]}_{replication_type[j]}_replication_values_{times[j]}_image_{k}.png')])
        os.chdir(home_dir)
        if j > 0:
            img = cv2.hconcat([img, temp])
        else:
            img = temp
    return img



if __name__ == "__main__":
    directory = '/Users/andrew/PycharmProjects/pace_outputs/'
    start(directory)