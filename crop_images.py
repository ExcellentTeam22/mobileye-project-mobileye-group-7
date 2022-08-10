from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


def crop_images():

    dir = "ME_code/train"

    filename = "ME_code/train_demo/attention_results/attention_results.h5"
    df = pd.read_hdf(filename)
    counter = 0
    list_for_pandas = []

    a = df.iterrows()

    path = next(a)

    row_path = path[1].tolist()[0]
    full_dir_path = dir + "\\" + row_path.split("_")[0]
    full_image_path = full_dir_path + "\\" + row_path
    image = Image.open(full_image_path)
    counter = 0

    for index, row in df.iterrows():
        if row['path'] != row_path:
            counter = 0
            row_path = row['path']
            full_dir_path = dir + "\\" + (row['path'].split("_"))[0]
            full_image_path = full_dir_path + "\\" + row_path
            image = Image.open(full_image_path)

        if row['col'] == 'r':
            img2 = image.crop((row['x'] - 35*(1 - row['zoom']), row['y'] - 20*(1 - row['zoom']), row['x'] + 35*(1 - row['zoom']), row['y'] + 110*(1 - row['zoom'])))
            img2 = img2.resize((55, 105))
            img2.save("ME_code/train_demo/attention_results/crop" + str(counter) + row['path'])
            full_image_path = full_dir_path + "\\" + row['path']
            list_for_pandas.append((full_image_path,str(counter) + row['path'], row['x'] - 35*(1 - row['zoom']), row['y'] - 20*(1 - row['zoom']), row['x'] + 35*(1 - row['zoom']), row['y'] + 110*(1 - row['zoom']), row['col']))
            counter += 1

        elif row['col'] == 'g':
            img2 = image.crop((row['x'] - 35*(1 - row['zoom']), row['y'] - 110*(1 - row['zoom']), row['x'] + 35*(1 - row['zoom']), row['y'] + 20*(1 - row['zoom'])))
            img2 = img2.resize((55, 105))
            img2.save("ME_code/train_demo/attention_results/crop" + str(counter) + row['path'])
            full_image_path = full_dir_path + "\\" + row['path']
            list_for_pandas.append((full_image_path,str(counter) + row['path'], row['x'] - 35*(1 - row['zoom']), row['y'] - 110*(1 - row['zoom']), row['x'] + 35*(1 - row['zoom']), row['y'] + 20*(1 - row['zoom']), row['col']))
            counter += 1
    df1 = pd.DataFrame(list_for_pandas, columns=['fullPath','path', 'x0', 'y0', 'x1', 'y1', 'color'])

    df1.to_hdf('crop_results.h5','data')



if __name__ == '__main__':
    crop_images()