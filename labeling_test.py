import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon
from scipy import ndimage, misc
from scipy.ndimage.interpolation import zoom
import shapely.geometry as shg
import json

train_path = 'gtFine_trainvaltest/gtFine/train/'
img8bit_path = 'ME_code/train/'
filename = "ME_code/train_demo/attention_results/crop_results.h5"


def get_colored_image_fullpath(path: str):
    colored_filename = path.replace ("_leftImg8bit.png", "_gtFine_color.png")
    #full_path = train_path + '/' + path.split ("_")[0] + '/' + colored_filename
    return 'gtFine_trainvaltest\\gtFine\\'+ colored_filename.replace('ME_code/','')


def get_image_full_path(path: str):
    return img8bit_path + '/' + path.split ("_")[0] + '/' + path


def openImage(path: str):
    img = Image.open (path)
    return img


def get_result_of_labeling(image, path):
    width, height = image.size
    file_names = 1
    count = 0
    for x in range (0, width):
        for y in range (0, height):
            r, g, b, alpha = image.getpixel ((x, y))
            if r == 250 and g == 170 and b == 30:
                count += 1
    orange_percentage = count / (width * height)
    if orange_percentage > 0.6:
        # true
        #image.save ("results/true/" + path)
        return True,False
    elif orange_percentage > 0.95 or orange_percentage > 0.25:
        #image.save ("results/ignore/" + path)
        # ignore
        return False,True
    #false
    #image.save ("results/false/" + path)
    return False,False


if __name__ == '__main__':
    df = pd.read_hdf (filename)
    is_true_list = []
    is_ignore_list = []
    for row in df.itertuples ():
        print(row)
        path = row[1]
        image = openImage (get_colored_image_fullpath (path))
        try:
            crop = image.crop ((row[3], row[4], row[5], row[6]))
            is_true,is_ignore = get_result_of_labeling (crop, get_colored_image_fullpath (path))
            is_true_list.append(is_true)
            is_ignore_list.append(is_ignore)
        except:
            print('error in image crop '+ path)
    df['is_true'] = is_true_list
    df['is_ignore'] = is_ignore_list
    lst = []
    for i in range (0, 5949):
        lst.append (i)
    df.to_hdf('ME_code/train_demo/attention_results/crop_results.h5', 'data')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    df = pd.read_hdf (filename)
    print(df)
