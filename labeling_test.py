
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon
from scipy import ndimage,misc
from scipy.ndimage.interpolation import zoom
import shapely.geometry as shg
#from matplotlib.patches import Rectangle
#from pylayers.util.geomutil import *
#from pylayers.util.plotutil import *
import json
from rectangle import *
from intersection import intersection


train_path = 'gtFine_trainvaltest/gtFine/train/'

def getTrafficLight(json_data):
    traffic_lights = []
    for i in json_data['objects']:
        if i['label'] == 'traffic light':
            for x in i['polygon']:
                xmax, ymax = np.array(i['polygon']).max (axis=0)
                xmin, ymin = np.array(i['polygon']).min (axis=0)
                traffic_lights.append({'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax})
                rect = Rectangle(xmin, ymin,xmax,ymax)
    return rect


def check_intersection_area(traffic_lights):
    for tfl in traffic_lights:
        print(tfl)


def get_json_fullpath(path:str):
    json_filename = path.replace ("_leftImg8bit.png", "_gtFine_polygons.json")
    full_path = train_path + '/' + path.split("_")[0] + '/' + json_filename
    return  full_path


def get_colored_image_fullpath(path:str):
    json_filename = path.replace ("_leftImg8bit.png", "_gtFine_color.png")
    full_path = train_path + '/' + path.split("_")[0] + '/' + json_filename
    return full_path

def openImage(path:str):
    img = Image.open (path)
    return img

filename = "attention__crop_results.h5"
df = pd.read_hdf(filename)

#image = np.array(Image.open('leftImg8bit_trainvaltest/leftImg8bit/train/aachen/aachen_000004_000019_leftImg8bit.png'))
for row in df.itertuples():
    path = row[1]
    image = openImage(get_colored_image_fullpath(path))
    crop = image.crop((row[4], row[5], row[6], row[7]))
    width, height = crop.size

    r_total = 0
    g_total = 0
    b_total = 0

    count = 0
    for x in range (0, width):
        for y in range (0, height):
            r, g, b, alpha = crop.getpixel ((x, y))
            if r== 250 and g == 170 and b == 30:
                count += 1
    if count/ (width*height) > 0.6:
        print(count, width*height)
        plt.imshow(crop)
        plt.show()

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#    print(df)
    #         json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')