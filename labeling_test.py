
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

train_path = 'gtFine_trainvaltest/gtFine/train/'

def getTrafficLight(json_data):
    traffic_lights = []
    for i in json_data['objects']:
        if i['label'] == 'traffic light':
            #plt.imshow (np.array(Image.open('leftImg8bit_trainvaltest/leftImg8bit/train/aachen/aachen_000004_000019_leftImg8bit.png')))
            for x in i['polygon']:
                xmax, ymax = np.array(i['polygon']).max (axis=0)
                xmin, ymin = np.array(i['polygon']).min (axis=0)
                traffic_lights.append({'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax})
                #plt.plot (x[0], x[1], 'ro', color='r', markersize=4)
                #plt.gca().add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin))
            #plt.show ()
    return traffic_lights


def check_intersection_area(traffic_lights):
    for tfl in traffic_lights:
        print(tfl)


def get_json_fullpath(path:str):
    json_filename = path.replace ("_leftImg8bit.png", "_gtFine_polygons.json")
    full_path = train_path + '/' + path.split("_")[0] + '/' + json_filename
    return  full_path

filename = "attention__crop_results.h5"

df = pd.read_hdf(filename)

#image = np.array(Image.open('leftImg8bit_trainvaltest/leftImg8bit/train/aachen/aachen_000004_000019_leftImg8bit.png'))
for row in df.itertuples():
    path = row[1]
    rectangle = Rectangle(row[2],row[3],row[4],row[5])
    #print(get_json_fullpath(path))
    file = open(get_json_fullpath(path))
    json_data = json.load(file)
    traffic_light = getTrafficLight(json_data)
    #print(traffic_light)

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#    print(df)
    #         json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')