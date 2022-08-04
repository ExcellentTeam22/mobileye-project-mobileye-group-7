from PIL import Image
import pandas as pd
import os


def crop_images():

    dir = os.getcwd() + "\\" + "train"

    filename = "attention_results.h5"
    df = pd.read_hdf(filename)

    list_for_pandas = []
    for index, row in df.iterrows():
        full_dir_path = dir + "\\" + (row['path'].split("_"))[0]
        image = Image.open(full_dir_path + "\\" + row['path'])
        if row['col'] == 'r':
            # img2 = image.crop((row['x'] - 12, row['y'] - 12, row['x'] + 12, row['y'] + 40))
            list_for_pandas.append((row['path'], row['x'] - 12, row['y'] - 12, row['x'] + 12, row['y'] + 40, row['col']))
        else:
            # img2 = image.crop((row['x'] - 20, row['y'] - 115, row['x'] + 25, row['y'] + 20))
            list_for_pandas.append((row['path'], row['x'] - 20, row['y'] - 115, row['x'] + 25, row['y'] + 20, row['col']))

    df1 = pd.DataFrame(list_for_pandas, columns=['path', 'x0', 'y0', 'x1', 'y1', 'color'])

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified als
    print(df1)





if __name__ == '__main__':
    crop_images()