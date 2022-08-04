import cv2
import imutils
from imutils import contours
from skimage import measure
from skimage.feature import peak_local_max
import pandas as pd

try:
    import os
    import json
    import glob
    import argparse

    import numpy as np
    from scipy import signal as sg, ndimage
    from scipy.ndimage import maximum_filter
    from scipy.signal import convolve2d

    from PIL import Image
    from skimage.io import imread, imshow
    from skimage.color import rgb2gray
    from skimage.transform import rescale

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


def rgb_convolve2d(image, kernel):
    red = convolve2d(image[:, :, 0], kernel, 'valid')
    green = convolve2d(image[:, :, 1], kernel, 'valid')
    blue = convolve2d(image[:, :, 2], kernel, 'valid')
    return np.stack([red, green, blue], axis=2)


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    ### WRITE YOUR CODE HERE ###
    # image_gray = rgb2gray(c_image)

    identity = np.array([[-23.98, -23.98, -23.98, -23.98, -23.98, -23.98, -23.98],
                         [-23.98, -23.98, -23.98, -23.98, -23.98, -23.98, -23.98],
                         [-23.98, -23.98, -23.98, -23.98, -23.98, -23.98, -23.98],
                         [-23.98, -23.98, -23.98, 189, 214, 216, -23.98],
                         [-23.98, -23.98, 185, 192, -23.98, -23.98, -23.98],
                         [-23.98, -23.98, -23.98, 203, -23.98, -23.98, -23.98],
                         [-23.98, -23.98, -23.98, -23.98, -23.98, -23.98, -23.98],
                         [-23.98, -23.98, -23.98, -23.98, -23.98, -23.98, -23.98]])

    # conv_im1 = rgb_convolve2d(c_image, identity)
    # fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # ax[0].imshow(identity, cmap='gray')
    # ax[1].imshow(abs(conv_im1), cmap='gray')

    return [500, 510, 520], [500, 500, 500], [700, 710], [500, 500]


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    kernel = np.array(Image.open(image_path))
    image_to_kernal = np.array(Image.open(
        "C:\\Users\\Shay Tobi\\PycharmProjects\\mobileye-project-mobileye-group-7\\berlin_000455_000019_leftImg8bit.png"))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    # show_image_and_gt(image, objects, fig_num)

    image = np.array(Image.open(image_path))

    kernal = image_to_kernal[255: 268, 1124: 1134]

    red1 = kernal[:, :, 0]

    image_red = image[:, :, 0]

    red2 = cv2.cvtColor(kernal, cv2.COLOR_BGR2GRAY)

    kernal_normlised1 = red1 - (np.sum(red1) / red1.size)

    con = sg.convolve(image_red, kernal_normlised1, mode="same")

    result = (255 * (con - np.min(con)) / np.ptp(con)).astype(int)
    print(result)

    maxed_image = peak_local_max(result, min_distance=100)
    print("maxed image is", maxed_image, len(maxed_image))
    plt.plot(maxed_image[:, 1], maxed_image[:, 0], 'ro', color='r', markersize=4)
    plt.figure()
    plt.imshow(result, cmap='gray')

    plt.figure()
    plt.imshow(result)
    plt.figure()
    plt.imshow(red2)

    plt.figure(56)
    plt.clf()
    h = plt.subplot(111)
    plt.imshow(image)
    plt.figure(57)
    plt.clf()
    plt.subplot(111, sharex=h, sharey=h)

    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)

    result = np.float32(result)
    max_suppression(image, result, image_path)


def max_suppression(image, filtered_image, image_path):
    thresh = cv2.threshold(filtered_image, 50, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=4)

    labels = measure.label(thresh, connectivity=2, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        # if the number of pixels in the component is sufficiently
        # large, then, add it to our mask of "large blobs"
        if numPixels >= 30:
            mask = cv2.add(mask, labelMask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]
    tfl_cords = []
    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        tfl_cords.append((int(cX), int(cY), (255, 0, 255), image_path))
        b, g, r = image[int(cY), int(cX)]
        print(r)
        if r >= 200:
            cv2.circle(image, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)
            cv2.putText(image, "#{}".format(i + 1), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    plt.imshow(image)
    df = pd.DataFrame(tfl_cords, columns=['x', 'y', 'color', 'image path'])
    print(df)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = "one_image"

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    df = pd.read_hdf('attention_results.h5')
    pd.set_option('display.max_rows', None)
    print(df)
    # main()
