import cv2
import imutils

try:
    import os
    import json
    import glob
    import argparse

    import numpy as np
    from scipy import signal as sg, ndimage
    from scipy.ndimage.filters import maximum_filter
    from scipy.signal import convolve2d
    from scipy.ndimage import white_tophat

    from PIL import Image
    from skimage.io import imread, imshow
    from skimage.color import rgb2gray
    from skimage.transform import rescale
    from skimage import measure


    from imutils import contours

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


def rgb_convolve2d(image, kernel):
    red = convolve2d(image[:,:,0], kernel, 'valid')
    green = convolve2d(image[:,:,1], kernel, 'valid')
    blue = convolve2d(image[:,:,2], kernel, 'valid')
    return np.stack([red, green, blue], axis=2)


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """

    # first step : ok
    maxed_image = ndimage.maximum_filter (c_image, size=4)
    # second step: ok
    gray_image = cv2.cvtColor (maxed_image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones ((11, 11), np.uint8)

    tophat = cv2.morphologyEx (gray_image, cv2.MORPH_TOPHAT, kernel)

    (T, threshInv) = cv2.threshold (tophat, 110, 255,
                                    cv2.THRESH_BINARY_INV)

    return [500, 510, 520], [500, 500, 500], [700, 710], [500, 500]


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    h = plt.subplot (111)
    plt.imshow(image)
    plt.figure (57)
    plt.clf ()
    plt.subplot (111, sharex=h, sharey=h)
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
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    # show_image_and_gt(image, objects, fig_num)
    plt.figure (56)
    plt.clf ()
    h = plt.subplot (111)
    plt.imshow (image)
    plt.figure (57)
    plt.clf ()
    plt.subplot (111, sharex=h, sharey=h)

    # apply laplacian blur
    # Applying the Black-Hat operation

    # first step : ok
    maxed_image = ndimage.maximum_filter (image, size=2)
    # second step: ok
    gray_image = cv2.cvtColor (maxed_image, cv2.COLOR_BGR2GRAY)

    #laplacian = cv2.Laplacian (gray_image, cv2.CV_64F)

    #dilation = cv2.dilate (gray_image, kernel, iterations=1)

    #kernel = np.ones ((9, 9), np.uint8)
    kernel = np.array ([[2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2],
                        [2, 1, 1, 1, 2, -2, 2, 1, 1, 1, 2],
                        [1, 1, 1, 2, -2, -2, -2, 2, 1, 1, 1],
                        [1, 1, 2, -3, -3, -3, -3, -3, 2, 1, 1],
                        [1, 2, -2, -3, -3, -3, -3, -3, -2, 2, 1],
                        [2, -2, -2, -3, -3, -8, -3, -3, -2, -2, 2],
                        [1, 2, -2, -3, -3, -3, -3, -3, -2, 2, 1],
                        [1, 1, 2, -3, -3, -3, -3, -3, 2, 1, 1],
                        [1, 1, 1, 2, -2, -2, -2, 2, 1, 1, 1],
                        [2, 1, 1, 1, 2, -2, 2, 1, 1, 1, 2],
                        [2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2]])
    tophat = cv2.morphologyEx (gray_image, cv2.MORPH_TOPHAT, kernel)


    # convert image to gray scale image
    #maxed_image = ndimage.maximum_filter (tophat, size=3)
    # (T, threshInv) = cv2.threshold (tophat, 100, 255,
    #                                 cv2.THRESH_BINARY)

    """dist_transform = cv2.distanceTransform (threshInv, cv2.DIST_L2, 5)
    ret, markers = cv2.connectedComponents (np.uint8 (dist_transform))
    watershed = cv2.watershed (image, markers)"""

    plt.imshow (tophat,cmap='gray')


    # result = ndimage.maximum_filter (image, size=3)
    # ret, thresh1 = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)
    # ret, thresh2 = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY_INV)
    # ret, thresh3 = cv2.threshold(image, 230, 255, cv2.THRESH_TRUNC)
    # ret, thresh4 = cv2.threshold(image, 230, 255, cv2.THRESH_TOZERO)
    # ret, thresh5 = cv2.threshold(image, 230, 255, cv2.THRESH_TOZERO_INV)
    # titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    # images = [image, thresh1, thresh2, thresh3, thresh4, thresh5]
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    #     plt.title(titles[i])
    thresh = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.erode(thresh, None, iterations=2)

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
        # large, then add it to our mask of "large blobs"
        if numPixels > 200:
            mask = cv2.add(mask, labelMask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]
    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(cX), int(cY)), int(radius),
                   (0, 0, 255), 3)
        cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    # show the output image
    cv2.imshow("Image", image)


    plt.imshow(gray_image)
    (w, h) = gray_image.shape[:2]
    print(h, w)
    # (b, g, r) = image[0, 0]  388 rows, 484 cols
    for i in range(w):
        for j in range(h):
            pixel = gray_image[i, j]
            if pixel >= 50:
                image[i, j] = (0, 0, 255)
    plt.imshow(image)


    # red_x, red_y, green_x, green_y = find_tfl_lights(image)
    # plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    # plt.plot(green_x, green_y, 'ro', color='g', markersize=4)



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
    main()