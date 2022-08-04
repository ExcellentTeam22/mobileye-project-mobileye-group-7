from collections import namedtuple


def intersection(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy # return the area of the intersection between the 2 rects.


Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

# Driver code to test the above function.
ra = Rectangle(3., 3., 5., 5.)
rb = Rectangle(1., 1., 4., 3.5)
print(intersection(ra, rb))