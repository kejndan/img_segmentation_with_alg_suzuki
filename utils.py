import numpy as np


def clockwise_walking(anchor_point, start_point=None, counter=False):
    i, j = anchor_point
    points = [(i + 1, j + 1), (i + 1, j), (i + 1, j - 1),
              (i, j - 1), (i - 1, j - 1), (i - 1, j),
              (i - 1, j + 1), (i, j + 1)]
    if counter:
        points = points[::-1]
    if start_point is not None:
        idx_start_point = points.index(start_point)
        points = points[idx_start_point + 1:] + points[:idx_start_point+1]
    points.append(None)
    for point in points:

        yield point

if __name__ == '__main__':
    print(list(clockwise_walking([0,0], (-1,-1),counter=True)))