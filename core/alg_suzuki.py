import numpy as np
import cv2 as cv



def counterclockwise_walking(anchor_point, start_point=None):
    i, j = anchor_point
    points = [(i, j+1), (i-1, j+1), (i-1, j),
            (i-1, j-1), (i, j-1), (i+1, j-1),
            (i+1, j), (i+1, j+1)]
    if start_point is not None:
        idx_start_point = points.index(start_point)
        points = points[idx_start_point+1:]+points[:idx_start_point+1]
    for point in points:
        yield point


def clockwise_walking(anchor_point, start_point=None):
    i, j = anchor_point
    points = [(i + 1, j + 1), (i + 1, j), (i + 1, j - 1), (i, j - 1), (i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j + 1),]

    if start_point is not None:
        idx_start_point = points.index(start_point)
        points = points[idx_start_point + 1:] + points[:idx_start_point + 1]
    for point in points:
        yield point

def algorithm_suzuki(mask):
    marker = 2
    for i in range(1,mask.shape[0]-2):
        for j in range(1, mask.shape[1]-2):
            start_point = None
            if mask[i, j - 1] == 0 and mask[i, j] == 1:
                start_point = (i, j - 1)
            elif mask[i, j] >= 1 and mask[i, j+1] == 0:
                start_point = (i, j+1)
            if start_point is not None:
                anchor_point = (i, j)
                init_points = None
                last_searching_point = None
                flag = False
                for searching_point in clockwise_walking(anchor_point, start_point):
                    if mask[searching_point] == 1:
                        init_points = (anchor_point, searching_point)
                        last_searching_point = searching_point
                        flag = True
                        break
                if not flag:
                    continue
                end = False
                # marker += 1
                while not end:

                    for searching_point in counterclockwise_walking(anchor_point, last_searching_point):
                        if searching_point[0] == mask.shape[0]-1 or searching_point[1] == mask.shape[1]-1:
                            continue
                        if mask[searching_point] != 0:
                            if mask[anchor_point[0], anchor_point[1] + 1] != 0 and mask[anchor_point] == 1:
                                mask[anchor_point] = marker
                            elif mask[anchor_point[0], anchor_point[1] + 1] == 0:
                                mask[anchor_point] = -marker


                            if searching_point == init_points[0] and anchor_point == init_points[1]:
                                print('END')
                                end =True
                                print(mask)
                                break
                            else:
                                last_searching_point = anchor_point
                                anchor_point = searching_point
                                print(mask)
                                break
    return mask







if __name__ == '__main__':
    mask = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 1, 1, 1, 1, 0],
                     [0, 0, 1, 1, 1, 1, 1, 0],
                     [0, 0, 1, 1, 1, 1, 1, 0],
                     [0, 0, 1, 1, 1, 1, 1, 0],
                     [0, 0, 1, 1, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0]])

    algorithm_suzuki(mask)