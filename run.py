import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from core.image import Image
from core.alg_suzuki import algorithm_suzuki
from utils import clockwise_walking
from icecream import ic
import pickle


class ImageSegmentation:
    def __init__(self, image):
        self.lvl_lower = (40, 30, 35)
        self.lvl_upper = (75, 215, 230)

        self.image = image
        self.image_rgb = cv.cvtColor(cv.imread('segment.jpg'), cv.COLOR_BGR2RGB)

        self.marker = 1
        self.counters = []
        self.inheritance_tree = {}

    def __select__object(self):
        self.image.img = cv.GaussianBlur(self.image.img, (17,17), 100)
        mask = cv.inRange(self.image.img, self.lvl_lower, self.lvl_upper)
        mask = np.where(mask == 255, 1, mask)
        self.find_counters(mask.astype(np.int32))
        counters = self.counters
        # #
        #
        #
        print()
        # for i in range(mask_hue.shape[0]):
        #     for j in range(mask_hue.shape[1]):
        #         if mask[i, j] != 0 and mask[i,j] != 1 and len(tree[abs(mask[i,j])]) == 1:
        #             self.image.img[i, j] = np.array([255, 255, 255])
        #         else:
        #             self.image.img[i, j] = np.array([0, 0, 0])
        #         print(mask_hue[i,j])
        # with open('save.pickle', 'wb') as f:
        #     pickle.dump(counters, f)
        # with open('save.pickle', 'rb') as f:
        #     counters =  pickle.load(f)
        boxes = {}
        for counter in counters:
            if len(counter) > 0:
                x, y, w, h = cv.boundingRect(counter)
                start = (y, x)
                end = (y+h, x+w)

                square = w*h
                boxes[square] = [start, end, counter]

        t1, _, t2 = sorted(boxes.items())[-3:]


        self.image_rgb = cv.rectangle(self.image_rgb, t1[1][0], t1[1][1], (255, 0, 0), 3)
        self.image_rgb = cv.rectangle(self.image_rgb, t2[1][0], t2[1][1], (255, 0, 0), 3)



        self.image_rgb[t1[1][2][:,0,:][:,0],t1[1][2][:,0,:][:,1]] = [255,0,255]
        self.image_rgb[t2[1][2][:,0,:][:,0],t2[1][2][:,0,:][:,1]] = [255, 0, 255]
        self.image.img = self.image_rgb

    def find_counters(self, mask):
        for i in range(1, mask.shape[0] - 1):
            inheritance = [1]
            for j in range(1, mask.shape[1] - 1):
                start_point = None

                if mask[i, j - 1] == 0 and mask[i, j] == 1:
                    start_point = (i, j - 1)
                elif mask[i, j] >= 1 and mask[i, j + 1] == 0:
                    start_point = (i, j + 1)

                if mask[i, j - 1] == 0 and mask[i, j] > 1:
                    if mask[i, j] == -inheritance[-1]:
                        inheritance.pop()
                    else:
                        inheritance.append(mask[i, j])
                elif mask[i, j] < -1 and mask[i, j + 1] == 0:
                    if mask[i, j] == -inheritance[-1]:
                        inheritance.pop()
                    else:
                        inheritance.append(mask[i, j])

                if start_point is not None:
                    self.counters.append([])
                    anchor_point = (i, j)
                    init_points = None
                    flag = False
                    for searching_point in clockwise_walking(anchor_point, start_point):
                        if mask[searching_point] == 1:
                            init_points = (anchor_point, searching_point)
                            flag = True
                            break
                    if not flag:
                        continue
                    self.marker += 1
                    self.inheritance_tree[self.marker] = inheritance[:]
                    self.__create_counter(mask, init_points)

                    inheritance.append(mask[i, j])
                    self.counters[-1] = np.asarray(self.counters[-1])


    def __create_counter(self, mask, init_points):
        anchor_point, last_searching_point = init_points
        end = False
        while not end:
            for searching_point in clockwise_walking(anchor_point, last_searching_point, counter=True):
                if searching_point[0] == mask.shape[0] - 1 or searching_point[1] == mask.shape[1] - 1:
                    continue
                if mask[searching_point] != 0:
                    if mask[anchor_point[0], anchor_point[1] + 1] != 0 and mask[anchor_point] == 1:
                        mask[anchor_point] = self.marker
                        self.counters[-1].append([list(anchor_point)])
                    elif mask[anchor_point[0], anchor_point[1] + 1] == 0:
                        mask[anchor_point] = -self.marker
                        self.counters[-1].append([list(anchor_point)])
                    if searching_point == init_points[0] and anchor_point == init_points[1]:
                        print('END')
                        end = True
                        break
                    else:
                        last_searching_point = anchor_point
                        anchor_point = searching_point
                        break



    def run(self):
        self.__select__object()

    def show_result(self):

        # title = f'H:{self.lvl_hue}'
        self.image.show_image()


if __name__ == '__main__':
    i = Image('segment.jpg', 'HSV')
    i_segment = ImageSegmentation(i)
    i_segment.run()
    i_segment.show_result()
    # img = cv.inRange(img, np.asarray([cfg.hue_value_lower, 0, 0]), np.asarray([cfg.hue_value_upper, 255, 255]))
    # print()
    # show_image(img)