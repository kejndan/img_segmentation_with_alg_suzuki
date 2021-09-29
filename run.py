import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from core.image import Image
from core.alg_suzuki import algorithm_suzuki
from icecream import ic
import pickle


class ImageSegmentation:
    def __init__(self, image):
        self.lvl_lower = (40, 30, 35)
        self.lvl_upper = (75, 215, 230)



        self.image = image
        self.image_rgb = cv.cvtColor(cv.imread('segment.jpg'), cv.COLOR_BGR2RGB)

    def __select__object(self):
        self.image.img = cv.GaussianBlur(self.image.img, (17,17), 100)
        # mask_hue = cv.inRange(self.image.img, self.lvl_lower, self.lvl_upper)
        #
        #
        # ic(np.unique(mask_hue))
        # mask_hue = np.where(mask_hue == 255, 1, mask_hue)
        # #
        # # t = cv.findContours(mask_hue,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
        # ic(np.unique(mask_hue))
        # #
        # mask, tree, counters = algorithm_suzuki(mask_hue.astype(np.int32))
        # #
        #
        #
        # print()
        # for i in range(mask_hue.shape[0]):
        #     for j in range(mask_hue.shape[1]):
        #         if mask[i, j] != 0 and mask[i,j] != 1 and len(tree[abs(mask[i,j])]) == 1:
        #             self.image.img[i, j] = np.array([255, 255, 255])
        #         else:
        #             self.image.img[i, j] = np.array([0, 0, 0])
        #         print(mask_hue[i,j])
        # with open('save.pickle', 'wb') as f:
        #     pickle.dump(counters, f)
        with open('save.pickle', 'rb') as f:
            counters =  pickle.load(f)
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