import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from core.image import Image


class ImageSegmentation:
    def __init__(self, image):
        self.lvl_lower = (40, 20, 30)
        self.lvl_upper = (75, 190, 240)

        self.image = image

    def __select__object(self):
        mask_hue = cv.inRange(self.image.img, self.lvl_lower, self.lvl_upper)
        self.image.img = cv.bitwise_and(self.image.img, self.image.img, mask=mask_hue)

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