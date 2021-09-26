import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class Image:
    def __init__(self, path=None, color_model='RGB'):
        self.color_model = color_model
        self.img = None
        if path is not None:
            self.read_image(path)


    def read_image(self, path):
        if self.color_model == 'RGB':
            convert = cv.COLOR_BGR2HSV
        elif self.color_model == 'HSV':
            convert = cv.COLOR_RGB2HSV
        self.img = cv.cvtColor(cv.imread(path), convert)

    def show_image(self, title=None, cmap=None):
        if cmap is not None:
            plt.imshow(np.uint8(self.img), cmap=cmap)
        else:
            plt.imshow(np.uint8(self.img))
        if title is not None:
            plt.title(title)

        plt.axis('off')
        plt.show()

