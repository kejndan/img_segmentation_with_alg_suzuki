import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def show_image(image, title=None, cmap=None):
    if cmap is not None:
        plt.imshow(np.uint8(image),cmap=cmap)
    else:
        plt.imshow(np.uint8(image))
    if title is not None:
        plt.title(title)

    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    img = cv.cvtColor(cv.imread('segment.jpg'), cv.COLOR_BGR2HSV)
    show_image(img)