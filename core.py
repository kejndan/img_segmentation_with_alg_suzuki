import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from utils import clockwise_walking


class ImageSegmentation:
    def __init__(self, image):
        self.image_hsv = cv.cvtColor(cv.imread(image), cv.COLOR_BGR2GRAY)
        self.image_rgb = cv.cvtColor(self.image_hsv, cv.COLOR_GRAY2RGB)

        self.lvl_lower_hsv = np.array([32, 40, 0])
        self.lvl_upper_hsv = np.array([80, 210, 160])
        self.lvl_lower_hsv = np.array([0, 0, 0])
        self.lvl_upper_hsv = np.array([255, 255, 255])
        self.blur_kernel_size = 21
        self.blur_sigma = 145

        self.marker = 1
        self.lnbd = None
        self.contours = []
        self.hierarchy = {1:[]}

    def _get_mask(self, show=False):
        # blurry_img = cv.GaussianBlur(self.image_hsv, (self.blur_kernel_size, self.blur_kernel_size), self.blur_sigma)
        # self.mask_image = np.where(cv.inRange(blurry_img, self.lvl_lower_hsv, self.lvl_upper_hsv) == 255, 1, 0)
        self.mask_image = np.where(self.image_hsv == 255, 1, 0)
        if show:
            plt.imshow(self.mask_image, cmap='gray')
            plt.show()
        return self.mask_image

    def _get_contour(self, show=False):
        for i in range(1, self.mask_image.shape[0] - 1):
            self.lnbd = 1
            for j in range(1, self.mask_image.shape[1] - 1):

                if self.mask_image[i, j] == 0:
                    continue

                start_point = None

                if abs(self.mask_image[i, j]) == self.lnbd and self.mask_image[i, j + 1] == 0 and self.marker > 1:
                    self.lnbd = abs(self.mask_image[i, j]) - 1
                elif self.mask_image[i, j] != 0 and self.mask_image[i, j] != 1 and self.mask_image[i, j - 1] == 0:
                    self.lnbd = abs(self.mask_image[i, j])

                if self.mask_image[i, j - 1] == 0 and self.mask_image[i, j] == 1:
                    start_point = (i, j - 1)

                elif self.mask_image[i, j] >= 1 and self.mask_image[i, j + 1] == 0:
                    start_point = (i, j + 1)
                if start_point is None:
                    continue


                anchor_point = (i, j)
                found_point = self.__search_nonzero_pixel(anchor_point, start_point)
                self.marker += 1
                self.hierarchy[self.lnbd].append(self.marker)
                if found_point is not None:
                    init_points = (anchor_point, found_point)
                    self.contours.append([])

                    self.__create_contour(init_points)
                    self.contours[-1] = np.asarray(self.contours[-1])
                else:
                    self.mask_image[anchor_point] = -self.marker

        if show:
            img = np.where(self.mask_image == 1, 0, self.mask_image)
            img = np.where(img != 0, 1, img)
            plt.imshow(img, cmap='gray')
            plt.show()

    def __search_nonzero_pixel(self, anchor_point, start_point,counter=False):
        for searching_point in clockwise_walking(anchor_point, start_point, counter):
            if searching_point is not None:
                if searching_point[0] > self.mask_image.shape[0] - 1\
                        or searching_point[1] > self.mask_image.shape[1] - 1\
                        or searching_point[0] < 0 or searching_point[1] < 0:
                    continue
                elif self.mask_image[searching_point] != 0:
                    return searching_point
            else:
                return None

    def __create_contour(self, init_points):
        anchor_point, last_searching_point = init_points
        while True:
            found_point = self.__search_nonzero_pixel(anchor_point, last_searching_point, True)

            if found_point is None:
                break
            else:
                if anchor_point[1] + 1 < self.mask_image.shape[1]:
                    if self.mask_image[anchor_point[0], anchor_point[1] + 1] == 0:
                        self.mask_image[anchor_point] = -self.marker
                        self.contours[-1].append(list(anchor_point))
                    elif self.mask_image[anchor_point] == 1:
                        self.mask_image[anchor_point] = self.marker
                        self.contours[-1].append(list(anchor_point))
                if found_point == init_points[0] and anchor_point == init_points[1]:
                    self.lnbd = self.marker
                    self.hierarchy[self.lnbd] = []

                    break

                else:
                    last_searching_point = anchor_point
                    anchor_point = found_point

    def _get_biggest_boxes(self, thr):
        boxes = {}
        for contour in self.contours:
            if len(contour) > 0:
                x, y, w, h = cv.boundingRect(contour)
                start = (y, x)
                end = (y+h, x+w)
                square = w*h
                if square > thr:
                    boxes[square] = [start, end, contour]
        self.boxes = boxes
        return boxes

    def _fill_object(self):
        for box in self.boxes.values():
            q = box[2]
            cv.fillPoly(self.image_rgb, [box[2][:, [1, 0]]], color=(255, 0, 255))

    def _draw_boxes(self):
        for rect in self.boxes.values():
            self.image_rgb = cv.rectangle(self.image_rgb, rect[0], rect[1], (255, 0, 0), 3)


    def run(self):
        self._get_mask(True)
        mask = np.asarray(self.mask_image*255, dtype=np.uint8)
        self._get_contour(True)

        self._get_biggest_boxes(130000).keys()
        self._fill_object()
        self._draw_boxes()
        plt.imshow(self.image_rgb)
        plt.show()


if __name__ == '__main__':
    IS = ImageSegmentation('segment.jpg')
    pass



