import cv2 as cv
import numpy as np

class FeatureDetector:
    def __init__(self, img):
        self.img = img
        self.max_corners = 40
        self.quality_level = 0.001
        self.min_distance = 1
        self.dim = (self.img.shape[0], self.img.shape[1])
        self.define_grid()

    def define_grid(self, x=10, y=10, x_lim=0, y_lim=0):
        self.grid_x = x
        self.grid_y = y

        #Ignores the last grids
        self.grid_limit_x = x_lim
        self.grid_limit_y = y_lim


    def get_key_points(self):
        x_step = self.dim[0] // self.grid_x
        y_step = self.dim[1] // self.grid_y

        points = []

        for i in range(0, self.grid_y - self.grid_limit_y):
            for j in range(0, self.grid_x - self.grid_limit_x):
                temp_corners = cv.goodFeaturesToTrack(self.img[int(y_step*i):int(y_step*(i+1)), int(x_step*j):int(x_step*(j+1))], self.max_corners, self.quality_level, self.min_distance)
                temp_corners = np.int0(temp_corners)

                #add the grid difference
                for k in range(0, temp_corners.shape[0]):
                    temp_corners[k][0][0] = temp_corners[k][0][0] + int(x_step*(j))
                    temp_corners[k][0][1] = temp_corners[k][0][1] + int(y_step*(i))

                points = np.concatenate((points, temp_corners), axis=None)

        kp = []
        for _ , (x, y) in enumerate(zip(points[0::2], points[1::2])):
            kp.append(cv.KeyPoint(x=int(x), y=int(y), _size=20))

        return kp
