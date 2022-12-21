import cv2 as cv

class Utils:
    @staticmethod
    def draw_circles(img, feature_points):
        radius = 5
        RED = (0,0,255)
        for elem in feature_points:
            #print(elem)
            # x = elem[0]
            # y = elem[1]
            x = int(elem.pt[0])
            y = int(elem.pt[1])
            cv.circle(img,(x, y), radius, RED)

        return img

    @staticmethod
    def load_images(path, start_index, end_index):
        images = []
        for i in range(start_index, end_index):
            images.append(cv.imread(f"{path}{i}.jpg"))
        return images
