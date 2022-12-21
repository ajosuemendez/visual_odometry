import cv2 as cv

class Utils:
    @staticmethod
    def draw_circles(img, feature_points, radius=3, color=(0, 0, 255)):
        for elem in feature_points:
            #print(elem)
            # x = elem[0]
            # y = elem[1]
            x = int(elem.pt[0])
            y = int(elem.pt[1])
            cv.circle(img,(x, y), radius, color)

        return img

    @staticmethod
    def draw_kp_vectors(img, matches, prev_kp, current_kp, thickness=1, color=(0, 255, 0)):
        match_coordinates_prev_img = [(int(prev_kp[m.queryIdx].pt[0]), int(prev_kp[m.queryIdx].pt[1])) for m in matches]
        match_coordinates_current_img = [(int(current_kp[m.trainIdx].pt[0]), int(current_kp[m.trainIdx].pt[1])) for m in matches]

        for i, (prev_img_coord, current_img_coord) in enumerate(zip(match_coordinates_prev_img, match_coordinates_current_img)):
            cv.line(img, current_img_coord, prev_img_coord, color, thickness)

        return img

    @staticmethod
    def load_images(path, start_index, end_index):
        images = []
        for i in range(start_index, end_index):
            images.append(cv.imread(f"{path}{i}.jpg"))
        return images
