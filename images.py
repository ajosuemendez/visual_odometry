import cv2 as cv

def load_images(path, start_index, end_index):
    images = []
    for i in range(start_index, end_index):
        images.append(cv.imread(f"{path}{i}.jpg"))
    return images
