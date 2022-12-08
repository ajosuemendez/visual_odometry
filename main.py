from images import load_images
from feature_detector import FeatureDetector
import cv2 as cv
import numpy as np

def preprocess_img(img):
    width, height = 600, 600
    dim = (width, height)
    resized_img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    img_grey = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
    return img_grey

if __name__ == "__main__":
    #get images function takes (path to the first image, starting index to slice, ending index to slice)
    print("Getting Images....")
    img_collection = load_images("/home/alejandro/Downloads/Run2/Center/frame000", 165, 300)
    print("Images loaded....")

    current_image = preprocess_img(img_collection[5])

    fd = FeatureDetector(current_image)
    kp = fd.get_key_points()
    print(kp)
