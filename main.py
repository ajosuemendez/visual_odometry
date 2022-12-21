from feature_detector import FeatureDetector
from matcher_extractor import MatcherExtractor
from utils import Utils
import cv2 as cv
import numpy as np

def preprocess_img(img):
    #print(img.shape) #shape(height, width)
    width, height = img.shape[1]//2, img.shape[0]//2
    dim = (width, height)
    resized_img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    img_grey = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
    return img_grey, resized_img

if __name__ == "__main__":
    #get images function takes (path to the first image, starting index to slice, ending index to slice)
    print("Getting Images....")
    img_collection = Utils.load_images("/home/alejandro/Downloads/Run2/Center/frame000", 165, 170)
    print("Images loaded....")

    fd = FeatureDetector()
    matcher_extractor = MatcherExtractor()

    for index, image in enumerate(img_collection):
        current_gray_img, current_color_img = preprocess_img(image)
        fd.feed_image(current_gray_img)
        kp = fd.get_key_points()

        kp, desc = matcher_extractor.generate_descriptions(current_gray_img, kp)

        if matcher_extractor.has_current_desc:
            matcher_extractor.set_previous_desc(matcher_extractor.get_current_desc())
            matcher_extractor.set_previous_kp(matcher_extractor.get_current_kp())


        matcher_extractor.set_current_desc(desc)
        matcher_extractor.set_current_kp(kp)

        if matcher_extractor.get_previous_desc() is not None:
            matches = matcher_extractor.compute_matches(desc1 = matcher_extractor.get_previous_desc(),
                                                        desc2 = matcher_extractor.get_current_desc())

            print(f"Num Matches: {len(matches)}")

        print("Number of KeyPoints:", len(kp))

        current_color_img = Utils.draw_circles(current_color_img, kp)
        cv.imshow(f"Image {index}", current_color_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
