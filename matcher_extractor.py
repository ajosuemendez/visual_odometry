import cv2 as cv

class MatcherExtractor:
    def __init__(self):
        self.brute_force_matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        self.orb = cv.ORB_create()
        self.has_current_desc = False
        self.previous_desc = None
        self.current_desc = None
        self.previous_kp = None
        self.current_desc = None

    ##############SETTERS####################

    def set_current_desc(self, desc):
        self.current_desc = desc
        self.has_current_desc = True

    def set_previous_desc(self, desc):
        self.previous_desc = desc

    def set_current_kp(self, kp):
        self.current_kp = kp

    def set_previous_kp(self, kp):
        self.previous_kp = kp

    ############GETTERS######################

    def get_current_desc(self):
        return self.current_desc

    def get_previous_desc(self):
        return self.previous_desc

    def get_current_kp(self):
        return self.current_kp

    def get_previous_kp(self):
        return self.previous_kp

    #########METHODS#########################

    def generate_descriptions(self, image, keypoints):
        kp, desc = self.orb.compute(image, keypoints)
        return kp, desc

    def compute_matches(self, desc1, desc2):
        matches = self.brute_force_matcher.knnMatch(desc1,desc2,k=2)
        #matches = list(filter(lambda elem: (elem.distance <20), matches))
        good_matches = []
        for m, n in matches:
            if m.distance < 0.50*n.distance:
                good_matches.append(m)
        return good_matches
