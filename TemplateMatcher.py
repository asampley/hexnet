import cv2

class TemplateMatcher:
    def __init__( self, template, threshold, method = cv2.TM_SQDIFF_NORMED ):
        """
        template is a numpy array
        threshold is a number
        """
        self.template = template
        self.threshold = threshold
        self.method = method

    def matchRating(self, image):
        """
        image is a numpy array, with the same number of dimensions
        as template.
        image has the first two dimensions reshaped to match template.
        """
        image = cv2.resize(image, (self.template.shape[1], self.template.shape[0]))

        if image.shape != self.template.shape:
            raise ValueError('image and template shape do not match: image is ' 
                    + str(image.shape) + ' and template is ' + str(self.template.shape))

        result = cv2.matchTemplate(image, self.template, self.method)
        minVal, maxVal, _, _ = cv2.minMaxLoc(result)

        if self.method == cv2.TM_SQDIFF or self.method == cv2.TM_SQDIFF_NORMED:  
            return minVal
        else:
            return maxVal

    def isMatch(self, image):
        """
        image is a numpy array, with the same number of dimensions
        as template.
        image has the first two dimensions reshaped to match template.
        """
        image = cv2.resize(image, (self.template.shape[1], self.template.shape[0]))

        if image.shape != self.template.shape:
            raise ValueError('image and template shape do not match: image is ' 
                    + str(image.shape) + ' and template is ' + str(self.template.shape))

        result = cv2.matchTemplate(image, self.template, self.method)
        minVal, maxVal, _, _ = cv2.minMaxLoc(result)

        if self.method == cv2.TM_SQDIFF or self.method == cv2.TM_SQDIFF_NORMED:  
            return minVal <= self.threshold
        else:
            return maxVal >= self.threshold
