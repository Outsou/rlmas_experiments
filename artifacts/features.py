from creamas.rules.feature import Feature
from utilities.math import box_count

import numpy as np
import cv2

class NoveltyFeature(Feature):
    def __init__(self, agent):
        super().__init__('novelty', ['image'], float)
        self.agent = agent

    def extract(self, artifact):
        return float(self.agent.novelty(artifact))


class ImageComplexityFeature(Feature):
    def __init__(self):
        super().__init__('image_complexity', ['image'], float)

    def extract(self, artifact):
        img = artifact.obj * 255
        np.around(img)
        img_uint8 = np.uint8(img)
        edges = cv2.Canny(img_uint8, 100, 200)
        return float(box_count(edges))
