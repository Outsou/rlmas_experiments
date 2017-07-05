from creamas.rules.feature import Feature
from utilities.math import box_count

import numpy as np
import cv2

class AgentAttacher():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent = None

    def attach_agent(self, agent):
        self.agent = agent


class NoveltyFeature(AgentAttacher, Feature):
    def __init__(self):
        kwargs = {'name': 'novelty', 'domains': ['image'], 'rtype': float}
        super().__init__(**kwargs)

    def extract(self, artifact):
        if self.agent is None:
            raise ValueError('Agent not set')
        return float(self.agent.novelty(artifact))


class ImageComplexityFeature(Feature):
    def __init__(self):
        super().__init__('image_complexity', ['image'], float)

    def extract(self, artifact):
        edges = get_edges(artifact.obj)
        return float(box_count(edges))


def get_edges(img):
    img_255 = np.around(img * 255)
    img_uint8 = np.uint8(img_255)
    edges = cv2.Canny(img_uint8, 100, 200)
    return edges