import cv2
import numpy as np
from ..builder import PIPELINES

def sharpen(image):
    # 定义一个卷积核
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # 对图像进行卷积运算
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

@PIPELINES.register_module()
class Sharpen():
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def __call__(self, results):
        """Call function to make a mixup of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mixup transformed.
        """
        print(results)
        print(type(results))
        print(len(results['img']))
        # results = self.apply_image(results['img'])
        if np.random.rand() < self.prob:
            results['img'] = sharpen(results['img'])
            return results
        # return img
        return results

    # def apply_image(self, img):
    #     if np.random.rand() < self.prob:
    #         sharpened = sharpen(img)
    #         return sharpened
    #     return img