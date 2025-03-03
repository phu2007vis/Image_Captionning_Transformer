import numpy as np
from torchvision import transforms
class GaussianBlur:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        #kernel = [(31,31)] prev 1 level only
        kernel = (31, 31)
        sigmas = [.5, 1, 2]
        if mag<0 or mag>=len(kernel):
            index = np.random.randint(0, len(sigmas))
        else:
            index = mag

        sigma = sigmas[index]
        return transforms.GaussianBlur(kernel_size=kernel, sigma=sigma)(img)