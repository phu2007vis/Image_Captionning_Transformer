import numpy as np
import torch
def tensor_to_image(tensor_image,
                    mean = np.array([1,1, 1]),
                    std = np.array([0, 0, 0])
                    ):
    tensor_image = tensor_image.squeeze(0)  # Remove batch dimension
    numpy_image = tensor_image.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
    
    # Denormalize the image
    denormalized_image = (numpy_image * std + mean) * 255
    denormalized_image = denormalized_image.astype(np.uint8)
    return denormalized_image

import random
import math

class RandomErasing(object):
    """Randomly selects a rectangle region in an image and erases its pixels.
       'Random Erasing Data Augmentation' by Zhong et al.
       See https://arxiv.org/pdf/1708.04896.pdf
    Args:
        probability: The probability that the Random Erasing operation will be performed.
        sl: Minimum proportion of erased area against input image.
        sh: Maximum proportion of erased area against input image.
        r1: Minimum aspect ratio of erased area.
        mean: Erasing value (not used if you don't want to change mean).
    """

    def __init__(self, p=0.5, sl=0.02, sh=0.08, r1=0.5, value = 1):
        self.probability = p
        self.value = value  # mean is not used anymore
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        # Decide whether to apply random erasing or not based on probability
        if random.uniform(0, 1) >= self.probability:
            return img
        C,H,W = img.shape
            # Get the image dimensions (C, H, W)
        area = img.size()[1] * img.size()[2]

        # Calculate the target area for the erased region
        target_area = random.uniform(self.sl, self.sh) * area
        aspect_ratio = random.uniform(self.r1, 1 / self.r1)

        # Calculate the height and width of the erased region based on the aspect ratio
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        h = min(H-2,h)
        w = min(W-2,w)
        
        
        x1 = random.randint(2, img.size()[1] - h)
        y1 = random.randint(2, img.size()[2] - w)

        mask = torch.zeros((H, W), dtype=torch.bool)
        mask[x1:x1 + h, y1:y1 + w] = torch.randn((h,w)) > 0.2
        
        img = img.masked_fill(mask.unsqueeze(0), self.value)
        
        # img[:, x1:x1 + h, y1:y1 + w] = (self.value,self.value,self.value)  #
        
        return img
        

