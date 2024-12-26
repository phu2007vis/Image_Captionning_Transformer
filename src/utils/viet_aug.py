from PIL import Image
import numpy as np

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
import random


class RandomDottedLine(ImageOnlyTransform):
    def __init__(self, num_lines=1, p=0.5):
        super(RandomDottedLine, self).__init__(p=p)
        self.num_lines = num_lines

    def apply(self, img, **params):
        h, w = img.shape[:2]
        for _ in range(self.num_lines):
            # Random start and end points
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
            # Random color
            color = tuple(np.random.randint(0, 256, size=3).tolist())
            # Random thickness
            thickness = np.random.randint(1, 5)
            # Draw dotted or dashed line
            line_type = random.choice(["dotted", "dashed", "solid"])
            if line_type != "solid":
                self._draw_dotted_line(
                    img, (x1, y1), (x2, y2), color, thickness, line_type
                )
            else:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

        return img

    def _draw_dotted_line(self, img, pt1, pt2, color, thickness, line_type):
        # Calculate the distance between the points
        dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        # Number of segments
        num_segments = max(int(dist // 5), 1)
        # Generate points along the line
        x_points = np.linspace(pt1[0], pt2[0], num_segments)
        y_points = np.linspace(pt1[1], pt2[1], num_segments)
        # Draw segments
        for i in range(num_segments - 1):
            if line_type == "dotted" and i % 2 == 0:
                pt_start = (int(x_points[i]), int(y_points[i]))
                pt_end = (int(x_points[i]), int(y_points[i]))
                cv2.circle(img, pt_start, thickness, color, -1)
            elif line_type == "dashed" and i % 4 < 2:
                pt_start = (int(x_points[i]), int(y_points[i]))
                pt_end = (int(x_points[i + 1]), int(y_points[i + 1]))
                cv2.line(img, pt_start, pt_end, color, thickness)
        return img

    def get_transform_init_args_names(self):
        return ("num_lines",)


class ImgAugTransformV2:
    def __init__(self):
        self.aug = A.Compose(
            [
                A.ColorJitter(p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.Perspective(scale=(0.01, 0.05)),
                # RandomDottedLine(),
            ]
        )

    def __call__(self, img):
        img = np.asarray(img)
        transformed = self.aug(image=img)
        img = transformed["image"]
        img = Image.fromarray(img)
        return img
    
if __name__ == "__main__":
    image_path  = "/work/21013187/phuoc/Image_Captionning_Transformer/data/OCR_Phuoc/train/images/1PlateBaza493.jpg"
    out_path = "/work/21013187/phuoc/Image_Captionning_Transformer/data/test/test.png"
    transform = ImgAugTransformV2()
    pil_image = Image.open(image_path)
    pil_image = transform(pil_image)
    pil_image.save(out_path)
    import os
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
    image_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/OCR_Phuoc/train/images"
    image_save_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/test"
    image_height =  224
    image_min_width =  120
    image_max_width=  448
    from utils.resize_image import process_image
  
    os.makedirs(image_folder,exist_ok= True)
    for image_name in os.listdir(image_folder):
        
        image_path = os.path.join(image_folder,image_name)
        save_path  = os.path.join(image_save_folder,image_name)
        pil_image = Image.open(image_path)
        pil_image = process_image(pil_image,image_height,image_min_width,image_max_width)
        pil_image = transform(pil_image)
        pil_image.save(save_path)
    
    
    