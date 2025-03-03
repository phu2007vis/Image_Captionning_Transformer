import torch
import numpy as np
from torchvision import transforms
from PIL import Image

class HideAndSeekErasing(object):
    def __init__(self, length, random_value=True):
        """
        Parameters:
        length (int): Length of the erasing region (width and height).
        random_value (bool): If True, use random values for the erased region.
                             If False, set the region to black (zeros).
        """
        self.length = length
        self.random_value = random_value

    def __call__(self, img):
        # Get image dimensions
        _,W, H = img.shape[::-1]
        
        # Randomly select the center of the erased patch
        x = np.random.randint(0, W)
        y = np.random.randint(0, H)
        
        # Define the coordinates of the rectangular region to be erased
        x1 = np.clip(x - self.length // 2, 0, W)
        y1 = np.clip(y - self.length // 2, 0, H)
        x2 = np.clip(x + self.length // 2, 0, W)
        y2 = np.clip(y + self.length // 2, 0, H)

        # Convert image to numpy array
        img_np = np.array(img)

        # Erase the region
        if self.random_value:
            # Replace with random values (could be noise, here we use random values between 0-255)
            img_np[y1:y2, x1:x2, :] = np.random.randint(0, 256, size=(y2 - y1, x2 - x1, img_np.shape[2]))
        else:
            # Replace with black (set to 0)
            img_np[y1:y2, x1:x2, :] = 0

        # Convert back to PIL Image
        return Image.fromarray(img_np)

# Example usage with a dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    HideAndSeekErasing(length=50),  # Erasing a 50x50 region with random values
])

# Load an image
img = Image.open(r"/work/21013187/phuoc/Image_Captionning_Transformer/results/test_reuslt/khong_dung/0->8->rotatengoaigiao13_jpg.rf.2014c897380cea51b2cf0cffe0f2fd81_8.jpg")

# Apply Hide-and-Seek Erasing transform
img_transformed = transform(img)

# Convert tensor back to PIL image for saving
img_transformed_pil = transforms.ToPILImage()(img_transformed)

# Save the transformed image to a file
img_transformed_pil.save("transformed_image.jpg")  # You can choose a different file format or name
