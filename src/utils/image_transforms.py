import numpy as np
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