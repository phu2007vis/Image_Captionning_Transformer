import cv2
import os
import numpy as np

def preprocess_image(image_path):
    """
    Preprocesses an image by converting it to grayscale, applying adaptive thresholding, 
    and reducing noise with median blur.
    """
    # Read the image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    binary_image = cv2.medianBlur(binary_image, 3)
    return binary_image

def preprocess_folder(input_folder, output_folder):
    """
    Processes all images in the input folder, combines the original and processed images side by side,
    and saves the combined images to the output folder.
    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save processed images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if os.path.isfile(input_path):  # Skip directories
            try:
                # Read the original image
                original_image = cv2.imread(input_path)

                # Preprocess the image
                processed_image = preprocess_image(input_path)

                # Convert processed image to BGR for concatenation
                processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

                # Resize images to ensure they have the same height
                original_height, original_width = original_image.shape[:2]
                processed_image_resized = cv2.resize(
                    processed_image_bgr, (original_width, original_height)
                )

                # Concatenate original and processed images
                combined_image = np.hstack((original_image, processed_image_resized))

                # Save the combined image to the output folder
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, combined_image)
                # print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

# Example usage:
# preprocess_folder("path/to/input/folder", "path/to/output/folder")

preprocess_folder("/work/21013187/phuoc/Image_Captionning_Transformer/results/test_reuslt_300/dung", "/work/21013187/phuoc/Image_Captionning_Transformer/results/preprocesing_result")
