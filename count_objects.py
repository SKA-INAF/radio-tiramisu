import cv2
import torch
import numpy as np
from pathlib import Path
from astropy.io import fits

def compute_connected_components(image_tensor):
    # Convert the image tensor to a NumPy array
    image_array = image_tensor.numpy()

    # Convert the array to uint8 and transpose dimensions if necessary
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    image_array = np.transpose(image_array, (1, 2, 0))

    image_array = np.repeat(image_array, 3, axis=2)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Apply a binary threshold to convert to black and white
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute the number of contours (connected components)
    num_components = len(contours)

    return num_components

total_objects = 0
for mask_path in Path("data/rg-dataset/data/synthetic/gen_masks/cond_fits_extended").glob("*.fits"):
    mask = torch.from_numpy(fits.getdata(mask_path).astype(np.float32)).float()
    num_components = compute_connected_components(mask)
    total_objects += num_components
    print(f"{mask_path.name}: {num_components}")

print("Total Objects: ", total_objects)