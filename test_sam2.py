import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Assuming your module is named sam2 and SAM2Model is in that module
from sam2_model import SAM2Model

def main():
    # Create an instance of the SAM2 model
    sam2_model = SAM2Model()
    
    # Read the image from the specified path
    image_path = "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/demo_data/snackbox/snackbox_flip/rgb/0001.png"
    
    # Load the image using PIL
    image = Image.open(image_path)
    
    # Convert the image to an array that can be processed
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Use the SAM2 model to predict the mask
    mask = sam2_model.predict(image_np, first=True, viz=False)

    # Show the original image and the mask
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    
    # Masked image
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title("Predicted Mask")
    
    # Display the plots
    plt.show()

if __name__ == "__main__":
    main()
