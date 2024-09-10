import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Assuming your module is named sam2 and SAM2Model is in that module
from sam2rt_model import SAM2Model


def main():
    # Create an instance of the SAM2 model
    sam2_model = SAM2Model()

    # Read the image from the specified path
    image_path = "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/demo_data/snackbox/snackbox_flip/rgb/0001.png"

    # Load the image using PIL
    image = Image.open(image_path)

    # Convert the image to an array that can be processed
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Read the image from the specified path
    image_path2 = "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/demo_data/snackbox/snackbox_flip/rgb/0002.png"

    # Load the image using PIL
    image2 = Image.open(image_path2)

    # Convert the image to an array that can be processed
    image_np2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)

    # Use the SAM2 model to predict the mask
    mask = sam2_model.predict(image_np, first=True, viz=False)
    mask2 = sam2_model.predict(image_np2, first=False, viz=False)

    # Show the original image and the mask
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original Image")

    # Masked image
    plt.subplot(1, 4, 2)
    plt.imshow(mask)
    plt.title("Predicted Mask")

    # Original image
    plt.subplot(1, 4, 3)
    plt.imshow(image2)
    plt.title("Original Image 2")

    # Masked image
    plt.subplot(1, 4, 4)
    plt.imshow(mask2)
    plt.title("Predicted Mask 2")

    # Display the plots
    plt.show()


if __name__ == "__main__":
    main()
