import cv2
import numpy as np
import matplotlib.pyplot as plt


# 1. Define a function for Otsu's thresholding
def otsu_thresholding_and_display(image_path):
    """
    Loads an image, converts it to grayscale, applies Otsu's thresholding,
    and displays the original and processed images along with the threshold value.
    The performance is verified by displaying the calculated threshold value and
    the resulting binary image.
    """
    # Load the image in grayscale (fallback to synthetic image if loading fails)
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Synthetic image (simple two-level image for clear demo)
            img = np.array([
                [10, 10, 10, 150, 150, 150],
                [10, 10, 10, 150, 150, 150],
                [10, 10, 10, 150, 150, 150],
                [10, 10, 10, 150, 150, 150],
            ], dtype=np.uint8)
            print("Note: Could not load image. Using a synthetic image for demonstration.")
    except Exception:
        img = np.array([
            [10, 10, 10, 150, 150, 150],
            [10, 10, 10, 150, 150, 150],
            [10, 10, 10, 150, 150, 150],
            [10, 10, 10, 150, 150, 150],
        ], dtype=np.uint8)
        print("Note: Could not load image. Using a synthetic image for demonstration.")

    # 2. Pre-processing: Apply Gaussian Blur
    # Often necessary to remove noise that could skew Otsu's histogram analysis.
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

    # 3. Apply Otsu's thresholding
    # cv2.THRESH_OTSU automatically finds the optimal threshold value.
    thresh_value, thresholded_img = cv2.threshold(
        blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --- Performance Verification Output ---
    print(f"Calculated Otsu's Threshold Value: {thresh_value:.2f}")

    # 4. Display Results (Visual Verification)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Grayscale Image')
    axes[0].axis('off')

    axes[1].imshow(thresholded_img, cmap='gray')
    axes[1].set_title(f"Otsu's Result (Thresh={thresh_value:.0f})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


# --- Example Usage ---
# Replace "path/to/your/image.jpg" with the actual path to your image.
# The code includes a fallback to a synthetic image if a real file is not accessible.
image_file_path = "path/to/your/image.jpg"
otsu_thresholding_and_display(image_file_path)