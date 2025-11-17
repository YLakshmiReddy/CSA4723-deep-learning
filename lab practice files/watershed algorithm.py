import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage


# 1. Define the function for Watershed Algorithm application
def watershed_segmentation_and_verify(image_path):
    """
    Loads an image, applies pre-processing, uses the Watershed algorithm for segmentation.
    Performance is verified by visualizing key intermediate steps (markers) and the
    final segmented image with boundaries highlighted.
    """
    # Load the image (Fallback to synthetic image if loading fails)
    try:
        img = cv2.imread(image_path)
        if img is None:
            # Synthetic image simulating two overlapping objects
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.circle(img, (30, 30), 20, (255, 255, 255), -1)
            cv2.circle(img, (70, 70), 25, (255, 255, 255), -1)
            cv2.circle(img, (80, 20), 15, (255, 255, 255), -1)
            print("Note: Could not load image. Using a synthetic image for demonstration.")
    except Exception:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (30, 30), 20, (255, 255, 255), -1)
        cv2.circle(img, (70, 70), 25, (255, 255, 255), -1)
        cv2.circle(img, (80, 20), 15, (255, 255, 255), -1)
        print("Note: Could not load image. Using a synthetic image for demonstration.")

    # Pre-processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Otsu's Thresholding (to get initial binary map of objects)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Noise removal/Smoothing using Morphological Opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- Marker Extraction ---

    # 3. Sure Background Area (Dilating the opened image)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 4. Sure Foreground Area (Finding peaks using Distance Transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # 5. Unknown region (Area between sure background and sure foreground)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Create Markers for sure foreground
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is 1 (not 0)
    markers = markers + 1

    # Mark the unknown region with 0
    markers[unknown == 255] = 0

    # 7. Apply Watershed
    # Watershed segments the image based on the markers matrix.
    markers = cv2.watershed(img, markers)

    # 8. Verification: Highlight boundaries on the original image
    # The watershed output "markers" has boundaries marked with -1.
    img[markers == -1] = [255, 0, 0]  # Mark boundaries in Red (BGR format)

    # --- Display Results for Performance Verification ---

    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), gray, thresh, opening, sure_fg, markers]
    titles = [
        'Final Segmented Image (Red Boundaries)',
        'Grayscale',
        "Otsu's Threshold (Binary)",
        'Morphological Opening',
        'Sure Foreground Markers',
        'Watershed Markers Matrix'
    ]
    cmaps = [None, 'gray', 'gray', 'gray', 'gray', 'jet']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # Handle BGR to RGB conversion for the final image
        if titles[i] == 'Final Segmented Image (Red Boundaries)':
            ax.imshow(images[i])
        else:
            ax.imshow(images[i], cmap=cmaps[i])
        ax.set_title(titles[i])
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# --- Example Usage ---
# Use a placeholder path. The code is designed to fallback to a synthetic image.
image_file_path = "path/to/your/image.jpg"
watershed_segmentation_and_verify(image_file_path)