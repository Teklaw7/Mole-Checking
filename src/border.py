import cv2
import numpy as np
import matplotlib.pyplot as plt


def analyze_mole_border(image_path):
    image = cv2.imread(image_path)
    original_image = image.copy()
    mask = np.zeros(image.shape[:2], np.uint8)

    height, width = image.shape[:2]
    margin = 20
    rect = (margin, margin, width - margin * 2, height - margin * 2)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    mole_segmented = image * mask2[:, :, np.newaxis]

    mole_gray = cv2.cvtColor(mole_segmented, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(mole_gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No contours found. Ensure the mole is clear in the image.")
        return

    contour = max(contours, key=cv2.contourArea)

    cv2.drawContours(original_image, [contour], -1, (0, 255, 0), 2)

    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    
    irregularity_ratio = perimeter ** 2 / (4 * np.pi * area)
    print(f"Irregularity Ratio: {irregularity_ratio:.2f}")

    if irregularity_ratio > 1.5:  # Irregularity threshold
        print("The mole has an irregular border (potential concern).")
    else:
        print("The mole's border looks regular.")

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask2, cmap='gray')
    plt.title("GrabCut Segmentation")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Mole with Detected Contour")
    plt.axis("off")

    plt.show()


def main():
    analyze_mole_border('data/normal.jpg')

if __name__ == "__main__":
    main()
