import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils

def asymmetry(filename):
    # Read the image
    img = cv2.imread(filename, cv2.IMREAD_COLOR)

    # Select ROI and crop the image
    r = cv2.selectROI("select the area", img)
    img = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Apply morphological opening
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Invert the image and flip it
    opening = cv2.bitwise_not(opening)
    pivot_opening = cv2.flip(opening, 1)

    # Compute union, intersection, and XOR
    union, intersection, ext = utils.get_union_intersection_xor(opening, pivot_opening)

    # Compute asymmetry and Dice coefficient
    asymmetry = utils.asymmetry_computation(union, ext)
    dice = utils.dice_computation(intersection, opening, pivot_opening)

    if asymmetry < 0.8:
        print('Asymmetry: ', asymmetry)
        print('Dice: ', dice)
        print('The lesion is malignant')
    else:
        print('Asymmetry: ', asymmetry)
        print('Dice: ', dice)
        print('The lesion is benign')

def main():
    asymmetry('data/both.jpg')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
