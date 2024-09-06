import cv2
import numpy as np
import matplotlib.pyplot as plt

def asymmetry(filename):
    # Read the image
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    cv2.imshow('Original Image', img)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray Image', gray)

    #let the user add a bounding box on the picture and then crop the image from the bounding box
    r = cv2.selectROI("select the area", img)
    print(r)
    #ok now with r crop the image to get a square picture
    imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    cv2.imshow('Cropped Image', imCrop)

    #turn it into gray
    gray = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray Image', gray)

    #threshold
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('Threshold Image', thresh)

    #open 
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imshow('Opening Image', opening)

    #reverse the colors
    opening = cv2.bitwise_not(opening)

    #pivot 
    pivot_opening = cv2.flip(opening, 1)
    cv2.imshow('Pivot Image', pivot_opening)

    #get the intersection between the two images
    intersection = cv2.bitwise_and(opening, pivot_opening)
    cv2.imshow('Intersection Image', intersection)

    #get the union between the two images
    union = cv2.bitwise_or(opening, pivot_opening)
    cv2.imshow('Union Image', union)

    #get number of 1 in the intersection
    count = cv2.countNonZero(intersection)
    print(count)

    #get the number of 1 in the opening
    count_opening = cv2.countNonZero(opening)
    print(count_opening)

    #get the number of 1 in the pivot
    count_pivot = cv2.countNonZero(pivot_opening)
    print(count_pivot)

    #compute dice coefficient
    dice = (2*count)/(count_opening + count_pivot)
    print(dice)

    #get the union - intersection
    ext = cv2.bitwise_xor(opening, pivot_opening)
    cv2.imshow('Union - Intersection Image', ext)
    
    count_union = cv2.countNonZero(union)
    print(count_union)
    count_ext = cv2.countNonZero(ext)
    print(count_ext)

    #compute the asymmetry
    asymmetry = count_ext/count_union
    print(asymmetry)


    print(union.shape)
    print(ext.shape)



def main():
    asymmetry('data/asymmetry.jpg')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()