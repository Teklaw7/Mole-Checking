import cv2

def get_union_intersection_xor(img1, img2):
    union = cv2.bitwise_or(img1, img2)
    intersection = cv2.bitwise_and(img1, img2)
    xor = cv2.bitwise_xor(img1, img2)

    return union, intersection, xor

def asymmetry_computation(union, xor):
    count_union = cv2.countNonZero(union)
    count_xor = cv2.countNonZero(xor)

    return 1 - (count_xor/count_union)

def dice_computation(intersection, opening, pivot_opening):
    count = cv2.countNonZero(intersection)
    count_opening = cv2.countNonZero(opening)
    count_pivot = cv2.countNonZero(pivot_opening)

    return (2*count)/(count_opening + count_pivot)
