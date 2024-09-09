import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
from sklearn.cluster import KMeans

def color(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)

    r = cv2.selectROI("select the area", img)
    img = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    cv2.imshow("ROI", img)

    #now make a treshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Threshold", thresh)

    #ok now from the treshold result get all the points which are one
    points = np.argwhere(thresh == 255)
    #then on the ROI image, make all the points which are in points to be at 0
    img[points[:,0], points[:,1]] = [0, 0, 0]
    cv2.imshow("Color", img)

    #now make the picture in gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)


    gray_without_black = gray[gray != 0]
    #get number of unique values in the gray image where the black is not included
    unique = np.unique(gray_without_black)
    print('Number of unique values in the gray image: ', len(unique) - 1)
    print('The unique values in the gray image: ', unique)
    sum = []
    for i in range(len(unique)):
        sum.append(np.sum(gray == unique[i]))

    print('The number of pixels for each unique value: ', sum)

    #now create groups for the unique values : create a group for values between 200 and 220 and another between 220 and 240
    distance_unique_values = np.max(unique) - np.min(unique)
    print('The distance between the unique values: ', distance_unique_values)

    #start a loop from np.min to np.max with a step of 20 and get the number of pixels for each group
    groups = []
    for i in range(np.min(unique), np.max(unique), 30):
        groups.append(np.sum((gray >= i) & (gray < i+30)))
    print('The number of pixels for each group: ', groups)

    #if there is a group with more than 50% of the pixels, then it is benin
    ratio = []
    total = 0
    for i in range(len(groups)):
        total += groups[i]

    for i in range(len(groups)):
        ratio.append(groups[i]/total)

    print('The ratio of the pixels for the groups is: ', ratio)

    if max(ratio) > 0.5:
        print('The tumor is benin')
    else:
        #check if there is another group with more than 25% of the pixels
        print('The tumor is probably malign')

        
def main():
    color('data/both.jpg')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()