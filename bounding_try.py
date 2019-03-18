import cv2
import pandas as pd
import matplotlib.pyplot as plt
from bounding_boxes import biggest_box
import numpy as np

train_images = pd.read_pickle('./input/train_images.pkl')
train_labels = pd.read_csv('./input/train_labels.csv')

def box_size2(contours):
    (min_x, min_y, max_x, max_y) = contours
    return (max_x-min_x+1) * (max_y-min_y+1)

img_idx = 0
errors = 0

plt.title('Label: {}'.format(train_labels.iloc[img_idx]['Category']))
plt.imshow(biggest_box(train_images[img_idx]))
plt.show()

while True:
    print("Is the label correct? (y/n)")
    if (input()=='n'):
        errors +=1
        print(img_idx)
        plt.title('Label: {}'.format(train_labels.iloc[img_idx]['Category']))
        plt.imshow(train_images[img_idx])
        plt.show()
    print("Press s to stop, any other key to continue")
    if (input()=='s'):
        break
    img_idx +=1
    plt.title('Label: {}'.format(train_labels.iloc[img_idx]['Category']))
    plt.imshow(biggest_box(train_images[img_idx]))
    plt.show()

print(img_idx, errors, "accuracy = ", 1- (errors/(img_idx+1)))


