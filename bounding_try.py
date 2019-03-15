import cv2
import pandas as pd
import matplotlib.pyplot as plt
from bounding_boxes import biggest_box
import numpy as np

train_images = pd.read_pickle('./input/train_images.pkl')
train_labels = pd.read_csv('./input/train_labels.csv')



img_idx = 500


plt.title('Label: {}'.format(train_labels.iloc[img_idx]['Category']))
plt.imshow(train_images[img_idx])
plt.show()

plt.title('box')
plt.imshow(biggest_box(train_images[img_idx]))
plt.show()


