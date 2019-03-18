import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("./input"))

# Any results you write to the current directory are saved as output

train_images = pd.read_pickle('./input/test_images.pkl')
# train_labels = pd.read_csv('./input/train_labels.csv')

print(train_images.shape)
#print(train_labels.shape)
#print(train_labels[0:20])

img_idx = 0
# plt.title('Label: {}'.format(train_labels.iloc[img_idx]['Category']))
plt.imshow(train_images[img_idx])
plt.show()
