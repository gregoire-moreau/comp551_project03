import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

losses = pd.read_csv('losses.csv')
acc = pd.read_csv('acc.csv')


plt.plot(acc.loc[:, 'Epoch'], acc.loc[:, 'Loss'], marker='o')
plt.xticks(acc.loc[:, 'Epoch'])
plt.xlabel("Training epoch")
plt.ylabel("Average NLL loss")
# plt.annotate()
for i in range(2):
    plt.annotate(str(round(acc.loc[i, 'Loss'], 3)), (acc.loc[i, 'Epoch'], acc.loc[i, 'Loss']), xytext=(acc.loc[i, 'Epoch'] +0.1, acc.loc[i, 'Loss']))
for i in range(2, 9):
    plt.annotate(str(round(acc.loc[i, 'Loss'], 3)), (acc.loc[i, 'Epoch'], acc.loc[i, 'Loss']), xytext=(acc.loc[i, 'Epoch']-0.15, acc.loc[i, 'Loss']+0.003))
for i in range(9, 10):
    plt.annotate(str(round(acc.loc[i, 'Loss'], 3)), (acc.loc[i, 'Epoch'], acc.loc[i, 'Loss']), xytext=(acc.loc[i, 'Epoch']-0.3, acc.loc[i, 'Loss']+0.003))

plt.autoscale()
plt.show()
