import numpy as np
import cv2
from math import floor, ceil

def biggest_box(img):
    contours = find_contours(img)
    max_size = 0
    for x in contours:
        size = box_size(x)
        if size > max_size:
            max_size = size
            biggest = x
    to_ret = np.zeros((biggest[2] - biggest[0] + 1, biggest[3] - biggest[1] + 1), dtype=np.float)
    for i in range(biggest[2] - biggest[0] + 1):
        for j in range(biggest[3] - biggest[1] + 1):
            to_ret[i][j] = 1 if img[biggest[0] + i][biggest[1] + j] >220 else 0
    h = len(to_ret)
    w = len(to_ret[0])
    if h <= 28 and w <= 28:
        to_ret = cv2.copyMakeBorder(to_ret,floor((28-h)/2),ceil((28-h)/2),floor((28-w)/2),ceil((28-w)/2),cv2.BORDER_CONSTANT,value=0)
        to_ret = cv2.resize(to_ret, (28, 28))
    else :
        to_ret = cv2.resize(to_ret, (28, 28))
    return to_ret


def box_size(contours):
    (min_x, min_y, max_x, max_y) = contours
    return (max_x-min_x+1) if (max_x-min_x+1) > (max_y-min_y+1) else (max_y-min_y+1)

def find_contours(img):
    tab = []
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] > 220:
                tab.append((i, j))
    results = []
    while len(tab) > 0 :
        results.append(exploit(tab))
    return results

def exploit(tab):
    (i, j) = tab[0]
    min_x = i
    min_y = j
    max_x = i
    max_y = j
    queue = [(i,j)]
    while len(queue) > 0:
        (i, j) = queue[0]
        queue.remove((i, j))
        if (i, j) in tab:
            if i < min_x :
                min_x = i
            if i > max_x :
                max_x = i
            if j < min_y :
                min_y = j
            if j > max_y :
                max_y = j
            tab.remove((i, j))
            queue_add(queue, i, j)
    return (min_x, min_y, max_x, max_y)

def queue_add(arr, i, j):
    arr.append((i-1, j))
    arr.append((i+1, j))
    arr.append((i, j -1))
    arr.append((i, j+1))
