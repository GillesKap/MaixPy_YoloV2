import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

"""
Convert an image to an array of bytes in a .c file 
source: https://github.com/kendryte/nncase/blob/master/examples/20classes_yolo/k210/kpu_20classes_example/img2c.py
"""

target_shape = (224, 224)
if __name__ == '__main__':
    img = plt.imread(sys.argv[1])
    # img = cv2.imread(sys.argv[1])
    img = cv2.resize(img, target_shape)
    cv2.imwrite("c_img.png", img)
    print(img.shape)
    # img = np.transpose(img, [2, 0, 1])
    with open('image.c', 'w') as f:
        print('const unsigned char gImage_image[]  __attribute__((aligned(128))) ={', file=f)
        print(', '.join([str(i) for i in img.flatten()]), file=f)
        print('};', file=f)
