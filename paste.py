import os
import cv2
import numpy as np
from glob import glob

bg_paths = glob('bgs/*')
char_paths = glob('AlpChars/*') + glob('ProChars/*')

for char_path in char_paths:
    char_img = cv2.imread(char_path, -1)
    char = os.path.basename(char_path).split('.')[0]
    save_dir = os.path.join('paste_img', char)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    h, w, _ = char_img.shape
    for bg_path in bg_paths:
        color = 0
        if 'blue' in bg_path or 'black' in bg_path:
            color = 255
        bg_img = cv2.imread(bg_path, -1)
        bg_img = cv2.resize(bg_img, (w, h))

        y, x = np.where(char_img[:, :, 3] > 200)

        bg_img[y, x] = color
        img_name = os.path.basename(bg_path).split('.')[0] + '.jpg'
        cv2.imwrite(os.path.join(save_dir, img_name), bg_img)
