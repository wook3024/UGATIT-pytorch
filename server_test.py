import requests 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import glob
import random
import struct


# 01043, 01125, 01543, 01619, 01904, 01916, 02603, 29734, 29879, 30032, 30325, 31841, 32379
# ========================= infomation ========================= #
ip_address = '192.168.154.29'
port = '8000'
root_dir = './dataset/FFHQ_AnimeFaces256Cleaner/testA/'
image_name = '01904.png'
target_dir = os.path.join(root_dir, image_name)
image_set_shuffle = True
ratio_control = False
style_control = True


image_set = glob.glob(os.path.join(root_dir, '*.png'))
if image_set_shuffle:
    random.shuffle(image_set)

concat_image_set = None
for index, image_dir in enumerate(image_set):
    try:
        ratio_control_value += 0.1
        style_control_value += 0.1
    except:
        ratio_control_value = -1.0
        style_control_value = 0.0
    
    if not ratio_control: ratio_control_value = 0.0
    if not style_control: style_control_value = 1.0
    # ========================= original ========================= #
    original_image = Image.open(target_dir if ratio_control or style_control else image_dir).convert('RGB')
    original_image_arr = np.array(original_image, dtype=np.uint8)
    original_image_arr = cv2.resize(original_image_arr, dsize=(256, 256), interpolation=cv2.INTER_AREA)

    # ========================= translation ========================= #
    resp = requests.post(f"http://{ip_address}:{port}/predict", files={
        "image": open(target_dir if ratio_control or style_control else image_dir, 'rb'), 
        "ratio_control": struct.pack('f', ratio_control_value), 
        "style_control": struct.pack('f', style_control_value)})
    translation_image_arr = np.array(resp.json()['image'], dtype=np.uint8)
    # translation_image = Image.fromarray(translation_image_arr)

    # ========================= concat and show ========================= #
    concat_image = np.concatenate((original_image_arr, translation_image_arr), axis=0)
    if concat_image_set is not None:
        concat_image_set = np.concatenate((concat_image_set, concat_image), axis=1)
    else:
        concat_image_set = concat_image

    if (index+1) % 20 == 0:
        ratio_control_value = -1.0
        style_control_value = 0.0
        target_dir=image_dir
        height = np.shape(concat_image_set)[1]
        # separate to 2 rows
        if height > 512: concat_image_set = np.concatenate((concat_image_set[:, :height//2, :], concat_image_set[:, height//2:, :]), axis=0)
        plt.imshow(concat_image_set)
        plt.show()
        concat_image_set = None
    
    
# for index, image_dir in enumerate(image_set):
#     # ========================= original ========================= #
#     original_image = Image.open(target_dir if only_one_image else image_dir).convert('RGB')
#     original_image_arr = np.array(original_image, dtype=np.uint8)
#     original_image_arr = cv2.resize(original_image_arr, dsize=(256, 256), interpolation=cv2.INTER_AREA)

#     # ========================= translation ========================= #
#     resp = requests.post(f"http://{ip_address}:{port}/predict", files={"file": open(target_dir if only_one_image else image_dir, 'rb'), "control_value": struct.pack('f', index*0.1-1)})
#     translation_image_arr = np.array(resp.json()['image'], dtype=np.uint8)
#     # translation_image = Image.fromarray(translation_image_arr)

#     # ========================= concat and show ========================= #
#     concat_image = np.concatenate((original_image_arr, translation_image_arr), axis=0)
#     if concat_image_set is not None:
#         concat_image_set = np.concatenate((concat_image_set, concat_image), axis=1)
#     else:
#         concat_image_set = concat_image

#     if index % 20 == 0:
#         height = np.shape(concat_image_set)[1]
#         # separate to 2 rows
#         if height > 512: concat_image_set = np.concatenate((concat_image_set[:, :height//2, :], concat_image_set[:, height//2:, :]), axis=0)
#         plt.imshow(concat_image_set)
#         plt.show()
#         concat_image_set = None
    
#     if only_one_image: break
