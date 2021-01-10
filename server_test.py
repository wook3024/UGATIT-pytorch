import requests 
import numpy as np
import cv2
import os
import random
import struct
import matplotlib.pyplot as plt
from PIL import Image
from munch import  Munch


# 01043, 01125, 01543, 01619, 01904, 01916, 32379 
# 02603, 29734, 29879, 30032, 30325, 31841
def munch_server_value(ip_address, 
                         port, 
                         image_extentions, 
                         root_dir, 
                         image_name, 
                         shuffle_images, 
                         control_IN_LN_ratio, 
                         control_style,
                         multi_inference):
    server_args = Munch()
    server_args.ip_address = ip_address
    server_args.port = port
    server_args.image_extentions = image_extentions
    server_args.root_dir = root_dir
    server_args.image_name = image_name
    server_args.shuffle_images = shuffle_images
    server_args.control_IN_LN_ratio = control_IN_LN_ratio
    server_args.control_style = control_style
    server_args.multi_inference = multi_inference
    
    return server_args

def extension_check(file_name, server_args):
    file_name_lower = file_name.lower()
    possible = any(file_name_lower.endswith(image_extention) for image_extention in server_args.image_extentions)

    return possible

def make_dataset(root_dir, server_args):
    image_dirs = []
    for root, _, files in sorted(os.walk(root_dir)):
        for file_name in sorted(files):
            if extension_check(file_name, server_args):
                image_dir = os.path.join(root, file_name)
                image_dirs.append(image_dir)

    return image_dirs

def get_original_image_arr(server_args, image_dir, target_dir):
    original_image = Image.open(target_dir if server_args.control_IN_LN_ratio or server_args.control_style else image_dir).convert('RGB')
    original_image_arr = np.array(original_image, dtype=np.uint8)
    original_image_arr = cv2.resize(original_image_arr, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    
    return original_image_arr

def request_translation(server_args, image_dir, ratio_control_value, style_control_value, target_dir):
    resp = requests.post(f"http://{server_args.ip_address}:{server_args.port}/predict", files={
        "image": open(target_dir if server_args.control_IN_LN_ratio or server_args.control_style else image_dir, 'rb'), 
        "ratio_control_value": struct.pack('f', ratio_control_value), 
        "style_control_value": struct.pack('f', style_control_value)})
    translation_image_arr = np.array(resp.json()['image'], dtype=np.uint8)

    return translation_image_arr

def concat_image(original_image_arr, translation_image_arr, concat_image_set):
    concat_image = np.concatenate((original_image_arr, translation_image_arr), axis=0)
    if concat_image_set is not None:
        concat_image_set = np.concatenate((concat_image_set, concat_image), axis=1)
    else:
        concat_image_set = concat_image
    
    return concat_image_set
    
def show_multi_image(concat_image_set):
    height = np.shape(concat_image_set)[1]
    concat_image_set = np.concatenate((concat_image_set[:, :height//2, :], concat_image_set[:, height//2:, :]), axis=0) # separate to 2 rows
    plt.imshow(concat_image_set)
    plt.show()

def reset_value(image_dir):
    ratio_control_value, style_control_value = -1.0, 0.0
    target_dir = image_dir
    concat_image_set = None

    return ratio_control_value, style_control_value, target_dir, concat_image_set

def translation_image(ip_address='192.168.154.29',
        port='8000',
        image_extentions=['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif'],
        root_dir='./dataset/FFHQ_AnimeFaces256Cleaner/testA/',
        image_name='32379.png',
        shuffle_images=False,
        control_IN_LN_ratio=False,
        control_style=False,
        multi_inference=True):
    server_args = munch_server_value(ip_address,
                        port,
                        image_extentions,
                        root_dir,
                        image_name,
                        shuffle_images,
                        control_IN_LN_ratio,
                        control_style,
                        multi_inference)

    image_dirs = make_dataset(server_args.root_dir, server_args)
    target_dir = os.path.join(server_args.root_dir, server_args.image_name)
    if server_args.shuffle_images:
        random.shuffle(image_dirs)

    start_iteration(server_args, image_dirs, target_dir)

def start_iteration(server_args, image_dirs, target_dir):
    concat_image_set = None
    for index, image_dir in enumerate(image_dirs):
        try:
            ratio_control_value += 0.1
            style_control_value += 0.1
        except:
            ratio_control_value = -1.0
            style_control_value = 0.0
        
        if not server_args.control_IN_LN_ratio: ratio_control_value = 0.0
        if not server_args.control_style: style_control_value = 1.0

        original_image_arr = get_original_image_arr(server_args, image_dir, target_dir)
        translation_image_arr = request_translation(server_args, image_dir, ratio_control_value, style_control_value, target_dir)
        # translation_image = Image.fromarray(translation_image_arr)

        concat_image_set = concat_image(original_image_arr, translation_image_arr, concat_image_set)

        if (index+1) % 20 == 0:
            show_multi_image(concat_image_set)
            ratio_control_value, style_control_value, target_dir, concat_image_set = reset_value(image_dir)
        
        if not server_args.multi_inference:
            plt.imshow(concat_image_set)
            plt.show()
            break


translation_image(ip_address='192.168.154.29',
                  port='8000',
                  image_extentions=['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif'],
                  root_dir='./dataset/FFHQ_AnimeFaces256Cleaner/testA/',
                  image_name='32379.png',
                  shuffle_images=False,
                  control_IN_LN_ratio=False,
                  control_style=False,
                  multi_inference=True)