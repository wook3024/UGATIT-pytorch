import requests 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

resp = requests.post("http://192.168.154.29:8000/predict", files={"file": open('./dataset/FFHQ_AnimeFaces256Cleaner/testA/00008.png', 'rb')})

img_arr = np.array(resp.json()['image'], dtype=np.uint8)
img = Image.fromarray(img_arr)
# img = Image.open('./00006.png')
plt.imshow(img)
plt.show()

