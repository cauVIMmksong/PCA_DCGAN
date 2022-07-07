import os
import numpy as np
import glob
from matplotlib import pyplot as plt
from PIL import Image as im


#dataset_path = 'C:\\Users\\User\\Downloads\\Face-Recognition-using-eigen-faces-master\\Face-Recognition-using-eigen-faces-master\\img_align_celeba\\img_align_celeba'
dataset_path = 'celeba/img_align_celeba'
file_names = os.listdir(dataset_path)
file_names
i = 1 
for name in file_names:
    src = os.path.join(dataset_path, name)
    dst = 'image'+str(i) + '.jpg'
    dst = os.path.join(dataset_path, dst)
    os.rename(src,dst)
    i+=1
