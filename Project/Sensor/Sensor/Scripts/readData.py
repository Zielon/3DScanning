# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

root_dir = "C:/Users/Lukas/Desktop/3DScanning/Project/Sensor/Sensor/Data"

w = 320   
h = 240

f_num = []
files = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]

print (files)

for file in files:
    
    print(file)

    color_buffer = np.fromfile(join(root_dir, file), np.uint8)

    color_buffer = color_buffer.reshape((h,w,3))
        
    plt.figure()
    plt.subplot(131)
    plt.imshow(color_buffer[:,:,0:3])
    plt.show()