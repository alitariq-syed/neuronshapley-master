# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 15:24:13 2021

@author: Ali
"""
import numpy as np
import os

val_images_path = 'G:/CUB_200_2011/CUB_200_2011/train_test_split/test'

classes = os.listdir(val_images_path)

image_path=[]
for label in classes:
    imgs = os.listdir(os.path.join(val_images_path, label))
    
    for img in imgs:
        image_path.append(label+'/'+img)
        
image_paths = np.array(image_path)
np.savetxt('val_images_CUB.txt',image_paths,fmt='%s')