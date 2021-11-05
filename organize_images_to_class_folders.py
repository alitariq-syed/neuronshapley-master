# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:23:48 2021

@author: Ali

Move all imagenet2012 validation set images to their respective class folders
"""


import os
import pandas as pd
from PIL import Image

imageNet_input_path = 'G:/ILSVRC2012_img_val/'
imageNet_output_path = 'G:/ILSVRC2012_img_val_classes/'

input_data = os.listdir(imageNet_input_path)
size = (299,299)
val_images = open('val_images.txt', "r")
count=1
for x in val_images:
  print(count)
  val_class, val_img = x.split('/')
  
  #copy or move the val_img to val_class/val_img
  input_path = imageNet_input_path+val_img[:-1]#to remove \n 
  output_dir = imageNet_output_path+val_class+'/'
  
  #read image and resize it
  img = Image.open(input_path)
  img = img.resize(size)
  
  os.makedirs(os.path.dirname(output_dir), exist_ok=True)
  output_path = output_dir+val_img[:-1]
  img.save(output_path)
  count+=1
  
val_images.close()