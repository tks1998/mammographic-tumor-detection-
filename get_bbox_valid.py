"""
    This program find bbox of valid image
"""
import os
from PIL import Image,ImageDraw
import numpy as np


path_valid = "../DATA TRAIN/MASK/VALID/"

for folder_name in os.listdir(path_valid):

    path_folder = os.path.join(path_valid,folder_name)
    
    for img in os.listdir(path_folder):
        
        path_img = os.path.join(path_folder,img)

        mask = Image.open(path_img)
        
        mask_img = mask.copy()

        mask = np.array(mask)
        
        obj_ids = np.unique(mask)
        
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]
        
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        draw = ImageDraw.Draw(mask_img)

        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline ="red")
        mask_img.save("test.jpg")
        break
    break

