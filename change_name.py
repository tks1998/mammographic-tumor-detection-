import os
import cv2

main_path = "../DATA TRAIN/MASK/"

for folder in os.listdir(main_path):
    path_img = os.path.join(main_path,folder)
    print(path_img)
    for img in os.listdir(path_img):
        
        path_image = os.path.join(path_img,img)
        
        name = folder+"_"+img 
        
        # print(name)
        
        path_new = os.path.join(path_img,name)
        
        os.rename(path_image,path_new)

    

        
