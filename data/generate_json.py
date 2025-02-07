import numpy as np
import os
import json

'''
for HL_MF_set
images are grouped: '_0.png', '_1.png', '_2.png'
'''

WIDTH = 1280
HEIGHT = 720
folder_name = "250207_bare_hand/"
imgs = sorted(os.listdir(folder_name + "images/"))

images_list = []
annot_list = []
imgcnt = 0
for imgname in imgs:
    if (int(imgname[-5]) == 0):
        img_dict = {}
        img_dict["id"] = imgcnt
        img_dict["file_name"] = "images/" + imgname
        img_dict["width"] = WIDTH
        img_dict["height"] = HEIGHT
        images_list.append(img_dict)

        annot_dict = {}
        annot_dict["id"] = imgcnt
        annot_dict["image_id"] = imgcnt
        annot_list.append(annot_dict)

        imgcnt += 1

print("image_list num: ", len(images_list))
print("annot_list num: ", len(annot_list))
json_dict = {}
json_dict["images"] = images_list
json_dict["annotations"] = annot_list

if not os.path.exists(folder_name + "annotations"):
    os.makedirs(folder_name + "annotations")
    
with open(folder_name + "annotations/HL_test_data.json", "w") as f:
    json.dump(json_dict, f)
    