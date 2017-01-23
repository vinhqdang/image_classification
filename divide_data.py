"""
    Dividing the images directory to tomato and non_tomato
"""

# ID of tomato image in JSON files
TOMATO_ID = "939030726152341c154ba28629341da6_train"

import json
from pprint import pprint
from shutil import copyfile

with open ('train_database.txt') as train_data:
    train_json = json.load (train_data)

with open ('test_database.txt') as test_data:
    test_json = json.load (test_data)

# process train data
for i in range (len (train_json)):
    print ("Processing file " + str(i+1))
    cur_image = train_json[i]
    file_name = cur_image['name']
    boxes = cur_image['boxes']
    has_tomato = False
    for j in range (len (boxes)):
        cur_box = boxes[0]
        box_id = cur_box['id']
        if box_id == TOMATO_ID:
            has_tomato = True
            break
    src_file = "./Images/Train/" + file_name
    dst_file = "./problem2/Train/non_tomato/" + file_name
    if has_tomato:
        dst_file = "./problem2/Train/tomato/" + file_name
    copyfile (src_file, dst_file)

# process test data
for i in range (len (test_json)):
    print ("Processing file " + str(i+1))
    cur_image = test_json[i]
    file_name = cur_image['name']
    boxes = cur_image['boxes']
    has_tomato = False
    for j in range (len (boxes)):
        cur_box = boxes[0]
        box_id = cur_box['id']
        if box_id == TOMATO_ID:
            has_tomato = True
            break
    src_file = "./Images/Test/" + file_name
    dst_file = "./problem2/Test/non_tomato/" + file_name
    if has_tomato:
        dst_file = "./problem2/Test/tomato/" + file_name
    copyfile (src_file, dst_file)
