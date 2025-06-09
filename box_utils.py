import os
import csv 
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from urllib.request import urlopen
import pandas as pd
import pickle
import json 

def draw_box_2(text, boxes,output_folder, img_name):
    width, height = 512, 512
    image = Image.new('RGB', (width, height), 'gray')
    
    draw = ImageDraw.Draw(image)
    for i, bbox in enumerate(boxes):
        for box in bbox:
            if i==0:
                #  [(x0, y0), (x1, y1)] 
                
                draw.rectangle([(box[0] * 512, box[1]* 512),(box[2]* 512, box[3]* 512)], outline='red', width=6)
                
            elif i==1:
                draw.rectangle([(box[0]* 512, box[1]* 512),(box[2]* 512, box[3]* 512)], outline='green', width=6)
            else:
                draw.rectangle([(box[0]* 512, box[1]* 512),(box[2]* 512, box[3]* 512)], outline='blue', width=6)
    image.save(os.path.join(output_folder, img_name))


def draw_box(text, boxes,output_folder, img_name):
    width, height = 512, 512
    image = Image.new('RGB', (width, height), 'gray')
    
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("Roboto-LightItalic.ttf", size=20)
    for i, box in enumerate(boxes):
        t = text[i]
        draw.rectangle([(box[0], box[1]),(box[2], box[3])], outline=128, width=2)
        mean_box_x, mean_box_y = int((box[0] + box[2] )/ 2) + int((box[1] + box[3] )/ 2)
        draw.text((mean_box_x, mean_box_y), t, fill=200,font=font )
    image.save(os.path.join(output_folder, img_name))

def save_img(folder_name, img, prompt, iter_id, img_id,count):
    os.makedirs(folder_name, exist_ok=True)
    img_name = str(count)+'.jpg'
    img.save(os.path.join(folder_name, img_name))


def format_box(names, boxes):
    result_name = []
    resultboxes = []
    for i, name in enumerate(names):
        name = remove_numbers(name)
        result_name.append('a ' + name.replace('_',' '))
        if name == 'person': 
            boxes[i] = boxes[i]
        resultboxes.append([boxes[i]])
    return result_name, np.array(resultboxes)

def remove_numbers(text):
    result = ''.join([char for char in text if not char.isdigit()])
    return result
def process_box_phrase(names, bboxes):
    d = {}
    for i, phrase in enumerate(names):
        #phrase = phrase.replace('_',' ')
        
        list_noun = phrase.split(' ')
        for n in list_noun:
            n = remove_numbers(n)
            if not n in d.keys():
                d.update({n:[np.array(bboxes[i])/512]})
            else:
                d[n].append(np.array(bboxes[i])/512)
    return d

def Pharse2idx_2(prompt, name_box):
    prompt = prompt.replace('.','')
    prompt = prompt.replace(',','')
    prompt_list = prompt.strip('.').split(' ')
    object_positions = []
    bbox_to_self_att = []
    for obj in name_box.keys():
        obj_position = []
        in_prompt = False
        for word in obj.split(' '):
            if word in prompt_list:
                obj_first_index = prompt_list.index(word) + 1
                obj_position.append(obj_first_index)
                in_prompt = True
            elif word +'s' in prompt_list:
                obj_first_index = prompt_list.index(word+'s') + 1
                obj_position.append(obj_first_index)
                in_prompt = True
            elif word +'es' in prompt_list:
                obj_first_index = prompt_list.index(word+'es') + 1
                obj_position.append(obj_first_index)
                in_prompt = True 
        if in_prompt :
            bbox_to_self_att.append(np.array(name_box[obj]))
        
            object_positions.append(obj_position)

    return object_positions, bbox_to_self_att
