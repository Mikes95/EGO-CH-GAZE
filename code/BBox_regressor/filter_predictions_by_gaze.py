
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json



with open('<path>/coco_instances_results.json') as f:
  predictions_to_be_filtered = json.load(f)

with open('<path>/frame_list_new.json') as f:
  annotations = json.load(f)
c=0
filtering_dict={}
fileterd=[]
image_id_array = []
for element in predictions_to_be_filtered:
  print(c,'/',len(predictions_to_be_filtered))
  

  image_id = (element['image_id'])
  '''  if(int(annotations[image_id]["looking_at"])<7):
    element['bbox'][0] = (element['bbox'])[0]-290
    element['bbox'][1] = (element['bbox'])[1]-290
    element['bbox'][2] = (element['bbox'])[2]+290
    element['bbox'][3] = (element['bbox'])[3]+290'''
  x = (element['bbox'])[0]
  y = (element['bbox'])[1]
  w = (element['bbox'])[2]
  h = (element['bbox'])[3]
  #print(x,y,w,h)
  
  if(image_id not in image_id_array):
    gaze_x = int(annotations[image_id]["gaze_x"])
    gaze_y = int(annotations[image_id]["gaze_y"])
    
    if(gaze_x >x and gaze_x < x+w):
      if(gaze_y >y and gaze_y < y+h):
        fileterd.append(element)
        image_id_array.append(image_id)
        
        
        
  c+=1

  
  
  
    

with open('<path>/coco_instances_results_gaze_filtered.json', 'w') as outfile:
    json.dump(fileterd, outfile)