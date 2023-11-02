
import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import json
import cv2
import urllib.request
import sys
sys.path.append("..")
with open('<path>/frame_list_new.json', 'r') as f:
            frame_list_new = json.loads(f.read())

with open("<path>/validate_coco_new_dataset_gaze_filter.json", 'r') as f:
            validation = json.loads(f.read())
input_folder = '<path>/inference'  # Replace with the path to the input folder
output_folder = '<path>/exported_bbox'  # Replace with the path to the output folder
rgb_path = "<path>/all_frames"
from segment_anything import sam_model_registry, SamPredictor
predictions = []
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

sam_checkpoint = "<path>/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

c=0
lenx= len(validation['images'])
for filename in validation['images']:
    #print(filename['file_name'])
    print(c,lenx)
    c+=1
    image = cv2.imread(rgb_path+"/"+filename['file_name'])
    frame_label = frame_list_new[filename['id']]
    #print(frame_label)
    gaze_x=frame_label['gaze_x']
    gaze_y=frame_label['gaze_y']
    if(gaze_x>0 and gaze_x<2278):
        if(gaze_y>0 and gaze_y<1200):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            input_point = np.array([[gaze_x, gaze_y]])
            input_label = np.array([1])
            predictor.set_image(image)
            masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
            )
            binary_mask = masks[2].squeeze().astype(np.uint8)

            # Find the contours of the mask
            contours, hierarchy = cv2.findContours(binary_mask,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)

            # Get the largest contour based on area
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the new bounding box
            bbox = [int(x) for x in cv2.boundingRect(largest_contour)]
            print(bbox)
            prediction = {
                "image_id": filename['id'],
                "category_id": frame_list_new[filename['id']]['looking_at'],
                "bbox": [bbox[0], bbox[1], bbox[2] , bbox[3]],
                "score": scores[2]
            }
            predictions.append(prediction)
          
             
            
with open("<path>/inference_with_cat.json", "w") as f:
                f.write(str(predictions))