import json
import os
import cv2

with open('<path>/frame_list_new.json', 'r') as f:
            frame_list_new = json.loads(f.read())

import os
import numpy as np
from PIL import Image, ImageDraw
from skimage import measure
import matplotlib.pyplot as plt

input_folder = '<path>/inference'  # Replace with the path to the input folder
output_folder = '<path>/exported_bbox'  # Replace with the path to the output folder
rgb_path = "<path>/all_frames"
# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
predictions = []

# Iterate over all files in the input folder
c=0
def get_bounding_box(label):
            indices = np.where(labels == label)
            x_min, x_max = np.min(indices[1]), np.max(indices[1])
            y_min, y_max = np.min(indices[0]), np.max(indices[0])
            return x_min*1.42, y_min*1.42, x_max*1.42, y_max*1.42
lenx= len(os.listdir(input_folder))
for filename in os.listdir(input_folder):
    print(c,lenx)
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Process only image files
        # Load the image and convert it to grayscale
        
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert('L')
        
        threshold = 128  # Adjust the threshold value as needed
        binary_image = np.array(image.point(lambda x: 0 if x < threshold else 1), dtype=np.uint8)

        # Find the connected components in the binary image
        labels = measure.label(binary_image, background=0)

        # Define a function that returns the bounding box of a connected component given its label
        
        gaze_x =int(frame_list_new[filename.replace('.jpg','')]['gaze_x'])
        gaze_y =int(frame_list_new[filename.replace('.jpg','')]['gaze_y'])
        if(gaze_x>0 and gaze_x<2278 and gaze_y>0 and gaze_y<1278):
            gaze_x =int(frame_list_new[filename.replace('.jpg','')]['gaze_x']/1.42)
            gaze_y =int(frame_list_new[filename.replace('.jpg','')]['gaze_y']/1.42)
            if(gaze_x>1600):gaze_x=1600-1
            if(gaze_y>900):gaze_x=900-1
            point = (gaze_y,gaze_x)
            label = labels[point]
            if label > 0:
                #print(filename)
                #print(gaze_x,gaze_y)
                
                bbox = get_bounding_box(label)
          
                prediction = {
                "image_id": filename.replace('.png',''),
                "category_id": frame_list_new[filename.replace('.jpg','')]['looking_at'],
                "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                "score": 1.0
                }
                
                predictions.append(prediction)
                
                
                
            else:
                #print("The point is not inside a white connected component")
                continue
        c+=1
with open("<path>/inference.json", "w") as f:
                f.write(str(predictions))
                    
                