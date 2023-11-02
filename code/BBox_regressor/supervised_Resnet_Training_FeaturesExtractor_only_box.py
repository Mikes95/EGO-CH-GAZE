


import torch
from torchvision.models import resnet50, resnet18, resnet152
from torchvision.models.feature_extraction import create_feature_extractor
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
from itertools import chain
import numpy as np
from PIL import Image,ImageEnhance
from math import e
import random
from collections import OrderedDict
import torch.nn as nn
import io
from scipy.ndimage import label, generate_binary_structure
import cv2
import datetime
from scipy import stats
import pandas as pd 
import os
from PIL import Image
from scipy.stats import norm
import matplotlib.patches as patches
import json
import wandb
from skimage.measure import  regionprops
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
from torchvision import transforms
from numpy import sqrt 
import math
import torch.optim as optim
import copy
from datetime import datetime  
from my_function2 import *

def my_logging(msg):
    redf.write(str(datetime.fromtimestamp(datetime.now().timestamp())) +"\t"+ str(msg)+"\n")
    redf.flush()

def convertTuple(tup):
    str = ' '.join(tup)
    return str
eps=1e-10
input_resolution_x=2272
input_resolution_y=1278
redf = open('<path>\supervised_Resnet_Training_FeaturesExtractor_Conv2D_only_box.txt', 'w', buffering=1)
redf.write("\n\n\n\n")
my_logging("_______________________________________________________________________")

X = 160*10 #input_resolution_x
Y = 90*10 #input_resolution_y

transform = transforms.Compose(
    [#transforms.ToPILImage(),
    transforms.Resize((Y, X)),
    transforms.ToTensor(),
])


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
all_frames_path = "<path>/all_frames/"
files = os.listdir(all_frames_path)
with open('<path>/train_video_gaze_inside.txt', 'r') as f:
            train_video_gaze_inside = json.loads(f.read())
random_seed=12345
random.seed=random_seed
with open('<path>/frame_list_new.json', 'r') as f:
            frame_list_new = json.loads(f.read())



paths,label,gazes_x,gazes_y,boxes = [],[],[],[],[]
good_frames=0

random.shuffle(train_video_gaze_inside)
for element in train_video_gaze_inside: 
    
    current_video = element.split('_')[0]
    current_id = element.split('_')[1]
    current_frame = int(element.split('_')[-1].replace('.jpg',''))
    current_video_max_frame = int((video_lenght_dict["video_"+element.split('_')[0]]))
    if( int(frame_list_new[element]['looking_at']) <15):   
            #if (int(frame_list_new[element]['looking_at'])  in frame_list_new[element]['object_list']):
            
            if  (frame_list_new[element]['looking_at'] in frame_list_new[element]['object_list']):
                
                bb_index =frame_list_new[element]['object_list'].index(frame_list_new[element]['looking_at'])
                looked_at_box = frame_list_new[element]['boxes'][bb_index]

                x = looked_at_box[0]
                y = looked_at_box[1]
                w = looked_at_box[2]
                h = looked_at_box[3]

                looked_at_box[0]=x
                looked_at_box[1]=y
                looked_at_box[2]=w
                looked_at_box[3]=h

            gaze_x = int(frame_list_new[element]['gaze_x'])
            gaze_y = int(frame_list_new[element]['gaze_y'])
            if(gaze_x >0 and gaze_x < input_resolution_x):
              if(gaze_y >0 and gaze_y < input_resolution_y):
                good_frames+=1
                paths.append([all_frames_path+element+'.jpg'])
                label.append([int(frame_list_new[element]['looking_at'])])
                gazes_x.append(gaze_x)
                gazes_y.append(gaze_y)
                looked_at_box = torch.Tensor(looked_at_box)
                boxes.append(looked_at_box)

print("good_frames:",good_frames)
my_logging("good_frames: " + str(good_frames))

def train_triplet(optimizer, resnet_feature_extractor,std_estimation_module,gaussian_offset_estimation,gaussian_module,train_loader, test_loader, exp_name='experiment', lr=1, epochs=1, momentum=0.99, margin=1, logdir='logs'): 
    criterion_box = nn.MSELoss()
    
    criterion_box.to(device)
    
    global_step = 0 
    mode='train'
    
    for e in range(epochs): 
        print('TRAIN', e)
        correct = 0
        total=0
        with torch.set_grad_enabled(mode=='train'): #abilitiamo i gradienti solo in training 
            for i, batch in enumerate(train_loader): 
                optimizer.zero_grad()
                original,g_x_o,g_y_o,labels,bb,name_original,i= [b for b in batch]

                original = original.to(device)
                #FEATURES EXTRACTED FROM RESNET
                outputs_original = resnet_feature_extractor(original)['layer4']
                #GAZE SCALED IN FEATURE SPACE 
                g_x_o_scaled = ((g_x_o/input_resolution_x)* outputs_original.shape[3]).to(device)
                g_y_o_scaled = ((g_y_o/input_resolution_y)* outputs_original.shape[2]).to(device)
                #STD ESTIMATION FROM FEATURES 
                std_estimate_original = std_estimation_module(outputs_original)
                #OFFSET ESTIMATION FROM FEATURES
                gaussian_offset_original = gaussian_offset_estimation(outputs_original)
                #GAUSSIAN ESTIMATION USING STD AND OFFSET
                features_multiplead_by_gaussian, gaussian_for_original,gaussian_boxes,offset_2,std_2 = gaussian_module(outputs_original,std_estimate_original,gaussian_offset_original,g_x_o_scaled,g_y_o_scaled)
                my_logging("Frame "+"\t" +convertTuple(name_original))
                for name, param in resnet_feature_extractor.named_parameters():
                    #print(name)
                    if param.requires_grad:
                        my_logging("resnet_feature_extractor " +name +" "+str(param.data.sum().item()))
                        #print(name, param.data.sum())
                        break
                        
                for name, param in gaussian_offset_estimation.named_parameters():
                    #print(name)
                    if param.requires_grad:
                        my_logging("gaussian_offset_estimation " +name +" "+str(param.data.sum().item()))
                        #print(name, param.data.sum())
                        break
                for name, param in std_estimation_module.named_parameters():
                    #print(name)
                    if param.requires_grad:
                        my_logging("std_estimation_module " +name +" "+str(param.data.sum().item()))
                        #print(name, param.data.sum())
                        break
                gaussian_boxes[:,0]=gaussian_boxes[:,0]/features_multiplead_by_gaussian.shape[3]
                gaussian_boxes[:,1]=gaussian_boxes[:,1]/features_multiplead_by_gaussian.shape[2]
                gaussian_boxes[:,2]=gaussian_boxes[:,2]/features_multiplead_by_gaussian.shape[3]
                gaussian_boxes[:,3]=gaussian_boxes[:,3]/features_multiplead_by_gaussian.shape[2]
                offset_2[:,0] =offset_2[:,0]/features_multiplead_by_gaussian.shape[3]
                offset_2[:,1] =offset_2[:,1]/features_multiplead_by_gaussian.shape[2]
                bb[:,0]=bb[:,0]/input_resolution_x
                bb[:,1]=bb[:,1]/input_resolution_y
                bb[:,2]=bb[:,2]/input_resolution_x
                bb[:,3]=bb[:,3]/input_resolution_y
                labels = labels.squeeze(1)
             
                loss_gaussian = criterion_box(gaussian_boxes,bb.to(device))
                loss_gaussian.backward() 
                optimizer.step() 
                
                        
                n = outputs_original.shape[0] 
                global_step += n 
                examples = []
                gaussian_boxes[:,0]=gaussian_boxes[:,0]*X
                gaussian_boxes[:,1]=gaussian_boxes[:,1]*Y
                gaussian_boxes[:,2]=gaussian_boxes[:,2]*X
                gaussian_boxes[:,3]=gaussian_boxes[:,3]*Y
                bb[:,0]=bb[:,0]*X
                bb[:,1]=bb[:,1]*Y
                bb[:,2]=bb[:,2]*X
                bb[:,3]=bb[:,3]*Y
                offset_2[:,0] =offset_2[:,0]*input_resolution_x
                offset_2[:,1] =offset_2[:,1]*input_resolution_y
                g_x_o=((g_x_o/input_resolution_x)*X).to(torch.int32)
                g_y_o=((g_y_o/input_resolution_y)*Y).to(torch.int32)
                ##  PRINT IMAGE TO WANDB
                
                my_logging("\n")
                print('EPOCH:',e, "STEP:", global_step,'/',len(dataset),"Loss_box:", loss_gaussian.item())       

                
                
                
               
        #SALVARE I PESI DEL CLASSIFICATORE    
        dir_name="<path>"
        if not os.path.exists('<path>/supervised/'+dir_name):
            os.mkdir('<path>/supervised/'+dir_name)
        torch.save(std_estimation_module.state_dict(), '<path>/supervised/'+dir_name+'/'+"std_estimation_module_e_"+str(e)+'_seed_'+str(random_seed)+'.pth')
        torch.save(gaussian_offset_estimation.state_dict(), '<path>/supervised/'+dir_name+'/'+"gaussian_offset_estimation_e_"+str(e)+'_seed_'+str(random_seed)+'.pth')
        torch.save(resnet_feature_extractor.state_dict(), '<path>/supervised/'+dir_name+'/'+"feature_extractor"+str(e)+'_seed_'+str(random_seed)+'.pth')
        
         
    
    return std_estimation_module



batch_size = 8

dataset = Dataset_Supervised(paths,label,gazes_x,gazes_y,boxes,transforms=transform)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model_type = "resnet18"
if(model_type =="resnet18"):
    layer= "layer4"
    model = resnet18(pretrained=True)
    model.to(device)
    return_nodes = {
        "layer4": "layer4"
    }

model.train()
#model.eval()
feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
#feature_extractor.eval()
'''for param in feature_extractor.parameters():
    param.requires_grad = True'''
    



feature_extractor.to(device)

resnet_50 = resnet50()
gaussian_std_estimation = nn.Sequential(copy.deepcopy(resnet_50.layer3),
                        nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.Conv2d(128, 2, kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.LeakyReLU(0.1),
                       ).train().to(device)
gaussian_offset_estimation = nn.Sequential(copy.deepcopy(resnet_50.layer3),
                        nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.Conv2d(128, 2, kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                       ).train().to(device)


#gaussian_std_estimation.weight.data.fill_(1.0)
gaussian_module = GaussianModule()

all_param = chain(feature_extractor.parameters(),gaussian_std_estimation.parameters(), gaussian_offset_estimation.parameters())
optim = torch.optim.SGD(all_param, lr=0.001, momentum=0.99)  
#optim = torch.optim.Adam(all_param,lr = 0.001)

train_triplet(optim, feature_extractor,gaussian_std_estimation,gaussian_offset_estimation,gaussian_module,train_dataloader,train_dataloader,"test",epochs=10)