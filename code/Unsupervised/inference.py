from pathlib import Path
from rembg import remove, new_session
from PIL import Image
import io
from datetime import datetime
import numpy as np
from timeit import default_timer as timer

session = new_session()
def replace_color(pixel):
    pixel = tuple(pixel)
    # Check if the pixel is colored
    if pixel[3] != 0 and pixel[0] != pixel[1] or pixel[0] != pixel[2]:
        return (255, 255, 255, 255)  # White
    else:
        return pixel

c=0
times=[]
for file in Path('<path>/value_validation_resized').glob('*.jpg'):
    input_path = str(file)
    output_path = "./"+  str(file.stem) + ".out.png"

    with open(input_path, 'rb') as i:
       
        #with open(output_path, 'wb') as o:
        input = i.read()
         
        start = timer()
        output = remove(input, session=session)
        end = timer()
        times.append(end-start)
        #o.write(output)
        image = Image.open(io.BytesIO(output))
        image = image.convert('L')
        print(np.mean(times))
        # Threshold the image to convert it to black and white
        threshold_value = 2  # Adjust this value to control the threshold
        image = image.point(lambda p: 255 if p > threshold_value else 0, mode='1')
        #image.save('/home/mmazzamuto/UNet/inference/'+input_path.split('/')[-1])
        c+=1
        