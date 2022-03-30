import os
import imageio
from PIL import Image
from imgaug import BoundingBoxesOnImage

original_images_path = 'ORIGINAL/'
resized_images_path = 'resized_images/'
new_width = 416
new_height = 416
for image_file in os.listdir(original_images_path):
    ima = Image.open(original_images_path + image_file)
    new_ima = ima.resize((new_width, new_height), Image.ANTIALIAS)
    new_ima.save(resized_images_path + image_file)

import os
import shutil
import random

images_path = 'resized_images/'
labels_path = 'labels/'
train_path = 'dataset/train/'
validation_path = 'dataset/Validation/'

for image_file in os.listdir(images_path):
    labels_file = image_file.replace('.jpg', '.xml')
    if random.uniform(0, 1) > 0.2:
        shutil.copy(images_path + image_file, train_path + 'images/'
                    + image_file)
        shutil.copy(labels_path + labels_file, train_path +
                    'annotations/' + labels_file)
    else:
        shutil.copy(images_path + image_file, validation_path +
                    'images/' + image_file)
        shutil.copy(labels_path + labels_file, validation_path +
                    'annotations/' + labels_file)


import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
annotations = glob('labels/*.xml')
df = []
cnt = 0
for file in annotations:
    filename = file.split('\\')[-1]
    filename =filename.split('.')[0] + '.jpg'
    row = []
    parsedXML = ET.parse(file)
    for node in parsedXML.getroot().iter('object'):
        blood_cells = node.find('name').text
        xmin = int(node.find('bndbox/xmin').text)
        xmax = int(node.find('bndbox/xmax').text)
        ymin = int(node.find('bndbox/ymin').text)
        ymax = int(node.find('bndbox/ymax').text)

        row = [filename, blood_cells, xmin, xmax, ymin, ymax]
        df.append(row)
        cnt += 1

data = pd.DataFrame(df, columns=['filename', 'cell_type', 'xmin', 'xmax', 'ymin', 'ymax'])

data[['filename', 'cell_type', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv('test.csv', index=False)

from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="dataset")
trainer.setTrainConfig(object_names_array=["sicklecell", "redbloodcell"], batch_size=2, num_experiments=20,
                       train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()
