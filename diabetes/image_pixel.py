from numpy import asarray
from PIL import Image
import pandas as pd
import numpy as np
import csv
from sklearn.datasets import load_breast_cancer

csv_file = "./dataset/pneumonie_img_norm.csv"

image = Image.open('./image_pneumonie/xray_pneumonie.jpeg')
pixels = asarray(image)
pixels = pixels.astype('float32')
pixels /= 255.0
data_pixel = pixels[0]
# df = pd.DataFrame(data_pixel)
# df.to_csv(csv_file)
# print(data_pixel)
cancer_data = load_breast_cancer()

array = np.array(data_pixel)
array.resize((30, 1))
print(array)

cancer_array = np.array(cancer_data.data)
cancer_array.resize((30, 96))
print(cancer_array)
# print(data_pixel[0])
# print(len(data_pixel))
# print(cancer_data.data[1])
# print(len(cancer_data.data))

# from 705 to 30pixels (need convolutonal matrices)