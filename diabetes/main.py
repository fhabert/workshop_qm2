import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

csv_file = "./dataset/data_diabetes.csv"
dataset = pd.read_csv(csv_file, delimiter=";")
df = pd.DataFrame(dataset)

sample_points = df[0:50]
pos_outcome = sample_points[sample_points["Outcome"] == 1]
neg_outcome = sample_points[sample_points["Outcome"] == 0]

columns_name = df.columns.tolist()
sample_wt_outcome = sample_points[columns_name[:-1]]
sample_outcome = sample_points["Outcome"]

def normalize(ds):
    list_points1 = []
    min_point = min(ds)
    max_point = max(ds)
    for i in range(len(ds)):
        points1 = (ds.iloc[i] - min_point) / (max_point - min_point)
        normalized_point1 = 1 / points1
        list_points1.append(normalized_point1)
    return list_points1

def standard_info(datapoints):
    print(datapoints.head())
    print(datapoints.describe())

# standard_info(df)


