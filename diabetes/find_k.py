import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import main


x_train, x_test, y_train, y_test = train_test_split(main.sample_wt_outcome, main.sample_outcome, 
                                    train_size=0.8, random_state=6)

def distance(db1, db2):
    squared_dif = 0
    for i in range(len(db1)):
        distance = (db2[i] - db1[i]) ** 2
        squared_dif += distance
    final_distance = squared_dif ** 0.5
    return final_distance

def sigmoid(ds):
    list_points = []
    for row in ds.iterrows():
        points = []
        for i in range(1, 7):
            nb = -row[1][i]
            point = (1 + math.exp(nb))
            normalized_point = 1 / point
            points.append(normalized_point)
        list_points.append(points)
    return list_points

def classify(unknown, dataset, labels, k):
    distances = []
    for i in range(len(dataset)):
        dis = distance(unknown, dataset[i])
        distances.append([dis, i])
    distances.sort()
    sample = distances[0:k]
    count_pos = 0
    count_neg = 0
    for item in sample:
        if labels.iloc[item[1]] == 1:
            count_pos += 1
        elif labels.iloc[item[1]] == 0:
            count_neg += 1
    if count_pos > count_neg:
        return 1
    return 0

x_train_sigmoid = sigmoid(x_train)

def percentage_score(x_train_data, y_train_data, k):
    count_pos = 0.0
    l = len(x_train_data)
    for i in range(l):
        result = classify(x_train_data[i], x_train_data, y_train_data, k)
        if result == y_train_data.iloc[i]:
            count_pos += 1
    return count_pos / len(x_train_data)

k_number_test = 70
k_values = [x for x in range(2, k_number_test)]

def get_scored_store():
    li = []
    for k in range(2, k_number_test):
        score = percentage_score(x_train_sigmoid, y_train, k)
        li.append(score)
    return li

score_stored = get_scored_store()
elbow_point = 7

