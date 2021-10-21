import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import main
from scipy.interpolate import make_interp_spline
import find_k

def get_pedigree_outcome():
    plt.scatter(main.pos_outcome['Outcome'], main.pos_outcome['DiabetesPedigreeFunction'], color='r', alpha=0.7)
    plt.scatter(main.neg_outcome['Outcome'], main.neg_outcome['DiabetesPedigreeFunction'], color='black', alpha=0.7)
    plt.xlabel("Age")
    plt.ylabel("Diabetes Pedigre")
    plt.title("Positive and Negative diabetes in terms of age and pedigre")
    plt.show()


def get_k_graph():
    X_Y_Spline = make_interp_spline(find_k.k_values, find_k.score_stored)
    X_ = np.linspace(min(find_k._values), max(find_k.k_values), 100)
    Y_ = X_Y_Spline(X_)
    plt.plot(X_, Y_)
    plt.xlabel("Values of K")
    plt.ylabel("Score get for specific K")
    plt.title("Finding correct K for clustering")
    plt.show()


def get_age_outcome():
    plt.scatter(main.pos_outcome['Age'], main.pos_outcome['Insulin'], color='r')
    plt.scatter(main.neg_outcome['Age'], main.neg_outcome['Insulin'], color='black')
    plt.xlabel("Age")
    plt.ylabel("Insulin")
    plt.title("Positive and Negative diabetes in terms of age and insulin")
    plt.show()

# get_pedigree_outcome()
# get_age_outcome()



# plt.clf()
# fig = plt.figure()
# a1 = fig.add_subplot(1,1,1)
# a1.plot(k_values, score_stored)
# a1.set_title("Finding correct K for clustering")
# a1.set_ylim(0,1)
# plt.show()