################
# Script to make data objects starting from raw excel files
# uses scripts Session.py, which uses Trial.py
################


import os
import pandas as pd
import numpy as np
from Session import SessionClass
#from archive import Combined_Plots
import TrajectoryAnalysis
import pickle
import re

import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')


path = r"your path"
os.chdir(path)


separate_2 = []
separate_3 = []
separate_4 = []
combined_2 = []
combined_3 = []
combined_4 = []
Pucktrial_accuracies = np.zeros((1,3))
counter = np.zeros((1,3))

Data_dictionary = dict([])


CirclesPositions = np.array([[-4, 4], [4, 4], [-4, -4], [4, -4]])

def DefineCircles(correct, ax):
    circleDiams = [12,13,14,15]
    for circle in range(4):
        if circle == correct:
            color = 'g'
            radius = 1.2
        else:
            color = 'r'
            radius = 1.2
        circle1 = plt.Circle((CirclesPositions[circle]), radius, color=color, fill=False)
        ax.add_patch(circle1)

class _type:
    def __init__(self):
        return

class _size:
    def __init__(self):
        return

def GenerateDataFrame():
    group_sizes = dict([])
    for i in range(3):
        group_sizes[i + 2] = _size()
        trial_types = dict([])
        group_sizes[i + 2] = trial_types
        for j in range(2):
            trial_types[j + 1] = _type()
            trial_types[j + 1].correct_high_conf = np.zeros((3000,))
            trial_types[j + 1].correct_low_conf = np.zeros((3000,))
            trial_types[j + 1].wrong_high_conf = np.zeros((3000,))
            trial_types[j + 1].wrong_low_conf = np.zeros((3000,))
    return group_sizes

def GenerateDataFrame_nan():
    group_sizes = dict([])
    for i in range(3):
        group_sizes[i + 2] = _size()
        trial_types = dict([])
        group_sizes[i + 2] = trial_types
        for j in range(2):
            trial_types[j + 1] = _type()
            trial_types[j + 1].correct_high_conf = np.zeros((3000,))
            trial_types[j + 1].correct_high_conf[:] = np.nan
            trial_types[j + 1].correct_low_conf = np.zeros((3000,))
            trial_types[j + 1].correct_low_conf[:] = np.nan
            trial_types[j + 1].wrong_high_conf = np.zeros((3000,))
            trial_types[j + 1].wrong_high_conf[:] = np.nan
            trial_types[j + 1].wrong_low_conf = np.zeros((3000,))
            trial_types[j + 1].wrong_low_conf[:] = np.nan
    return group_sizes

group_sizes = GenerateDataFrame_nan()


def create_data(path):
    game = 1
    group_sizes = GenerateDataFrame()
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith((".csv")):
                print(int(re.findall(r'\d+', name.split("_")[0])[0]))
                newpath = os.path.join(root, name)
                df = pd.read_csv(os.path.abspath(os.path.join(root, name)), sep='delimiter', header=None,
                                 engine='python')
                print(os.path.abspath(os.path.join(root, name)))
                data = df[0]
                session = SessionClass(game)
                session.ParseSession(data)
                print(session.PlayerSize)
                session.Session_statistics()
                session.PlotTrialResults()
                if session.PlayerSize == 2:
                    separate_2.append(np.mean(session.Trial_accuracies[0,:]))
                    combined_2.append(np.mean(session.PuckTrial_accuracies[0,:]))

                if session.PlayerSize == 3:
                    separate_3.append(np.mean(session.Trial_accuracies[0,:]))
                    combined_3.append(np.mean(session.PuckTrial_accuracies[0,:]))

                if session.PlayerSize == 4:
                    separate_4.append(np.mean(session.Trial_accuracies[0, :]))
                    combined_4.append(np.mean(session.PuckTrial_accuracies[0,:]))
                Data_dictionary[game-1] = session


                game += 1
    return #group_sizes



create_data(path)


with open(r"TouchDataDictionary.pickle", "wb") as output_file:
     pickle.dump(Data_dictionary, output_file, protocol=pickle.HIGHEST_PROTOCOL)
