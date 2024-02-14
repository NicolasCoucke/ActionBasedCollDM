######################
# script to create single-trial videos
# with pre-processed data created with the "main" script
#######################


import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import animation
import matplotlib as mpl
import re
import seaborn as sns
from scipy.optimize import curve_fit
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

path = r"your path"
os.chdir(path)

CirclesPositions = np.array([[-4, 4], [4, 4], [-4, -4], [4, -4]])

def DefineCircles(correct, ax):
    circleDiams = [12,13,14,15]
    print(correct)
    print(CirclesPositions[correct])
    for circle in range(4):
        if circle == correct:
            color = 'g'
            radius = 1.4
        else:
            color = 'r'
            radius = 1.4
        circle1 = plt.Circle((CirclesPositions[circle]), radius, color=color, fill=False, linewidth= 5)
        ax.add_patch(circle1)


# Load the data dictionary (created in the 'main' script)
with open(r"TouchDataDictionary.pickle", "rb") as input_file:
    Data_dictionary = pickle.load(input_file)


game = 1
for session in Data_dictionary.values():
    if session.Session_number < 5:
        continue

    for trial in session.TrialList:
        circle_x = CirclesPositions[trial.CorrectLocation - 1][0]
        circle_z = CirclesPositions[trial.CorrectLocation - 1][1]

        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10))
        DefineCircles(trial.CorrectLocation-1, ax1)
        invalid_trial = False
        for playernumber in range(len(trial.Players_x)):
            if (not (trial.ActualAnswers[playernumber] in [1, 2, 3, 4])) or (not (trial.Confidences[playernumber] in [1, 2, 3, 4])):
                invalid_trial = True
        if invalid_trial:
            continue

        if trial.Condition == 2:
            continue


        lines = []
        dots = []
        texts = []
        line_colors = []
        labels = []

        red_cmap = plt.get_cmap('Reds')
        green_cmap = plt.get_cmap('Greens')

        for playernumber in range(len(trial.Players_x)):
            if trial.Answers[playernumber] == False:
                if trial.Confidences[playernumber] == 1:
                    line_color = red_cmap(0.2)
                elif trial.Confidences[playernumber] == 2:
                    line_color = red_cmap(0.4)
                elif trial.Confidences[playernumber] == 3:
                    line_color = red_cmap(0.6)
                elif trial.Confidences[playernumber] == 4:
                    line_color = red_cmap(0.8)
            else:
                if trial.Confidences[playernumber] == 1:
                    line_color = green_cmap(0.2)
                elif trial.Confidences[playernumber] == 2:
                    line_color = green_cmap(0.4)
                elif trial.Confidences[playernumber] == 3:
                    line_color = green_cmap(0.6)
                elif trial.Confidences[playernumber] == 4:
                    line_color = green_cmap(0.8)
            line_colors.append(line_color)
            line, = ax1.plot([0, 0], color = line_colors[playernumber], linewidth = 4)
            lines.append(line)
            dot = ax1.scatter(0, 0, s = 150, c = line_colors[playernumber])
            dots.append(dot)
            print(str(playernumber+1))
            text = ax1.text(0, 0, s = str(playernumber+1), fontsize = 26)
            texts.append(text)

            label = f"Player {playernumber+1}; Correct: {trial.Answers[playernumber]}; Confidence: {trial.Confidences[playernumber]}"
            labels.append(label)
        legend = ax1.legend(labels, title=f"Correct: {trial.Success}; Difficulty: {trial.Difficulty}", fontsize = 12)
        legend.get_title().set_fontsize(12)  # Change the font size (12 points)
        legend.get_title().set_fontweight('bold')  # Change the font weight (bold)

        # Set the position of the legend
        legend.set_bbox_to_anchor((1, 0))  #
        def animate(frame):

            for playernumber in range(len(trial.Players_x)):
                lines[playernumber].set_data(trial.Players_x[playernumber][:frame], trial.Players_y[playernumber][:frame])
                dots[playernumber].set_offsets([trial.Players_x[playernumber][frame],
                                             trial.Players_y[playernumber][frame]])
                texts[playernumber].set_position((trial.Players_x[playernumber][frame],
                                              trial.Players_y[playernumber][frame]))

            return lines

        anim = animation.FuncAnimation(fig, animate, frames=len(trial.Players_x[0]), interval = 5, blit=True)

        f = 'Videos\session' + str(session.Session_number) + 'trial' + str(trial.TrialNumber) + '.mp4'
        print(f)
        writervideo = animation.FFMpegWriter(fps=30)
        anim.save(f, writer=writervideo)