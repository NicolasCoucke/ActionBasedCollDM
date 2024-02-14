################
# script to generate ROC (metacognition) data that can be analysed in R
# with main.py preprocessed data
################



import os
import pandas as pd
import numpy as np
import cmasher as cmr
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from Session import SessionClass
from type_2_ROC import type2roc

import pickle
import re

import tkinter
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

path = r"your path"
os.chdir(path)

with open(r"TouchDataDictionary.pickle", "rb") as input_file:
    Data_dictionary = pickle.load(input_file)


def get_speed_measure(x_positions, y_positions, xmin, xmax):
    # Calculate the speed based on the x and y positions
    speed = np.sqrt(
        np.square(x_positions[1:] - x_positions[:-1]) + np.square(y_positions[1:] - y_positions[:-1]))
    window_length = 50
    # Apply the moving average filter
    speed = np.convolve(speed, np.ones(window_length) / window_length, mode='same')
    speed_measure = np.nanmean(speed[xmin: xmax])

    return speed_measure


def get_speed_preferred_direction(x_positions, y_positions, xmin, xmax, actual_answer):
    CirclesPositions = np.array([[-4, 4], [4, 4], [-4, -4], [4, -4]])

    # get euclidean distance to preferred option
    circle_x = CirclesPositions[actual_answer - 1][0]
    circle_z = CirclesPositions[actual_answer - 1][1]

    # calculate distance to consensuslocation
    eucl_distance = np.sqrt(np.square(x_positions - circle_x * np.ones(
        len(x_positions), )) + np.square(
        y_positions - circle_z * np.ones(len(y_positions), )))

    # Calculate the speed based on the x and y positions
    # need to negate the speed because if the distance decreases then you are moving towards it
    speed = - (eucl_distance[1:] - eucl_distance[:-1])
    window_length = 50
    # Apply the moving average filter
    speed = np.convolve(speed, np.ones(window_length) / window_length, mode='same')
    speed_measure = np.nanmean(speed[xmin: xmax])

    return speed_measure


##TRIAL BASED CORRECT
orders = np.zeros((2,))
trial_influence_array = np.zeros((0, 13))
base_player_id = 0



for time_bin in range(10):
    xmin = time_bin*100
    xmax = 99 + time_bin*100
    for session in Data_dictionary.values():

        # for preprocessing
        if np.max(session.invalid_answers) > 32:
            continue


        # now calculate the measure for the participants
        for p in range(session.PlayerSize):

            all_speeds = []
            # first get all speed values in threshold:
            for trial in session.TrialList:
                if trial.Condition == 2:
                    continue

                if (trial.Guiding[p] != -1) and (trial.Confidences[p] in [1, 2, 3, 4]) and (
                        trial.ActualAnswers[p] in [1, 2, 3, 4]):
                    x_positions = np.array(trial.Players_x[p])
                    y_positions = np.array(trial.Players_y[p])
                    actual_answer = trial.ActualAnswers[p]
                    # get speed at beginning (first cluster)

                    # speed = get_speed_measure(x_positions, y_positions, xmin, xmax)
                    speed = get_speed_preferred_direction(x_positions, y_positions, xmin, xmax, actual_answer)
                    print(speed)
                    all_speeds.append(speed)

            # Calculate the quartiles for binning
            min_speed = 0  # np.nanmin([all_speeds])
            max_speed = np.nanmax([all_speeds])
            first_quart = min_speed + 0.25 * (max_speed - min_speed)
            second_quart = min_speed + 0.5 * (max_speed - min_speed)
            third_quart = min_speed + 0.75 * (max_speed - min_speed)
            quartiles = [first_quart, second_quart, third_quart]

            # get the normalizing mean and std for this session
            speeds = []
            binned_speeds = []
            answers = []
            group_answers = []
            confidences = []
            guiding = []
            for trial in session.TrialList:

                if trial.Condition == 2:
                    continue

                performance = np.sum(session.Trial_accuracies[p + 1, :])

                if (trial.Guiding[p] != -1) and (trial.Confidences[p] in [1, 2, 3, 4]) and (
                        trial.ActualAnswers[p] in [1, 2, 3, 4]):
                    x_positions = np.array(trial.Players_x[p])
                    y_positions = np.array(trial.Players_y[p])
                    actual_answer = trial.ActualAnswers[p]
                    # get speed at beginning (first cluster)
                    xmin = 0
                    xmax = 113
                    # speed = get_speed_measure(x_positions, y_positions, xmin, xmax)
                    speed = get_speed_preferred_direction(x_positions, y_positions, xmin, xmax, actual_answer)
                    print(speed)
                    all_speeds.append(speed)

                    if speed <= quartiles[0]:
                        binned_speeds.append(1)
                    elif (speed > quartiles[0]) and (speed <= quartiles[1]):
                        binned_speeds.append(2)
                    elif (speed > quartiles[1]) and (speed <= quartiles[2]):
                        binned_speeds.append(3)
                    else:
                        binned_speeds.append(4)

                    confidence = trial.Confidences[p]
                    confidences.append(confidence)
                    if trial.Answers[p] == True:
                        correct = 1
                    else:
                        correct = 0
                    answers.append(correct)

                    isguiding = trial.Guiding[p]
                    guiding.append(isguiding)

                    if trial.Success == True:
                        group_answers.append(1)
                    else:
                        group_answers.append(0)

            accuracy = np.nanmean(answers)
            group_accuracy = np.nanmean(group_answers)
            explicit_confidence_bias = np.mean(confidences)
            explicit_metacog_sens = type2roc(answers, confidences, 4)

            implicit_confidence_bias = np.mean(binned_speeds)
            implicit_metacog_sens = type2roc(answers, binned_speeds, 4)

            influence = np.nanmean(guiding)

            # Get values for the column 'Leader' and determine which individuals are most seen as leaders
            subjective_leader = 0
            subjective_trust = 0

            row = [session.Session_number, session.PlayerSize, session.Condition_order, group_accuracy, id, accuracy,
                   explicit_confidence_bias, explicit_metacog_sens, implicit_confidence_bias, implicit_metacog_sens,
                   influence, subjective_leader, subjective_trust, time_bin]

            trial_influence_array = np.append(trial_influence_array,
                                              np.array([row]), axis=0)
            base_player_id += session.PlayerSize


