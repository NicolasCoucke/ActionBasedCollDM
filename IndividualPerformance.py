################
# script to calculate individual performance and metacognitive sensitivity
# based on data preprocessed in 'main'
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


path = r"C:\Users\Administrator\Documents\GAMEDATA\TouchDM"
os.chdir(path)

df = pd.read_excel(r'Questionnaire__Optimal_Embodied_Collective_Decision_Making.xlsx', sheet_name='Quanti')


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


def elongate_array(array, length):
    """
    takes in an array and if len(array) > length then the array is cut off
    if len(array) < length then the final entry of the array will be repeated until len(new_array) = length
    """

    if len(array) > length:
        new_array = array[:length]
    else:
        new_array = np.zeros((length,))
        new_array[:len(array)] = array
        new_array[len(array):] = array[-1]

    return new_array


def elongate_array_end_locked(array, length):
    """
    takes in an array and if len(array) > length then the array is cut off
    if len(array) < length then the final entry of the array will be repeated until len(new_array) = length
    """

    if len(array) > length:
        new_array = array[len(array)-length:]
    else:
        new_array = np.zeros((length,))
        new_array[length-len(array):] = array
        new_array[:length-len(array)] = array[0]

    return new_array


def add_nans_to_end(array, length):
    """
    takes in an array and if len(array) > length then the array is cut off
    if len(array) < length then the final entry of the array will be repeated until len(new_array) = length
    """

    if len(array) > length:
        new_array = array[:length]
    else:
        new_array = np.zeros((length,))
        new_array[:len(array)] = array
        new_array[len(array):] = np.nan

    return new_array

def add_zeros_to_end(array, length):
    """
    takes in an array and if len(array) > length then the array is cut off
    if len(array) < length then the final entry of the array will be repeated until len(new_array) = length
    """

    if len(array) > length:
        new_array = array[:length]
    else:
        new_array = np.zeros((length,))
        new_array[:len(array)] = array

    return new_array

def add_nans_to_start(array, length):
    """
    takes in an array and if len(array) > length then the array is cut off
    if len(array) < length then the final entry of the array will be repeated until len(new_array) = length
    """

    if len(array) > length:
        new_array = array[len(array)-length:]
    else:
        new_array = np.zeros((length,))
        new_array[length-len(array):] = array
        new_array[:length-len(array)] = np.nan

    return new_array

def add_zeros_to_start(array, length):
    """
    takes in an array and if len(array) > length then the array is cut off
    if len(array) < length then the final entry of the array will be repeated until len(new_array) = length
    """

    if len(array) > length:
        new_array = array[len(array)-length:]
    else:
        new_array = np.zeros((length,))
        new_array[length-len(array):] = array

    return new_array

def normalize_to_start(time_series):
    """
    gets the first (non-nan) value and then normalizes the time series with respect to that value
    """
    t = 0
    while np.isnan(time_series[t]):
        t += 1
    start_val = time_series[t + 1]
    time_series = time_series / start_val
    return time_series

def normalize_between_start_and_end(time_series):
    """
    gets the first (non-nan) value and then normalizes the time series with respect to that value
    """
    t = 0
    while np.isnan(time_series[t]):
        t += 1
    start_val = time_series[t + 1]

    end_val = time_series[-1]

    time_series = (time_series - end_val) / (start_val - end_val)
    return time_series

def normalize_between_start_and_preferred(time_series):
    """
    gets the first (non-nan) value and then normalizes the time series with respect to that value
    """
    t = 0
    while np.isnan(time_series[t]):
        t += 1
    start_val = time_series[t + 1]

    end_val = time_series[-1]

    time_series = (time_series - 0) / (start_val - 0)
    return time_series


def get_relative_speed_distance(trial, playernumber, xmin, xmax):

    CirclesPositions = np.array([[-4, 4], [4, 4], [-4, -4], [4, -4]])


    # get for every player the distance to their preferred location:
    circle_x = CirclesPositions[trial.ActualAnswers[playernumber] - 1][0]
    circle_z = CirclesPositions[trial.ActualAnswers[playernumber] - 1][1]

    # calculate distance to consensuslocation
    eucl_distance = np.sqrt(np.square(trial.Players_x[playernumber] - circle_x * np.ones(
        len(trial.Players_x[playernumber]), )) + np.square(
        trial.Players_y[playernumber] - circle_z * np.ones(len(trial.Players_y[playernumber]), )))

    # calculate speed during trial:

    eucl_distance = normalize_between_start_and_preferred(eucl_distance)
    # store start locked distance
    eucl_distance_start_locked = elongate_array(eucl_distance, 1000)


    # calculate speed based on moving to the target
    speed = -100 * (eucl_distance[1:] - eucl_distance[:-1])
    window_length = 50
    # Apply the moving average filter
    speed = np.convolve(speed, np.ones(window_length) / window_length, mode='same')

    # store start locked distance
    speed_start_locked = add_zeros_to_end(speed, 1000)


    # now make it relative also
    distance_to_consensus_start_locked_local = np.zeros((1000,))
    speed_to_consensus_start_locked_local = np.zeros((1000,))


    for playernumber_j in range(len(trial.Players_x)):
        # print(playernumber)
        if playernumber_j == playernumber:
            continue

        if (trial.ActualAnswers[playernumber_j] in [1, 2, 3, 4]) and (
                trial.Confidences[playernumber_j] in [1, 2, 3, 4]):
            # plt.plot(trial.Players_x[playernumber], trial.Players_y[playernumber])

            # calculate distance to consensuslocation
            eucl_distance = np.sqrt(np.square(trial.Players_x[playernumber_j] - circle_x * np.ones(
                len(trial.Players_x[playernumber_j]), )) + np.square(
                trial.Players_y[playernumber_j] - circle_z * np.ones(len(trial.Players_y[playernumber_j]), )))

            # calculate speed during trial:

            eucl_distance = normalize_between_start_and_preferred(eucl_distance)
            # store start locked distance
            eucl_distance_start_locked_local = elongate_array(eucl_distance, 1000)
            distance_to_consensus_start_locked_local = np.vstack(
                (distance_to_consensus_start_locked_local, eucl_distance_start_locked_local))

            # calculate speed based on moving to the target
            speed = -100 * (eucl_distance[1:] - eucl_distance[:-1])
            window_length = 50
            # Apply the moving average filter
            speed = np.convolve(speed, np.ones(window_length) / window_length, mode='same')

            # store start locked distance
            speed_start_locked_local = add_zeros_to_end(speed, 1000)
            speed_to_consensus_start_locked_local = np.vstack(
                (speed_to_consensus_start_locked_local, speed_start_locked_local))

            # axs[0].plot(speed_start_locked)
            # axs[1].plot(eucl_distance_start_locked )

            # store end locked distance
            speed_end_locked_local = add_zeros_to_start(speed, 1000)

        other_distance_to_consensus_start_locked = np.nanmean(distance_to_consensus_start_locked_local[1:, :], axis=0)

        other_speed_to_consensus_start_locked = np.nanmean(speed_to_consensus_start_locked_local[1:, :], axis=0)

        relative_distance_start_locked =  eucl_distance_start_locked - other_distance_to_consensus_start_locked
        relative_speed_start_locked = speed_start_locked - other_speed_to_consensus_start_locked

    return np.nanmean(eucl_distance_start_locked[xmin:xmax]), np.nanmean(speed_start_locked[xmin:xmax]), np.nanmean(relative_distance_start_locked[xmin:xmax]), np.nanmean(relative_speed_start_locked[xmin:xmax])


##TRIAL BASED CORRECT
orders = np.zeros((2,))
trial_influence_array = np.zeros((0,14))
base_player_id = 0





# first get all quartiles
all_speeds = []
for session in Data_dictionary.values():

    # for preprocessing
    if np.max(session.invalid_answers) > 32:
        continue

    # first get all speed values in threshold:
    for trial in session.TrialList:
        if trial.Condition == 2:
            continue
        for p in range(session.PlayerSize):
            if (trial.Guiding[p] != -1) and (trial.Confidences[p] in [1, 2, 3, 4]):
                x_positions = np.array(trial.Players_x[p])
                y_positions = np.array(trial.Players_y[p])

                # get speed at beginning (first cluster)
                xmin = 0
                xmax = 100
                speed = get_speed_measure(x_positions, y_positions, xmin, xmax)
                all_speeds.append(speed)

    # Calculate the quartiles for binning
    quartiles = np.percentile(all_speeds, [25, 50, 75])


for session in Data_dictionary.values():

    # for preprocessing
    if np.max(session.invalid_answers) > 32:
        continue


    # now calculate the measure for the participants
    for p in range(session.PlayerSize):
        for time_bin in range(10):
            xmin = time_bin * 100
            xmax = 99 + time_bin * 100
            all_speeds = []
            # first get all speed values in threshold:
            for trial in session.TrialList:
                if trial.Condition == 2:
                    continue

                valid_players = 0
                for playernumber in range(len(trial.Players_x)):
                    # print(playernumber)

                    if (trial.ActualAnswers[playernumber] in [1, 2, 3, 4]) and (
                            trial.Confidences[playernumber] in [1, 2, 3, 4]):
                        valid_players += 1

                if valid_players != trial.PlayerSize:
                    continue

                if valid_players < 2:
                    continue

                if (trial.Guiding[p] != -1) and (trial.Confidences[p] in [1, 2, 3, 4]) and (trial.ActualAnswers[p] in [1, 2, 3, 4]):
                    x_positions = np.array(trial.Players_x[p])
                    y_positions = np.array(trial.Players_y[p])
                    actual_answer = trial.ActualAnswers[p]

                    # get speed at beginning (first cluster)
                    eucl_distance_start_locked, speed_start_locked, relative_distance_start_locked, relative_speed_start_locked = get_relative_speed_distance(trial, p, xmin, xmax)
                    speed = relative_speed_start_locked

                    all_speeds.append(speed)

            # Calculate the quartiles for binning
            min_speed = np.nanmin([all_speeds])
            max_speed = np.nanmax([all_speeds])
            first_quart = min_speed + 0.25*(max_speed - min_speed)
            second_quart = min_speed + 0.5*(max_speed - min_speed)
            third_quart = min_speed + 0.75*(max_speed - min_speed)
            quartiles = [first_quart, second_quart, third_quart]
            all_speeds = []
            print(quartiles)

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

                valid_players = 0
                for playernumber in range(len(trial.Players_x)):
                    # print(playernumber)

                    if (trial.ActualAnswers[playernumber] in [1, 2, 3, 4]) and (
                            trial.Confidences[playernumber] in [1, 2, 3, 4]):
                        valid_players += 1

                if valid_players != trial.PlayerSize:
                    continue

                if valid_players < 2:
                    continue

                performance = np.sum(session.Trial_accuracies[p + 1,:])

                if (trial.Guiding[p] != -1) and (trial.Confidences[p] in [1, 2, 3, 4]) and (
                        trial.ActualAnswers[p] in [1, 2, 3, 4]):
                    x_positions = np.array(trial.Players_x[p])
                    y_positions = np.array(trial.Players_y[p])
                    actual_answer = trial.ActualAnswers[p]
                    # get speed at beginning (first cluster)

                    # speed = get_speed_measure(x_positions, y_positions, xmin, xmax)
                    #speed = get_speed_preferred_direction(x_positions, y_positions, xmin, xmax, actual_answer)
                    eucl_distance_start_locked, speed_start_locked, relative_distance_start_locked, relative_speed_start_locked = get_relative_speed_distance(
                        trial, p, xmin, xmax)
                    speed = relative_speed_start_locked
                    #print(speed)
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
            id = base_player_id + p + 1

            row = [session.Session_number, session.PlayerSize, session.Condition_order, group_accuracy,  id, accuracy, explicit_confidence_bias, explicit_metacog_sens, implicit_confidence_bias, implicit_metacog_sens, influence, subjective_leader, subjective_trust, time_bin]

            trial_influence_array = np.append(trial_influence_array,
                                      np.array([row]), axis=0)
        base_player_id+=session.PlayerSize

df = pd.DataFrame(trial_influence_array ) #easier to export if it is transformed to a pandas dataframe
df.columns = ["Session","Players","Order","GroupAccuracy","PlayerID", "Accuracy", "ExplicitBias","ExplicitSensitivity","ImplicitBias","ImplicitSensitivity", "Influence", "SubjectiveLeader", "SubjectiveTrust", "TimeBin"]
filepath = 'TouchDMIndividualPerformance_relative_speed.xlsx' #change here the output file (excel)
df.to_excel(filepath, index=False)





