#############
# Script creates excel data files that can be used in R
# based on data preprocessed in 'main.py'
#############


import os
import pandas as pd
import numpy as np
import cmasher as cmr
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from Session import SessionClass

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


def is_obstructed(playernumber, answer):
    """
    determines whether the player is obstructed in moving to the an option based on that option and the player number

    return 0 if not obstructed and 1 if obstructed

    """

    if playernumber == 1:
        if (answer == 1) or (answer == 2):
            obstructed = 0
        else:
            obstructed = 1
    elif playernumber == 2:
        if (answer == 3) or (answer == 4):
            obstructed = 0
        else:
            obstructed = 1
    elif playernumber == 3:
        if (answer == 2) or (answer == 4):
            obstructed = 0
        else:
            obstructed = 1
    elif playernumber == 4:
        if (answer == 1) or (answer == 3):
            obstructed = 0
        else:
            obstructed = 1
    else:
        obstructed = -1

    return obstructed



##TRIAL BASED CORRECT
orders = np.zeros((2,))
trial_influence_array = np.zeros((0,6))
base_player_id = 0
for session in Data_dictionary.values():

    # for preprocessing
    if np.max(session.invalid_answers) > 32:
        continue

    # get the normalizing mean and std for this session
    speeds = []
    for trial in session.TrialList:

        if trial.Condition == 2:
            continue

        for p in range(session.PlayerSize):
            if trial.Guiding[p] != -1:

                x_positions = np.array(trial.Players_x[p])
                y_positions = np.array(trial.Players_y[p])

                # get speed at beginning (first cluster)
                xmin = 0
                xmax = 113
                speed_measure_1 = get_speed_measure(x_positions, y_positions, xmin, xmax)
                speeds.append(speed_measure_1)

    normalizing_mean = np.nanmean(speeds)
    normalizing_std = np.nanstd(speeds)

    # now do the actual calculations
    for trial in session.TrialList:
        diff = trial.Difficulty

        if trial.Condition == 2:
            continue



        # look only for disagreements where at least one individual is correct
        n_correct = 0
        invalid = 0
        for p in range(session.PlayerSize):
            if trial.Answers[p] == True:
                n_correct+=1

            if trial.Confidences[p] not in [1, 2, 3, 4]:
                invalid+=1

        # of no one is correct or everyone is correct then discard trial
        if (n_correct == 0) or (n_correct == session.PlayerSize) or (invalid > 0):
            continue

        max_conf_ind = np.argmax(trial.Confidences)
        min_conf_ind = np.argmin(trial.Confidences)


        # get speed of first individual
        x_positions = np.array(trial.Players_x[max_conf_ind])
        y_positions = np.array(trial.Players_y[max_conf_ind])

        # get speed at beginning (first cluster)
        xmin = 0
        xmax = 113
        speed_measure_max_ind = get_speed_measure(x_positions, y_positions, xmin, xmax)


        # get speed of second individual
        x_positions = np.array(trial.Players_x[min_conf_ind])
        y_positions = np.array(trial.Players_y[min_conf_ind])

        # get speed at beginning (first cluster)
        xmin = 0
        xmax = 113
        speed_measure_max_ind = get_speed_measure(x_positions, y_positions, xmin, xmax)

        # get later speed (second cluster
        xmin = 142
        xmax = 341
        speed_measure_min_ind = get_speed_measure(x_positions, y_positions, xmin, xmax)

        # get difference of normalized speeds
        normalized_max_speed = (speed_measure_max_ind - normalizing_mean) / normalizing_std
        normalized_min_speed = (speed_measure_min_ind - normalizing_mean) / normalizing_std
        speed_difference = speed_measure_max_ind - speed_measure_min_ind

        if trial.Success == True:
            correct = 1
        else:
            correct = 0


        row = [session.Session_number, session.PlayerSize, session.Condition_order, diff, correct, speed_difference]

        trial_influence_array = np.append(trial_influence_array,
                                  np.array([row]), axis=0)
        base_player_id+=session.PlayerSize

print(orders)
df = pd.DataFrame(trial_influence_array ) #easier to export if it is transformed to a pandas dataframe
df.columns = ["Session","Players","Order","Difficulty", "Correct", "SpeedDifference"]
filepath = 'TouchDMTrial_Correct.xlsx' #change here the output file (excel)
df.to_excel(filepath, index=False)





##TRIAL BASED INFLUENCE
orders = np.zeros((2,))
trial_influence_array = np.zeros((0,12))
base_player_id = 0
for session in Data_dictionary.values():

    # for preprocessing
    if np.max(session.invalid_answers) > 32:
        continue

    for trial in session.TrialList:

        if trial.Condition == 2:
            continue

        for p in range(session.PlayerSize):
            if trial.Guiding[p] != -1:
                diff = trial.Difficulty
                confidence = trial.Confidences[p]
                correct = trial.Answers[p]
                average_accuracy = np.mean(session.Trial_accuracies[p+1,:])
                guiding = trial.Guiding[p]

                x_positions = np.array(trial.Players_x[p])
                y_positions = np.array(trial.Players_y[p])

                # get speed at beginning (first cluster)
                xmin = 0
                xmax = 113
                speed_measure_1 = get_speed_measure(x_positions, y_positions, xmin, xmax)

                # get later speed (second cluster
                xmin = 142
                xmax = 341
                speed_measure_2 = get_speed_measure(x_positions, y_positions, xmin, xmax)

                obstruced = is_obstructed(p+1, trial.ActualAnswers[p])

                id = base_player_id + p + 1

                if confidence not in [1, 2, 3, 4]:
                    continue

                row = [session.Session_number, session.PlayerSize, session.Condition_order, diff, id, confidence, correct, average_accuracy, guiding, speed_measure_1, speed_measure_2, obstruced]

                trial_influence_array = np.append(trial_influence_array,
                                          np.array([row]), axis=0)
        base_player_id+=session.PlayerSize

print(orders)
df = pd.DataFrame(trial_influence_array ) #easier to export if it is transformed to a pandas dataframe
df.columns = ["Session","Players","Order","Difficulty", "PlayerID", "Confidence", "Correct", "AverageAccuracy", "Guiding", "Speed1", "Speed2", "Obstructed"]
filepath = 'TouchDMTrial_Influence.xlsx' #change here the output file (excel)
df.to_excel(filepath, index=False)



##Group diversion from MCS#
orders = np.zeros((2,))
group_performance_array = np.zeros((0,9))
invalid_ratios = []
player_groups = []
for session in Data_dictionary.values():
    invalid_ratios.append(session.different_from_MCS)

    if np.max(session.invalid_answers) > 32:
       player_groups.append(0)
    else:
        player_groups.append(1)
cmap = plt.get_cmap('tab20')  # Choose a colormap, such as 'tab20' with 20 distinct colors

plt.bar(x=range(len(invalid_ratios)), height=np.array(invalid_ratios), color = cmap(player_groups))
plt.title('number of group decisions that are different from the MCS')
plt.show()


##Group Performance per difficulty#
orders = np.zeros((2,))
group_performance_array = np.zeros((0,9))
invalid_ratios = []
player_groups = []
for session in Data_dictionary.values():
    invalid_ratios = np.concatenate((np.array(invalid_ratios),session.invalid_answers))
    player_groups = np.concatenate((np.array(player_groups), int(session.Session_number)*np.ones((session.PlayerSize,), dtype = int)))

    print(session.invalid_answers)

cmap = plt.get_cmap('tab20')  # Choose a colormap, such as 'tab20' with 20 distinct colors

plt.bar(x=range(len(invalid_ratios)), height=invalid_ratios, color=cmap(player_groups))
plt.show()

number_of_groups_pre = np.zeros((3,))
number_of_groups_post = np.zeros((3,))
for session in Data_dictionary.values():
    number_of_groups_pre[session.PlayerSize - 2] += 1

    print(np.max(session.invalid_answers))
    if np.max(session.invalid_answers) > 32:
        continue
    number_of_groups_post[session.PlayerSize - 2] += 1

plt.bar(x = [2, 3, 4], height = number_of_groups_pre)
plt.bar(x = [2, 3, 4], height = number_of_groups_post)
plt.show()



##GrOUP PErFOMANCE per difficulty#
orders = np.zeros((2,))
group_performance_array = np.zeros((0,9))
for session in Data_dictionary.values():

    # for preprocessing
    if np.max(session.invalid_answers) > 32:
        continue

    for diff in range(4):
        group_performance = session.Trial_accuracies[0,diff]
        MCS_performance = session.Trial_MCS_accuracies[diff]

        player_performances = []
        for p in range(session.PlayerSize):
            player_performances.append(session.Trial_accuracies[p + 1, diff])

        max_perf = np.max(player_performances)
        min_perf = np.min(player_performances)
        avg_pef = np.mean(player_performances)
        row = [session.Session_number, session.PlayerSize, session.Condition_order, diff+1, min_perf, avg_pef,  max_perf, MCS_performance, group_performance]
        group_performance_array = np.append(group_performance_array,
                                  np.array([row]), axis=0)

    orders[session.Condition_order-1]+=1
print(orders)
df = pd.DataFrame(group_performance_array ) #easier to export if it is transformed to a pandas dataframe
df.columns = ["Session","Players","Order","Difficulty", "Worst", "Average", "Best", "MCS", "Group"]
filepath = 'TouchDMGroup_Individual_Performance.xlsx' #change here the output file (excel)
df.to_excel(filepath, index=False)
