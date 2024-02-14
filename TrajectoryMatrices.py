##############
# script to create transformed trajectories that can subsequently be analysed in "TrajectoryAnalysis.py"
# based on preprocessed data of "main.py"
##############


import os
import pandas as pd
import numpy as np
import cmasher as cmr
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import cm
from Session import SessionClass
import pickle

import matplotlib

path = r"your path"
os.chdir(path)
matplotlib.use('Qt5Agg')


# change here the input file (raw gamedata)
CirclesPositions = np.array([[-4, 4], [4, 4], [-4, -4], [4, -4]])

with open(r"TouchDataDictionary.pickle", "rb") as input_file:
   Data_dictionary = pickle.load(input_file)


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

def make_relative(trajectories, properties):
    # now make the trajectories relative
    player_interaction_array = np.zeros((1000,))
    properties_array = np.zeros((np.size(properties, 1)))
    sessions = np.unique(trajectory_properties[:, 0])
    properties.astype(int)
    for session in sessions:
        session_indices = np.where(trajectory_properties[:, 0] == session)[0]
        session_properties = trajectory_properties[session_indices, :]
        session_trajectories = trajectories[session_indices, :]
        trials = np.unique(session_properties[:, 2])

        for trial in trials:
            trial_indices = np.where(session_properties[:, 2] == trial)[0]
            trial_properties = session_properties[trial_indices, :]
            trial_trajectories = session_trajectories[trial_indices, :]
            trial_num_players = trial_properties[0, 1]


            player_correct = trial_properties[:, 6]
            player_consensus = trial_properties[:, 8]
            player_confidence = trial_properties[:, 7]
            player_answers = trial_properties[:, 11]

            print(np.shape(trial_trajectories))

            if np.size(trial_trajectories, 0) != trial_num_players:
                continue

            players_i = range(trial_num_players)
            for p_i in range(trial_num_players):
                # current player get trajectory
                player_trajectory = trial_trajectories[p_i, :]

                other_players = []
                for p_j in range(trial_num_players):
                    if p_i == p_j:
                        continue
                    other_players.append(p_j)

                other_trajectories = np.mean(trial_trajectories[other_players,:], 0)
                interactive_trajectory = player_trajectory - other_trajectories

                player_interaction_array = np.vstack((player_interaction_array, interactive_trajectory))
                properties_array = np.vstack((properties_array, trial_properties[p_i,:]))

    return player_interaction_array, properties_array


##ABSOLUTE TRAJECTORIES
orders = np.zeros((2,))
trial_influence_array = np.zeros((0, 9))
base_player_id = 0

distance_to_consensus_start_locked = np.zeros((1000,))
speed_to_consensus_start_locked = np.zeros((1000,))

distance_to_consensus_end_locked = np.zeros((1000,))
speed_to_consensus_end_locked = np.zeros((1000,))

# properties:
# session, players, trial, difficulty, player_id, group correct, individual_correct, individual_confidence,  individual_consensus, obstructed_consensus, obstructed_individual, actualanswer
trajectory_properties = np.zeros((13,), dtype=int)

reject_dict = {10: [3], 14: [26], 21: [28, 43], 42: [24], 45: [23, 29]}
total_trial_counter = 0
valid_trial_counter = 0
for session in Data_dictionary.values():

    # for preprocessing
    if np.max(session.invalid_answers) > 32:
        continue

    for trial in session.TrialList:

        circle_x = CirclesPositions[trial.ConsensusLocation - 1][0]
        circle_z = CirclesPositions[trial.ConsensusLocation - 1][1]

        if trial.Condition == 2:
            continue

        total_trial_counter += 1
        valid_trial_counter = 0

        # see if trial should be rejected

        if int(session.Session_number) in reject_dict.keys():
            if int(trial.TrialNumber) in reject_dict[session.Session_number]:
                continue
        # fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # Create a figure with one row and three columns

        for playernumber in range(len(trial.Players_x)):
            # print(playernumber)

            if (trial.ActualAnswers[playernumber] in [1, 2, 3, 4]) and (
                    trial.Confidences[playernumber] in [1, 2, 3, 4]):
                # plt.plot(trial.Players_x[playernumber], trial.Players_y[playernumber])

                # temporarily get for every player the distance to their preferred location:
                """
                circle_x = CirclesPositions[trial.ActualAnswers[playernumber] - 1][0]
                circle_z = CirclesPositions[trial.ActualAnswers[playernumber] - 1][1]
                """

                # calculate distance to consensuslocation
                eucl_distance = np.sqrt(np.square(trial.Players_x[playernumber] - circle_x * np.ones(
                    len(trial.Players_x[playernumber]), )) + np.square(
                    trial.Players_y[playernumber] - circle_z * np.ones(len(trial.Players_y[playernumber]), )))
                eucl_distance = normalize_between_start_and_end(eucl_distance)

                # store start locked distance
                eucl_distance_start_locked = elongate_array(eucl_distance, 1000)
                distance_to_consensus_start_locked = np.vstack(
                    (distance_to_consensus_start_locked, eucl_distance_start_locked))

                # store end locked distance
                eucl_distance_end_locked = elongate_array_end_locked(eucl_distance, 1000)
                distance_to_consensus_end_locked = np.vstack(
                    (distance_to_consensus_end_locked, eucl_distance_end_locked))

                # calculate speed during trial:

                # Convert the x and y position lists to NumPy arrays
                x_positions = np.array(trial.Players_x[playernumber])
                y_positions = np.array(trial.Players_y[playernumber])

                # Calculate the speed based on the x and y positions
                # speed = np.sqrt(
                # np.square(x_positions[1:] - x_positions[:-1]) + np.square(y_positions[1:] - y_positions[:-1]))
                # window_length = 50
                # Apply the moving average filter
                # calculate speed based on moving to the target
                speed = -100 * (eucl_distance[1:] - eucl_distance[:-1])
                window_length = 50
                # Apply the moving average filter
                speed = np.convolve(speed, np.ones(window_length) / window_length, mode='same')

                # store start locked distance
                speed_start_locked = add_zeros_to_end(speed, 1000)
                speed_to_consensus_start_locked = np.vstack((speed_to_consensus_start_locked, speed_start_locked))

                # axs[0].plot(speed_start_locked)
                # axs[1].plot(eucl_distance_start_locked )

                # store end locked distance
                speed_end_locked = add_zeros_to_start(speed, 1000)
                speed_to_consensus_end_locked = np.vstack((speed_to_consensus_end_locked, speed_end_locked))

                # save properties of trajectory
                session_number = int(session.Session_number)
                num_players = int(session.PlayerSize)
                trial_number = int(trial.TrialNumber)
                diff = int(trial.Difficulty)

                player_id = int(base_player_id + playernumber + 1)
                if trial.Success == True:
                    group_correct = 1
                else:
                    group_correct = 0

                if trial.Answers[playernumber] == True:
                    individual_correct = 1
                else:
                    individual_correct = 0
                individual_confidence = int(trial.Confidences[playernumber])

                if int(trial.ActualAnswers[playernumber]) == int(trial.ConsensusLocation):
                    individual_consensus = 1
                else:
                    individual_consensus = 0
                actual_answer = trial.ActualAnswers[playernumber]

                # check whether the player van move freely to option or is obstructed by the other players
                obstructed_individual = is_obstructed(playernumber + 1, trial.ActualAnswers[playernumber])
                obstructed_consensus = is_obstructed(playernumber + 1, trial.ConsensusLocation)

                completion_time = trial.CompletionTime
                print(completion_time)
                row = np.array(
                    [session_number, num_players, trial_number, diff, player_id, group_correct, individual_correct,
                     individual_confidence, individual_consensus, obstructed_individual, obstructed_consensus,
                     actual_answer, completion_time], dtype=float)
                trajectory_properties = np.vstack((trajectory_properties, row))

    # plt.title('session' + str(session.Session_number) + 'trial' + trial.TrialNumber)
    # plt.savefig('Visual_rejection_check/'+ 'session' + str(session.Session_number) + 'trial' + trial.TrialNumber)
    # plt.close('all')
    base_player_id += session.PlayerSize

with open(r"TrajectoryMatrices.pickle", "wb") as output_file:
    pickle.dump([distance_to_consensus_start_locked, speed_to_consensus_start_locked, distance_to_consensus_end_locked,
                 speed_to_consensus_end_locked, trajectory_properties], output_file, protocol=pickle.HIGHEST_PROTOCOL)

distance_to_consensus_start_locked, properties_array = make_relative(distance_to_consensus_start_locked,
                                                                     trajectory_properties)
distance_to_consensus_end_locked, properties_array = make_relative(distance_to_consensus_end_locked,
                                                                   trajectory_properties)
speed_to_consensus_start_locked, properties_array = make_relative(speed_to_consensus_start_locked,
                                                                  trajectory_properties)
speed_to_consensus_end_locked, trajectory_properties = make_relative(speed_to_consensus_end_locked,
                                                                     trajectory_properties)

with open(r"InteractiveTrajectoryMatrices.pickle", "wb") as output_file:
    pickle.dump([distance_to_consensus_start_locked, speed_to_consensus_start_locked, distance_to_consensus_end_locked,
                 speed_to_consensus_end_locked, trajectory_properties], output_file, protocol=pickle.HIGHEST_PROTOCOL)




##RELATIVE TRAJECTORIES
orders = np.zeros((2,))
trial_influence_array = np.zeros((0, 9))
base_player_id = 0

distance_to_consensus_start_locked = np.zeros((1000,))
speed_to_consensus_start_locked = np.zeros((1000,))

distance_to_consensus_end_locked = np.zeros((1000,))
speed_to_consensus_end_locked = np.zeros((1000,))

interactive_distance_to_consensus_start_locked = np.zeros((1000,))
interactive_speed_to_consensus_start_locked = np.zeros((1000,))

interactive_distance_to_consensus_end_locked = np.zeros((1000,))
interactive_speed_to_consensus_end_locked = np.zeros((1000,))

# properties:
# session, players, trial, difficulty, player_id, group correct, individual_correct, individual_confidence,  individual_consensus, obstructed_consensus, obstructed_individual, actualanswer, completion_time
trajectory_properties = np.zeros((13,), dtype=int)

reject_dict = {10: [3], 14: [26], 21: [28, 43], 42: [24], 45: [23, 29]}
total_trial_counter = 0
valid_trial_counter = 0
for session in Data_dictionary.values():

    # for preprocessing
    if np.max(session.invalid_answers) > 32:
        continue

    for trial in session.TrialList:



        if trial.Condition == 2:
            continue

        total_trial_counter += 1
        valid_trial_counter = 0

        # see if trial should be rejected

        if int(session.Session_number) in reject_dict.keys():
            if int(trial.TrialNumber) in reject_dict[session.Session_number]:
                continue
        # fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # Create a figure with one row and three columns

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

        for playernumber in range(len(trial.Players_x)):
            # print(playernumber)

            if (trial.ActualAnswers[playernumber] in [1, 2, 3, 4]) and (
                    trial.Confidences[playernumber] in [1, 2, 3, 4]):
                # plt.plot(trial.Players_x[playernumber], trial.Players_y[playernumber])

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
                distance_to_consensus_start_locked = np.vstack(
                    (distance_to_consensus_start_locked, eucl_distance_start_locked))

                # store end locked distance
                eucl_distance_end_locked = elongate_array_end_locked(eucl_distance, 1000)
                distance_to_consensus_end_locked = np.vstack(
                    (distance_to_consensus_end_locked, eucl_distance_end_locked))

                # calculate speed based on moving to the target
                speed = -100 * (eucl_distance[1:] - eucl_distance[:-1])
                window_length = 50
                # Apply the moving average filter
                speed = np.convolve(speed, np.ones(window_length) / window_length, mode='same')

                # store start locked distance
                speed_start_locked = add_zeros_to_end(speed, 1000)
                speed_to_consensus_start_locked = np.vstack((speed_to_consensus_start_locked, speed_start_locked))

                # axs[0].plot(speed_start_locked)
                # axs[1].plot(eucl_distance_start_locked )

                # store end locked distance
                speed_end_locked = add_zeros_to_start(speed, 1000)
                speed_to_consensus_end_locked = np.vstack((speed_to_consensus_end_locked, speed_end_locked))


                # now make it relative also
                distance_to_consensus_start_locked_local = np.zeros((1000,))
                speed_to_consensus_start_locked_local = np.zeros((1000,))

                distance_to_consensus_end_locked_local = np.zeros((1000,))
                speed_to_consensus_end_locked_local = np.zeros((1000,))
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

                        # store end locked distance
                        eucl_distance_end_locked_local = elongate_array_end_locked(eucl_distance, 1000)
                        distance_to_consensus_end_locked_local = np.vstack(
                            (distance_to_consensus_end_locked_local, eucl_distance_end_locked_local))

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
                        speed_to_consensus_end_locked_local = np.vstack(
                            (speed_to_consensus_end_locked_local, speed_end_locked_local))


                other_distance_to_consensus_start_locked = np.nanmean(distance_to_consensus_start_locked_local[1:, :],
                                                                      axis=0)
                other_distance_to_consensus_end_locked = np.nanmean(distance_to_consensus_end_locked_local[1:, :],
                                                                    axis=0)
                other_speed_to_consensus_start_locked = np.nanmean(speed_to_consensus_start_locked_local[1:, :], axis=0)
                other_speed_to_consensus_end_locked = np.nanmean(speed_to_consensus_end_locked_local[1:, :], axis=0)

                interactive_distance_to_consensus_start_locked = np.vstack((
                                                                           interactive_distance_to_consensus_start_locked,
                                                                           eucl_distance_start_locked - other_distance_to_consensus_start_locked))
                interactive_distance_to_consensus_end_locked = np.vstack((interactive_distance_to_consensus_end_locked,
                                                                          eucl_distance_end_locked - other_distance_to_consensus_end_locked))

                interactive_speed_to_consensus_start_locked = np.vstack((
                    interactive_speed_to_consensus_start_locked,
                    speed_start_locked - other_speed_to_consensus_start_locked))

                interactive_speed_to_consensus_end_locked = np.vstack((interactive_speed_to_consensus_end_locked,
                                                                       speed_end_locked - other_speed_to_consensus_end_locked))

                # save properties of trajectory
                session_number = int(session.Session_number)
                num_players = int(session.PlayerSize)
                trial_number = int(trial.TrialNumber)
                diff = int(trial.Difficulty)

                player_id = int(base_player_id + playernumber + 1)
                if trial.Success == True:
                    group_correct = 1
                else:
                    group_correct = 0

                if trial.Answers[playernumber] == True:
                    individual_correct = 1
                else:
                    individual_correct = 0
                individual_confidence = int(trial.Confidences[playernumber])

                if int(trial.ActualAnswers[playernumber]) == int(trial.ConsensusLocation):
                    individual_consensus = 1
                else:
                    individual_consensus = 0
                actual_answer = trial.ActualAnswers[playernumber]

                # check whether the player van move freely to option or is obstructed by the other players
                obstructed_individual = is_obstructed(playernumber + 1, trial.ActualAnswers[playernumber])
                obstructed_consensus = is_obstructed(playernumber + 1, trial.ConsensusLocation)

                completion_time = trial.CompletionTime
                print(completion_time)
                row = np.array(
                    [session_number, num_players, trial_number, diff, player_id, group_correct, individual_correct,
                     individual_confidence, individual_consensus, obstructed_individual, obstructed_consensus,
                     actual_answer, completion_time], dtype=int)
                trajectory_properties = np.vstack((trajectory_properties, row))
    # plt.title('session' + str(session.Session_number) + 'trial' + trial.TrialNumber)
    # plt.savefig('Visual_rejection_check/'+ 'session' + str(session.Session_number) + 'trial' + trial.TrialNumber)
    # plt.close('all')
    base_player_id += session.PlayerSize

with open(r"TrajectoryMatricesPreferred.pickle", "wb") as output_file:
    pickle.dump([distance_to_consensus_start_locked, speed_to_consensus_start_locked, distance_to_consensus_end_locked,
                 speed_to_consensus_end_locked, trajectory_properties], output_file, protocol=pickle.HIGHEST_PROTOCOL)

with open(r"InteractiveTrajectoryMatricesPreferred.pickle", "wb") as output_file:
    pickle.dump([interactive_distance_to_consensus_start_locked, interactive_speed_to_consensus_start_locked,
                 interactive_distance_to_consensus_end_locked, interactive_speed_to_consensus_end_locked,
                 trajectory_properties], output_file, protocol=pickle.HIGHEST_PROTOCOL)









