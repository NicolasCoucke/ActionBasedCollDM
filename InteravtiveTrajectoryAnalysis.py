##########################
# Script to create plots illustrating minority and majority influence dynamics
# Based on data preprocessed with the main script
##########################



import os
import pandas as pd
import numpy as np
import cmasher as cmr
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import cm
from Session import SessionClass
import pickle
import re
from matplotlib.lines import Line2D
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib
import mne
matplotlib.use('Qt5Agg')
from mne.stats import f_threshold_mway_rm, f_mway_rm, fdr_correction



path = r"C:\Users\Administrator\Documents\GAMEDATA\TouchDM"
os.chdir(path)

with open(r"TrajectoryMatrices.pickle", "rb") as input_file:
   distance_to_consensus_start_locked, speed_to_consensus_start_locked, distance_to_consensus_end_locked, speed_to_consensus_end_locked, trajectory_properties = pickle.load(input_file)



def elements_identical(arr):
    return all(x == arr[0] for x in arr)




fig, axs = plt.subplots(2, 3, figsize=(15, 5))  # Create a figure with one row and three columns

#[time, correct, group_size, n_leader]

consensus_time_array = np.zeros((0,4))
completion_times = trajectory_properties[:,12]
trajectory_properties = trajectory_properties.astype(int)
total_trials_3 = 0
total_trials_4 = 0
for group_size_idx, group_size in enumerate(range(2, 5)):
    # check effect of more people agreeing
    for leader_size_idx, leader_size in enumerate(range(1,4)):
        trajectory_samples = []
        total_trajectory_samples = []
        if (group_size == 2) and (leader_size == 2):
            continue
        if (group_size == 2) and (leader_size == 3):
            continue
        if (group_size == 3) and (leader_size == 3):
            continue

        if (group_size == 2) or (group_size == 3):
            ply = 0
        else:
            ply = 1

        if (group_size == 2):
            plx = 0
        elif (group_size == 3):
            plx = leader_size
        elif (group_size == 4):
            plx = leader_size_idx

        ax = axs[ply, plx]  # Select the corresponding subplot

        labels = []  # To store labels for the legend

        for p_correct_idx, p_correct in enumerate([0, 1]):
            for p_conf_idx, p_conf in enumerate([0, 1]):
                trajectory_array = np.zeros((1,1000))
                sessions = np.unique(trajectory_properties[:, 0])
                total_trajectories = 0
                influence_trajectories = 0
                for session in sessions:
                    session_indices = np.where(trajectory_properties[:, 0] == session)[0]
                    session_properties = trajectory_properties[session_indices, :]
                    session_trajectories = distance_to_consensus_start_locked[session_indices, :]
                    session_completion_times = completion_times[session_indices]
                    trials = np.unique(session_properties[:, 2])

                    for trial in trials:
                        trial_indices = np.where(session_properties[:, 2] == trial)[0]
                        trial_properties = session_properties[trial_indices, :]
                        trial_trajectories = session_trajectories[trial_indices, :]
                        trial_num_players = trial_properties[0, 1]
                        trial_completion_times = session_completion_times[trial_indices]


                        if trial_num_players != group_size:
                            continue



                        player_correct = trial_properties[:, 6]
                        player_consensus = trial_properties[:, 8]
                        player_confidence = trial_properties[:, 7]

                        # only take trials where all players have valid answers
                        if len(player_confidence) < group_size:
                            continue



                        # count the number of occurances of each opninion in the group
                        unique_elements, counts = np.unique(player_correct, return_counts=True)
                        for element in unique_elements[counts == leader_size]:
                            leaders = np.where(player_correct == element)[0]
                            # figure out if the leader is correct
                            if player_correct[leaders[0]] == 1:
                                if p_correct == 0:
                                    continue
                            else:
                                if p_correct == 1:
                                    continue

                            # figure out if the leader has the highest confidence
                            follower_confidences = np.delete(player_confidence, leaders)
                            # print(player_confidence)
                            # print(follower_confidences)
                            if np.max(player_confidence[leaders]) > np.max(follower_confidences):
                                if p_conf == 0:
                                    continue
                            elif np.max(player_confidence[leaders]) < np.max(follower_confidences):
                                if p_conf == 1:
                                    continue
                            else:
                                # if they are equal then continue in all cases
                                continue

                            # count this as the total for each bar
                            total_trajectories+=1


                            # we only want cases where there is the desired leader size
                            if player_consensus[leaders[0]] != 1:
                                continue



                            influence_trajectories+=1
                            # find the players that are driving the consensus
                            leaders = np.where(player_consensus == 1)[0]

                            row = [trial_completion_times[0], trial_properties[0,5], trial_properties[0,1], len(leaders)]
                            consensus_time_array = np.append(consensus_time_array,
                                                              np.array([row]), axis=0)

                            leader_trajectory = np.mean(trial_trajectories[leaders,:], axis = 0)
                            follower_trajectory = np.mean(np.delete(trial_trajectories,leaders,axis=0), axis = 0)

                            trajectory_difference = (leader_trajectory - follower_trajectory) #/ (leader_trajectory + follower_trajectory)

                            # check where artefact comes from
                            """
                            if np.max(player_confidence[leaders]) < np.max(follower_confidences):
                                if player_correct[leaders[0]] == 0:
                                    if (p_correct == 0) and (p_conf == 0):
                                        ax.plot(leader_trajectory, linewidth = 1)
                                        ax.plot(follower_trajectory)
                                        if np.min(trajectory_difference) < -4:
                                            print(session)
                                            print(trial)
                            """

                            trajectory_array = np.vstack((trajectory_array, trajectory_difference.reshape(1,1000)))
                           #print(np.size(trajectory_array))

                # Store label for the legend
                red_cmap = plt.get_cmap('Reds')
                green_cmap = plt.get_cmap('Greens')

                # Define line color based on 'correct' and 'confidence' values
                if p_correct == 0:
                    if p_conf == 0:
                        line_color = red_cmap(0.4)
                        label = 'incorrect less confident'
                    elif p_conf == 1:
                        line_color = red_cmap(0.8)
                        label = 'incorrect more confident'
                else:
                    if p_conf == 0:
                        line_color = green_cmap(0.4)
                        label = 'correct less confident'
                    elif p_conf == 1:
                        line_color = green_cmap(0.8)
                        label = 'correct more confident'

                if label not in labels:
                    labels.append(label)
                    trajectory_samples.append(influence_trajectories)
                    total_trajectory_samples.append(total_trajectories)

                ax.plot(np.linspace(0, 10, 1000),np.nanmean(trajectory_array[1:,:], 0), label=label, color = line_color)
                ax.set_ylim([-1, 0.1])

        colors = [red_cmap(0.4), red_cmap(0.8), green_cmap(0.4), green_cmap(0.8)]
        ax_inset = ax.inset_axes([0.70, 0.05, 0.25, 0.35])  # Adjust the inset position and size as needed
        bars = ax_inset.bar([1, 2, 3, 4], height=trajectory_samples, color = colors)#, label = map(str, trajectory_samples))
        bars2 = ax_inset.bar([1, 2, 3, 4], height=total_trajectory_samples, edgecolor=colors, fill = False)
        for rect, rect2 in zip(bars,bars2):
            height = rect.get_height()
            height2 = rect2.get_height()
            ax_inset.annotate(f'{height}', xy=(rect2.get_x() + rect2.get_width() / 2, height2),
                              xytext=(0, 3),  # 3 points vertical offset
                              textcoords="offset points",
                              ha='center', va='bottom')
        ax_inset.set_ylim([0, 260])
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_xticklabels([])
        ax_inset.set_yticklabels([])
        ax.axhline(y=0, color='black')

        # Add title to the subplot
        ax.set_title(f"Group Size: {group_size}; Leader Size: {leader_size}")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Leader-follower distance (a.u.)')
    # Add legend
#axs[0,0].legend(labels, loc='lower left',bbox_to_anchor=(0.28, 0.02, 0.5, 0.5), prop={'size': 7})

df = pd.DataFrame(consensus_time_array) #easier to export if it is transformed to a pandas dataframe
df.columns = ["CompletionTime", "Correct", "Groupsize", "Leaders"]
filepath = 'CompletionTimes.xlsx' #change here the output file (excel)
df.to_excel(filepath, index=False)

plt.tight_layout()  # Adjust spacing between subplots
plt.show()





#[time, correct, group_size, n_leader]

consensus_time_array = np.zeros((0,4))
completion_times = trajectory_properties[:,12]
trajectory_properties = trajectory_properties.astype(int)

group_size = 3

light_blue = (173/255, 216/255, 230/255)
medium_blue = (0, 119/255, 187/255)
dark_blue = (0, 51/255, 102/255)
labels_3 = ['1 Leader (Minority)', '2 Leaders (Majority)']
labels_4 = ['1 Leader (Minority)', '2 Leaders', '3 leaders (Majority)']
fig, axs = plt.subplots(1,2)  # Create a figure with one row and three columns
blues = [light_blue, medium_blue, dark_blue]

total_trials_3 = 0
total_trials_4 = 0
minority_trials_3 = 0
majority_trials_3 = 0
minority_trials_4 = 0
majority_trials_4 = 0
for group_size_idx, group_size in enumerate(range(3,5)):
    completion_time_bins = []

    if group_size == 3:
        labels = ['1 Leader (Minority)', '2 Leaders (Majority)']
        blues = [light_blue, dark_blue]

    else:
        labels = ['1 Leader (Minority)', '2 Leaders', '3 leaders (Majority)']
        blues = [light_blue, medium_blue, dark_blue]

    for leader_size_idx, leader_size in enumerate(range(1,4)):
        trajectory_samples = []
        total_trajectory_samples = []
        labels = []  # To store labels for the legend


        if (group_size == 3) and (leader_size == 3):
            continue

        trajectory_array = np.zeros((1,1000))
        sessions = np.unique(trajectory_properties[:, 0])
        total_trajectories = 0
        influence_trajectories = 0
        for session in sessions:
            session_indices = np.where(trajectory_properties[:, 0] == session)[0]
            session_properties = trajectory_properties[session_indices, :]
            session_trajectories = distance_to_consensus_start_locked[session_indices, :]
            session_completion_times = completion_times[session_indices]
            trials = np.unique(session_properties[:, 2])

            for trial in trials:
                trial_indices = np.where(session_properties[:, 2] == trial)[0]
                trial_properties = session_properties[trial_indices, :]
                trial_trajectories = session_trajectories[trial_indices, :]
                trial_num_players = trial_properties[0, 1]
                trial_completion_times = session_completion_times[trial_indices]



                if trial_num_players != group_size:
                    continue



                player_correct = trial_properties[:, 6]
                player_consensus = trial_properties[:, 8]
                player_confidence = trial_properties[:, 7]

                # only take trials where all players have valid answers
                if len(player_confidence) < group_size:
                    continue

                leaders = np.where(player_consensus == 1)[0]

                if group_size == 3:
                    total_trials_3 += 1
                    if len(leaders) == 2:
                        majority_trials_3+=1
                    elif len(leaders) == 1:
                        minority_trials_3+=1
                elif group_size == 4:
                    total_trials_4 += 1
                    if len(leaders) == 3:
                        majority_trials_4+=1
                    elif len(leaders) == 1:
                        minority_trials_4+=1

                if len(leaders) != leader_size:
                    continue


                influence_trajectories+=1
                # find the players that are driving the consensus
                leaders = np.where(player_consensus == 1)[0]


                leader_trajectory = np.mean(trial_trajectories[leaders,:], axis = 0)
                follower_trajectory = np.mean(np.delete(trial_trajectories,leaders,axis=0), axis = 0)

                trajectory_difference = (leader_trajectory - follower_trajectory) #/ (leader_trajectory + follower_trajectory)



                trajectory_array = np.vstack((trajectory_array, trajectory_difference.reshape(1,1000)), )
                #print(np.size(trajectory_array))

        print(' total trials for 3 players: ' + str(total_trials_3))
        print(' total trials for 4 players: ' + str(total_trials_4))
        print(' majority trials for 3 players: ' + str(majority_trials_3))
        print(' majority trials for 4 players: ' + str(majority_trials_4))
        print(' minority trials for 3 players: ' + str(minority_trials_3))
        print(' minority trials for 4 players: ' + str(minority_trials_4))
        axs[group_size_idx].plot(np.linspace(0, 10, 1000),np.nanmean(trajectory_array[1:,:], 0), color = blues[leader_size_idx], linewidth = 3)
        axs[group_size_idx].set_title('Group size ' + str(group_size))

    axs[group_size_idx].set_ylim([-0.5, 0.1])
    axs[group_size_idx].set_xlabel('Time (s)')
    axs[group_size_idx].set_ylabel('Leader-follower distance (a.u.)')


axs[0].legend(labels_3, loc='lower right')
axs[1].legend(labels_4, loc='lower right')
plt.tight_layout()  # Adjust spacing between subplots
plt.show()



