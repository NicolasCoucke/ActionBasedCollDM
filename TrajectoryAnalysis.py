############
# Script to perform cluster-based permutation analysis on transformed trajectories
# uses processed data generated in the 'TrajectoryMatrices.py' script
############

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


np.random.seed(seed=1)

cluster_array = np.zeros((3,))

def get_two_way_clusters(data, factor_levels, n_replications):
    """
    given a data array of dimensions (conditions x replications x times) this functions returns the significant clusters for the two main effects and an interaction effect
    other inputs of the function should be the factor levels (usually [2, 4])

    """

    significant_segments = []
    significant_pvals = []
    n_replications = np.size(data,1)
    for i, effect in enumerate(["A", "B", "A:B"]):
        #print(effect)
        # we have to define a new stat function for every effect
        def stat_fun(*args):
            return f_mway_rm(
                np.swapaxes(args, 1, 0),
                # args,
                factor_levels=factor_levels,
                effects=effect,
                return_pvals=False,
            )[0]

        pthresh = 0.05
        f_thresh = f_threshold_mway_rm(n_replications, factor_levels, effect, pthresh)

        tail = 1
        n_permutations = 1000

        F_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
            np.swapaxes(data, 0, 1),
            stat_fun=stat_fun,
            threshold=f_thresh,
            tail=tail,
            n_permutations=n_permutations,
            buffer_size=None,
            out_type="mask",
        )

        F_obs_plot = F_obs.copy()
        mask = np.ones_like(F_obs_plot, dtype=bool)
        p_vals = []
        for i_c, c in enumerate(clusters):
            c = c[0]
            if cluster_p_values[i_c] < 0.01:
                #exclude the significant parts from the mask
                mask[c.start:c.stop] = False
                p_vals.append(cluster_p_values[i_c])

        # Assign np.nan to elements outside the range
        F_obs_plot[mask] = np.nan

        significant_segments.append(F_obs_plot)
        significant_pvals.append(p_vals)
    return significant_segments, significant_pvals


def plot_clusters(significant_clusters, labels, pvals):
    ax = plt.gca()
    min_y = min(ax.get_ylim())
    plot_heights = min_y + [-0.02, -0.06, -0.10]
    colors = ['blue', 'orange', 'cyan']
    names = ['opinion', 'confidence', 'interaction']
    for i in range(3):
        cluster = significant_clusters[i]
        line = plot_heights[i] * np.ones((1000,))
        # apply the mask
        line[np.isnan(cluster)] = np.nan

        plt.plot(np.linspace(0, 10, 1000),line, alpha=0.6, color=colors[i], linewidth=3, label = labels[i])
        # Plot asterisk underneath each line segment in the middle
        x_positions = np.where(~np.isnan(cluster))[0]

        times = np.linspace(0, 10, 1000)
        k = 0
        while k < len(cluster):
            line_start = k
            line_end = k
            while not np.isnan(cluster[k]):
                line_end += 1
                k += 1
                if k >= len(cluster):
                    break
            if line_end != line_start:
                x_pos = line_end - 0.5 * (line_end - line_start)
                y_pos = plot_heights[i] - 0.025
                plt.text(times[int(np.round(x_pos))], y_pos, '*', color=colors[i], fontsize=16, horizontalalignment='center',
                         verticalalignment='center', weight='bold')
                print(names[i] + ' start ' + str(line_start) + ' end ' + str(line_end) + ' pval ' + str(pvals[i]))
            k += 1



path = r"C:\Users\Administrator\Documents\GAMEDATA\TouchDM"
os.chdir(path)


file_names = [r"TrajectoryMatrices.pickle", r"InteractiveTrajectoryMatrices.pickle", r"TrajectoryMatricesPreferred.pickle", r"InteractiveTrajectoryMatricesPreferred.pickle"]

distance_clusters = []
distance_pvals = []

speed_clusters = []
speed_pvals = []

for file in file_names:

    #with open(r"InteractiveTrajectoryMatrices.pickle", "rb") as input_file:
    with open(file, "rb") as input_file:
       distance_to_consensus_start_locked, speed_to_consensus_start_locked, distance_to_consensus_end_locked, speed_to_consensus_end_locked, trajectory_properties = pickle.load(input_file)




    # session, players, trial, difficulty, player_id, group correct, individual_correct, individual_confidence,  individual_consensus, obstructed_consensus, obstructed_individual
    speed_to_consensus_end_locked = speed_to_consensus_end_locked
    speed_to_consensus_start_locked = speed_to_consensus_start_locked
    # plots for DISTANCE and CORRECT/INCORRECT

    threshold = None
    permutations = 1000

    # look at trajectories of individuls that got their way
    # session, players, trial, difficulty, player_id, group correct, individual_correct, individual_confidence,  individual_consensus, obstructed_consensus, obstructed_individual, actualanswer

    group_sizes = []
    for correct in [0, 1]:
        for confidence in [1, 2, 3, 4]:
            confidence_indices = np.where((trajectory_properties[:, 5] == 1) & (trajectory_properties[:, 6] == correct) & (trajectory_properties[:, 7] == confidence))[0]
            #confidence_indices = np.where( (trajectory_properties[:, 6] == correct) & (trajectory_properties[:, 7] == confidence))[0]

            group_sizes.append(len(confidence_indices))
            confidence_average_trajectory = np.nanmean(distance_to_consensus_start_locked[confidence_indices, :], axis=0)
            red_cmap = plt.get_cmap('Reds')
            green_cmap = plt.get_cmap('Greens')

            # Define line color based on 'correct' and 'confidence' values
            if correct == 0:
                if confidence == 1:
                    line_color = red_cmap(0.2)
                elif confidence == 2:
                    line_color = red_cmap(0.4)
                elif confidence == 3:
                    line_color = red_cmap(0.6)
                elif confidence == 4:
                    line_color = red_cmap(0.8)
                label = "Correct: No, Confidence: {}".format(confidence)
            else:
                if confidence == 1:
                    line_color = green_cmap(0.2)
                elif confidence == 2:
                    line_color = green_cmap(0.4)
                elif confidence == 3:
                    line_color = green_cmap(0.6)
                elif confidence == 4:
                    line_color = green_cmap(0.8)
                label = "Correct: Yes, Confidence: {}".format(confidence)

            plt.plot(np.linspace(0, 10, 1000),confidence_average_trajectory, color=line_color, linewidth=2, label=label)




    # and now we want to put the trajectories into a mne permutation test

    smallest_group = np.min(group_sizes)-1
    print(smallest_group)
    times = range(1000)
    n_times = len(times)
    X = np.empty((smallest_group, n_times))  # Initialize an empty array


    # construct the matrix to start permutations
    for correct in [0, 1]:
        for confidence in [1, 2, 3, 4]:
            confidence_indices = np.where((trajectory_properties[:, 6] == correct) & (trajectory_properties[:, 7] == confidence))[0]
            #confidence_indices = np.where((trajectory_properties[:, 5] == 1) & (trajectory_properties[:, 6] == correct) & (trajectory_properties[:, 7] == confidence))[0]

            confidence_trajectories = distance_to_consensus_start_locked[confidence_indices, :]

            # Delete nan values
            nan_rows_confidence = np.isnan(confidence_trajectories).any(axis=1)
            confidence_trajectories = confidence_trajectories[~nan_rows_confidence]

            # Remove rows to equalize
            if len(confidence_trajectories) > smallest_group:
                random_indices = np.random.choice(len(confidence_trajectories), size=smallest_group, replace=False)
                confidence_trajectories = confidence_trajectories[random_indices]

            print(np.shape(confidence_trajectories))
            X = np.dstack((X, confidence_trajectories))  # Stack the matrices vertically



    # now swap axis so we have obs x condition x time
    data = np.swapaxes(np.asarray(X[:,:,1:]), 2, 1)
    n_replications = smallest_group
    print(n_replications)
    factor_levels = [2,4]
    significant_clusters_correct_distance, significant_pvals_correct_distance = get_two_way_clusters(data, factor_levels, n_replications)

    plt.tick_params(axis='both', which='major', labelsize=12)
    labels = ["Correct Cluster", "Confidence Cluster", "Correct*Confidence Cluster"]
    plot_clusters(significant_clusters_correct_distance, labels, significant_pvals_correct_distance)


    plt.xlabel("Time (s)", fontsize = 12)
    plt.ylabel("Distance to Consensus", fontsize = 12)
    #plt.legend(title="Correct and Confidence", fontsize = 10)
    plt.xlim(0, 10)
    #plt.ylim(-0.1,1.1)
    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()
    #plt.savefig('correct_distance_clusters.png')
    plt.clf()
    plt.figure()


    ########################################
    ########################################
    ########################################



    # plots for SPEED and CORRECT/INCORRECT
    group_sizes = []
    for correct in [0, 1]:
        for confidence in [1, 2, 3, 4]:
            confidence_indices = np.where((trajectory_properties[:, 5] == 1) & (trajectory_properties[:, 6] == correct) & (trajectory_properties[:, 7] == confidence))[0]
            group_sizes.append(len(confidence_indices))

            confidence_average_trajectory = np.nanmean(speed_to_consensus_start_locked[confidence_indices, :], axis=0)

            red_cmap = plt.get_cmap('Reds')
            green_cmap = plt.get_cmap('Greens')

            # Define line color based on 'correct' and 'confidence' values
            if correct == 0:
                if confidence == 1:
                    line_color = red_cmap(0.2)
                elif confidence == 2:
                    line_color = red_cmap(0.4)
                elif confidence == 3:
                    line_color = red_cmap(0.6)
                elif confidence == 4:
                    line_color = red_cmap(0.8)
                label = "Correct: No, Confidence: {}".format(confidence)
            else:
                if confidence == 1:
                    line_color = green_cmap(0.2)
                elif confidence == 2:
                    line_color = green_cmap(0.4)
                elif confidence == 3:
                    line_color = green_cmap(0.6)
                elif confidence == 4:
                    line_color = green_cmap(0.8)
                label = "Correct: Yes, Confidence: {}".format(confidence)

            plt.plot(np.linspace(0, 10, 1000),confidence_average_trajectory, color=line_color, linewidth=2, label=label)




    # and now we want to put the trajectories into a mne permutation test

    smallest_group = np.min(group_sizes)-1
    times = range(1000)
    n_times = len(times)
    X = np.empty((smallest_group, n_times))  # Initialize an empty array



    for correct in [0, 1]:
        for confidence in [1, 2, 3, 4]:
            confidence_indices = np.where((trajectory_properties[:, 5] == 1) & (trajectory_properties[:, 6] == correct) & (trajectory_properties[:, 7] == confidence))[0]
            confidence_trajectories = speed_to_consensus_start_locked[confidence_indices, :]

            # Delete nan values
            nan_rows_confidence = np.isnan(confidence_trajectories).any(axis=1)
            confidence_trajectories = confidence_trajectories[~nan_rows_confidence]

            # Remove rows to equalize
            if len(confidence_trajectories) > smallest_group:
                random_indices = np.random.choice(len(confidence_trajectories), size=smallest_group, replace=False)
                confidence_trajectories = confidence_trajectories[random_indices]

            X = np.dstack((X, confidence_trajectories))  # Stack the matrices vertically


    # now swap axis so we have obs x condition x time
    data = np.swapaxes(np.asarray(X[:,:,1:]), 2, 1)
    n_replications = smallest_group
    print(n_replications)
    significant_clusters_correct_speed, significant_pvals_correct_speed = get_two_way_clusters(data, factor_levels, n_replications)


    labels = ["Correct Cluster", "Confidence Cluster", "Correct*Confidence Cluster"]
    plot_clusters(significant_clusters_correct_speed, labels, significant_pvals_correct_speed)
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.xlabel("Time (s)", fontsize = 12)
    plt.ylabel("Speed", fontsize = 12)
    #plt.legend(title="Correct and Confidence", fontsize = 10)
    plt.xlim(0, 10)
    #plt.ylim(-0.1,0.1)
    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()
    #plt.savefig('correct_speed_clusters.png')
    plt.clf()
    plt.figure()

    distance_clusters.append(significant_clusters_correct_distance)
    distance_pvals.append(significant_pvals_correct_distance)

    speed_clusters.append(significant_clusters_correct_distance)
    speed_pvals.append(significant_pvals_correct_distance)

STOP
##############
#############
#############












































# plots for DISTANCE  and OBSTRUCTED - NOT OBSTRUCTED

# Check difference between high and low confidence movement with being obstructed
# fix: group correct, individual correct, group size
# vary: confidence, obstructed
for obstructed in [0, 1]:
    for confidence in [1, 2, 3, 4]:
        confidence_indices = np.where((trajectory_properties[:, 1] == 4) & (trajectory_properties[:, 5] == 1) & (trajectory_properties[:, 6] == 1) & (trajectory_properties[:, 7] == confidence) & (trajectory_properties[:, 9] == obstructed) )[0]
        confidence_average_trajectory = np.nanmean(distance_to_consensus_start_locked[confidence_indices, :], axis=0)
        red_cmap = plt.get_cmap('Blues')
        green_cmap = plt.get_cmap('Oranges')

        # Define line color based on 'correct' and 'confidence' values
        if obstructed == 0:
            if confidence == 1:
                line_color = red_cmap(0.2)
            elif confidence == 2:
                line_color = red_cmap(0.4)
            elif confidence == 3:
                line_color = red_cmap(0.6)
            elif confidence == 4:
                line_color = red_cmap(0.8)
            label = "Obstructed: No, Confidence: {}".format(confidence)
        else:
            if confidence == 1:
                line_color = green_cmap(0.2)
            elif confidence == 2:
                line_color = green_cmap(0.4)
            elif confidence == 3:
                line_color = green_cmap(0.6)
            elif confidence == 4:
                line_color = green_cmap(0.8)
            label = "Obstructed: Yes, Confidence: {}".format(confidence)

        plt.plot(np.linspace(0, 10, 1000),confidence_average_trajectory, color=line_color, linewidth=2, label=label)




smallest_group = 92
times = range(1000)
n_times = len(times)
X = np.empty((smallest_group, n_times))  # Initialize an empty array



for obstructed in [0, 1]:
    for confidence in [1, 2, 3, 4]:
        confidence_indices =  np.where((trajectory_properties[:, 1] == 4)  & (trajectory_properties[:, 5] == 1)  & (trajectory_properties[:, 6] == 1) & (trajectory_properties[:, 7] == confidence) & (trajectory_properties[:, 9] == obstructed))[0]
        confidence_trajectories = distance_to_consensus_start_locked[confidence_indices, :]

        # Delete nan values
        nan_rows_confidence = np.isnan(confidence_trajectories).any(axis=1)
        confidence_trajectories = confidence_trajectories[~nan_rows_confidence]

        # Remove rows to equalize
        if len(confidence_trajectories) > smallest_group:
            random_indices = np.random.choice(len(confidence_trajectories), size=smallest_group, replace=False)
            confidence_trajectories = confidence_trajectories[random_indices]

        X = np.dstack((X, confidence_trajectories))  # Stack the matrices vertically


# now swap axis so we have obs x condition x time
data = np.swapaxes(np.asarray(X[:,:,1:]), 2, 1)
n_replications = smallest_group
print(n_replications)
factor_levels = [2,4]
significant_clusters_obstructed_distance, significant_pvals_obstructed_distance = get_two_way_clusters(data, factor_levels, n_replications)


labels = ["Obstructed Cluster", "Confidence Cluster", "Obstructed*Confidence Cluster"]
plot_clusters(significant_clusters_obstructed_distance, labels)


plt.xlabel("Time (s)", fontsize = 12)
plt.ylabel("Distance to Consensus", fontsize = 12)
plt.legend(title="Obstructed and Confidence", fontsize = 10)
plt.xlim(0, 10)
#plt.ylim(-0.1,1.1)
plt.tight_layout()  # Adjust spacing between subplots
#plt.show()
plt.savefig('obstructed_distance_clusters.png')
plt.clf()
plt.figure()








# plots for SPEED and OBSTRUCTED - NOT OBSTRUCTED

# Check difference between high and low confidence movement with being obstructed
# fix: group correct, individual correct, group size
# vary: confidence, obstructed
for obstructed in [0, 1]:
    for confidence in [1, 2, 3, 4]:
        confidence_indices = np.where((trajectory_properties[:, 1] == 4) & (trajectory_properties[:, 5] == 1) & (trajectory_properties[:, 6] == 1) & (trajectory_properties[:, 7] == confidence) & (trajectory_properties[:, 9] == obstructed) )[0]
        confidence_average_trajectory = np.nanmean(speed_to_consensus_start_locked[confidence_indices, :], axis=0)
        red_cmap = plt.get_cmap('Blues')
        green_cmap = plt.get_cmap('Oranges')

        # Define line color based on 'correct' and 'confidence' values
        if obstructed == 0:
            if confidence == 1:
                line_color = red_cmap(0.2)
            elif confidence == 2:
                line_color = red_cmap(0.4)
            elif confidence == 3:
                line_color = red_cmap(0.6)
            elif confidence == 4:
                line_color = red_cmap(0.8)
            label = "Obstructed: No, Confidence: {}".format(confidence)
        else:
            if confidence == 1:
                line_color = green_cmap(0.2)
            elif confidence == 2:
                line_color = green_cmap(0.4)
            elif confidence == 3:
                line_color = green_cmap(0.6)
            elif confidence == 4:
                line_color = green_cmap(0.8)
            label = "Obstructed: Yes, Confidence: {}".format(confidence)

        plt.plot(np.linspace(0, 10, 1000),confidence_average_trajectory, color=line_color, linewidth=2, label=label)


# now make permutation test with the obstruction trajectories

smallest_group = 92
times = range(1000)
n_times = len(times)
X = np.empty((smallest_group, n_times))  # Initialize an empty array



for obstructed in [0, 1]:
    for confidence in [1, 2, 3, 4]:
        confidence_indices =  np.where((trajectory_properties[:, 1] == 4)  & (trajectory_properties[:, 5] == 1)  & (trajectory_properties[:, 6] == 1) & (trajectory_properties[:, 7] == confidence) &  (trajectory_properties[:, 9] == obstructed))[0]
        confidence_trajectories = speed_to_consensus_start_locked[confidence_indices, :]

        # Delete nan values
        nan_rows_confidence = np.isnan(confidence_trajectories).any(axis=1)
        confidence_trajectories = confidence_trajectories[~nan_rows_confidence]

        # Remove rows to equalize
        if len(confidence_trajectories) > smallest_group:
            random_indices = np.random.choice(len(confidence_trajectories), size=smallest_group, replace=False)
            confidence_trajectories = confidence_trajectories[random_indices]

        X = np.dstack((X, confidence_trajectories))  # Stack the matrices vertically


# now swap axis so we have obs x condition x time
data = np.swapaxes(np.asarray(X[:,:,1:]), 2, 1)
n_replications = smallest_group
print(n_replications)
factor_levels = [2,4]
significant_clusters_obstructed_speed, significant_pvals_obstructed_speed = get_two_way_clusters(data, factor_levels, n_replications)

labels = ["Obstructed Cluster", "Confidence Cluster", "Obstructed*Confidence Cluster"]
plot_clusters(significant_clusters_obstructed_speed, labels)


plt.xlabel("Time (s)", fontsize = 12)
plt.ylabel("Speed", fontsize = 12)
plt.legend(title="Obstructed and Confidence", fontsize = 10)
plt.xlim(0, 10)
#plt.ylim(-0.1,1.1)
plt.tight_layout()  # Adjust spacing between subplots
#plt.show()
plt.savefig('obstructed_speed_clusters.png')
plt.clf()
plt.figure()

# print the actual positions of these clusters
print('correct distance')
print(np.where(np.diff(np.isnan(significant_clusters_correct_distance[0])) != 0)[0])
print(significant_pvals_correct_distance[0])
print(np.where(np.diff(np.isnan(significant_clusters_correct_distance[1])) != 0)[0])
print(significant_pvals_correct_distance[1])
print(np.where(np.diff(np.isnan(significant_clusters_correct_distance[2])) != 0)[0])
print(significant_pvals_correct_distance[2])
print('correct speed')
print(np.where(np.diff(np.isnan(significant_clusters_correct_speed[0])) != 0)[0])
print(significant_pvals_correct_speed[0])
print(np.where(np.diff(np.isnan(significant_clusters_correct_speed[1])) != 0)[0])
print(significant_pvals_correct_speed[1])
print(np.where(np.diff(np.isnan(significant_clusters_correct_speed[2])) != 0)[0])
print(significant_pvals_correct_speed[2])
print('obstructed distance')
print(np.where(np.diff(np.isnan(significant_clusters_obstructed_distance[0])) != 0)[0])
print(significant_pvals_obstructed_distance[0])
print(np.where(np.diff(np.isnan(significant_clusters_obstructed_distance[1])) != 0)[0])
print(significant_pvals_correct_distance[1])
print(np.where(np.diff(np.isnan(significant_clusters_obstructed_distance[2])) != 0)[0])
print(significant_pvals_correct_distance[2])
print('obstructed speed')
print(np.where(np.diff(np.isnan(significant_clusters_obstructed_speed[0])) != 0)[0])
print(significant_pvals_obstructed_speed[0])
print(np.where(np.diff(np.isnan(significant_clusters_obstructed_speed[1])) != 0)[0])
print(significant_pvals_obstructed_speed[1])
print(np.where(np.diff(np.isnan(significant_clusters_obstructed_speed[2])) != 0)[0])
print(significant_pvals_obstructed_speed[2])


########################
########################
# stop analysis here
pause()
########################
########################








# and now we want to put the trajectories into a mne permutation test
# make two matrices
# first we test whether splitting the group in correct and wrong gives different trajectories
correct_indices = np.where((trajectory_properties[:, 5] == 1) & (trajectory_properties[:, 6] == 1))[0]
incorrect_indices = np.where((trajectory_properties[:, 5] == 1) & (trajectory_properties[:, 6] == 0))[0]

correct_trajectories = speed_to_consensus_start_locked[correct_indices,:]
incorrect_trajectories = speed_to_consensus_start_locked[incorrect_indices,:]

# delete nan values
nan_rows_correct = np.isnan(correct_trajectories).any(axis=1)
correct_trajectories = correct_trajectories[~nan_rows_correct]
nan_rows_incorrect = np.isnan(incorrect_trajectories).any(axis=1)
incorrect_trajectories = incorrect_trajectories[~nan_rows_incorrect]


X = [correct_trajectories, incorrect_trajectories]
print(np.shape(X))

T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(X,
                                                                           out_type="mask",
                                                                           n_permutations=permutations,
                                                                           threshold=threshold,
                                                                           tail=0,
                                                                           )
print(clusters)
times = range(1000)
for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] <= 0.05:
        h = plt.axhline(y= -0.02, xmin=times[c.start]/1000, xmax=times[c.stop - 1]/1000, alpha=0.3, color='blue', linewidth = 4)

# now do the same test with confidence

X = []
for confidence in [1,2,3,4]:
    confidence_indices = np.where((trajectory_properties[:, 5] == 1) & (trajectory_properties[:, 7] == confidence))[0]
    confidence_trajectories = distance_to_consensus_start_locked[confidence_indices, :]

    # delete nan values
    nan_rows_confidence = np.isnan(confidence_trajectories).any(axis=1)
    confidence_trajectories = confidence_trajectories[~nan_rows_confidence]

    X.append(confidence_trajectories)

print(np.shape(X))
T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(X,
                                                                           out_type="mask",
                                                                           n_permutations=permutations,
                                                                           threshold=threshold,
                                                                           tail=0,
                                                                           )
print(clusters)
times = range(1000)

for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] <= 0.05:
        h = plt.axhline(y = -0.06 , xmin = times[c.start]/1000, xmax = times[c.stop - 1]/1000, alpha=0.8, color = 'orange', linewidth = 4)

plt.xlim(0, 10)
#plt.ylim(-0.1,1.1)
plt.tight_layout()  # Adjust spacing between subplots
plt.show()














##############################################################################
#############################################################################





fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Create 3 subplots horizontally

for i, num_players in enumerate([2, 3, 4]):
    for correct in [0, 1]:
        for confidence in [1, 2, 3, 4]:
            confidence_indices = np.where((trajectory_properties[:, 1] == num_players) & (trajectory_properties[:, 5] == 1) & (trajectory_properties[:, 6] == correct) & (trajectory_properties[:, 7] == confidence))[0]
            confidence_average_trajectory = np.nanmean(distance_to_consensus_start_locked[confidence_indices, :], axis=0)

            # Define line color based on 'correct' and 'confidence' values
            if correct == 0:
                if confidence == 1:
                    line_color = 'lightcoral'
                elif confidence == 2:
                    line_color = (1, 0.6, 0.6)  # Lighter shade of red using RGBA values
                elif confidence == 3:
                    line_color = (1, 0.3, 0.3)  # Intermediate shade of red using RGBA values
                elif confidence == 4:
                    line_color = 'darkred'
            else:
                if confidence == 1:
                    line_color = 'lightgreen'
                elif confidence == 2:
                    line_color = (0.6, 1, 0.6)  # Lighter shade of green using RGBA values
                elif confidence == 3:
                    line_color = (0.3, 1, 0.3)  # Intermediate shade of green using RGBA values
                elif confidence == 4:
                    line_color = 'darkgreen'

            axs[i].plot(confidence_average_trajectory, color=line_color)

    axs[i].set_xlabel("Time")
    axs[i].set_ylabel("Average Distance to Consensus")
    axs[i].set_title(f"Num Players: {num_players}")
    axs[i].legend(title="Confidence and Correct")

plt.tight_layout()  # Adjust spacing between subplots
plt.show()








def Moving_average(time_series, num_samples):
    filtered_series = time_series
    half_num = int(num_samples / 2)
    for i in range(0, len(time_series) - 1):
        if i < half_num:
            pre_samples = time_series[0:i]
        else:
            pre_samples = time_series[i - half_num:i]

        if i > len(time_series) - (half_num + 1):
            end_samples = time_series[i:len(time_series) - 1]
        else:
           # print(i + half_num)
            end_samples = time_series[i:i + half_num]

        sample = np.average(np.append(pre_samples, end_samples))
        filtered_series[i] = sample
    return filtered_series

def generate_statistics_matrices(group_sizes,size,type):
    trial_types = group_sizes[size]
    trajectories = trial_types[type]

    merged_trajectories = np.vstack((trajectories.correct_high_conf,trajectories.correct_low_conf,trajectories.wrong_high_conf,trajectories.wrong_low_conf))
    correct_labels = np.concatenate((np.ones((np.size(trajectories.correct_high_conf,0) + np.size(trajectories.correct_low_conf,0),)),np.zeros((np.size(trajectories.wrong_high_conf,0) + np.size(trajectories.wrong_low_conf,0),))),0)
    confidence_labels = np.concatenate((np.ones((np.size(trajectories.correct_high_conf,0),)),np.zeros((np.size(trajectories.correct_low_conf,0),)),np.ones((np.size(trajectories.wrong_high_conf,0),)),np.zeros((np.size(trajectories.wrong_low_conf,0),))))

    labels = np.concatenate(([correct_labels],[confidence_labels]))
    labels = np.transpose(labels)
    return merged_trajectories, labels


#this function has all confidence levels (from 1 to 4)
def generate_statistics_matricesv2(Data_dictionary,size,type):
    merged_trajectories = np.zeros((3000,))
    correct_labels = []
    confidence_labels = []
    for session in Data_dictionary.values():
        if session.PlayerSize == size:
            for trial in session.TrialList:
                if trial.Success == 0:  # only look at correct trials
                    continue
                if len(trial.Players_x[0]) > 3000:  # remove trials that took too long
                    continue
                if trial.Condition == type:
                    trial_correct_labels, trial_confidence_labels, trial_trajectories = Combined_Plots.generate_trajectory_matrices_for_permutation(trial)
                    merged_trajectories = np.vstack((merged_trajectories, trial_trajectories))
                    correct_labels = np.concatenate((correct_labels,trial_correct_labels))
                    confidence_labels = np.concatenate((confidence_labels,trial_confidence_labels))
    merged_trajectories = merged_trajectories[1:,:]

    labels = np.concatenate(([correct_labels],[confidence_labels]))
    labels = np.transpose(labels)
    return merged_trajectories, labels

def create_permutation(labels):
    new_labels = np.zeros_like(labels)
    new_labels[:,0] = np.random.shuffle(labels[:,0])
    new_labels[:,1] = np.random.shuffle(labels[:,1])
    return new_labels

def get_grouped_samples(sample_column, labels):
    corr_high = []
    corr_low = []
    wrong_high = []
    wrong_low = []
    for i in range(len(sample_column)):
        if not np.isnan(sample_column[i]):
            if labels[i,0] == 1 and labels[i,1] == 1:
                corr_high.append(sample_column[i])
            elif labels[i,0] == 1 and labels[i,1] == 0:
                corr_low.append(sample_column[i])
            elif labels[i, 0] == 0 and labels[i, 1] == 1:
                wrong_high.append(sample_column[i])
            elif labels[i, 0] == 0 and labels[i, 1] == 0:
                wrong_low.append(sample_column[i])
    return corr_high, corr_low, wrong_high, wrong_low

def perform_anova_2way(sample_column, labels):

    for i in range(len(sample_column)):
        if i < len(sample_column):
            if np.isnan(sample_column[i]):
                sample_column = np.delete(sample_column, i)
                labels = np.delete(labels,i,0)

    df = pd.DataFrame({'correct': labels[:, 0], 'confidence': labels[:, 1], 'position': sample_column})
    model = ols('position ~ C(correct) + C(confidence) ', data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)

    return table


def get_significant_samples(merged_trajectories, labels):
    #first do anova on correct labels:
    correct_significantsamples = np.zeros((np.size(merged_trajectories,1),))
    confidence_significantsamples = np.zeros((np.size(merged_trajectories,1),))
    interaction_significantsamples = np.zeros((np.size(merged_trajectories,1),))

    correct_significantFvals= np.zeros((np.size(merged_trajectories,1),))
    confidence_significantFvals = np.zeros((np.size(merged_trajectories,1),))
    interaction_significantFvals = np.zeros((np.size(merged_trajectories,1),))

    for sample in range(np.size(merged_trajectories,1)):
        sample_column = merged_trajectories[:,sample]
        corr_high, corr_low, wrong_high, wrong_low = get_grouped_samples(sample_column, labels)
        # if not np.isnan(np.sum(corr_high) + np.sum(corr_low) + np.sum(wrong_high) + np.sum(wrong_low)):
        #F, p = f_oneway(corr_high, corr_low, wrong_high, wrong_low)
        table = perform_anova_2way(sample_column, labels)
        p_correct = table['PR(>F)'][0]
        p_confidence = table['PR(>F)'][1]
        p_interaction = table['PR(>F)'][2]
        # print(p)
        if p_correct < 0.05:
            correct_significantsamples[sample] = 1
            correct_significantFvals[sample] = table['F'][0]
        if p_confidence < 0.05:
            confidence_significantsamples[sample] = 1
            confidence_significantFvals[sample] = table['F'][1]
        if p_interaction < 0.05:
            interaction_significantsamples[sample] = 1
            interaction_significantFvals[sample] = table['F'][2]

    correct_maxcluster = 0
    confidence_maxcluster = 0
    interaction_maxcluster = 0

    correct_tempcluster = 0
    confidence_tempcluster = 0
    interaction_tempcluster = 0

    for t in range(1,np.size(merged_trajectories,1)):
        if (correct_significantsamples[t-1] == 1) & (correct_significantsamples[t] == 1):
            correct_tempcluster+=correct_significantFvals[t]
        else:
            if correct_tempcluster > correct_maxcluster:
                correct_maxcluster = correct_tempcluster
            correct_tempcluster = 0

        if (confidence_significantsamples[t - 1] == 1) & (confidence_significantsamples[t] == 1):
            confidence_tempcluster += confidence_significantFvals[t]
        else:
            if confidence_tempcluster > confidence_maxcluster:
                confidence_maxcluster = confidence_tempcluster
            confidence_tempcluster = 0

        if (interaction_significantsamples[t - 1] == 1) & (interaction_significantsamples[t] == 1):
            interaction_tempcluster += interaction_significantFvals[t]
        else:
            if interaction_tempcluster > interaction_maxcluster:
                interactionmaxcluster = interaction_tempcluster
            interaction_tempcluster = 0


    return  correct_maxcluster, confidence_maxcluster, interaction_maxcluster, correct_significantsamples, confidence_significantsamples, interaction_significantsamples

def create_F_distributions(labels, merged_trajectories):
    correct_distribution = []
    confidence_distribution = []
    interaction_distribution = []
    for i in range(1000):
        print("permutation " + str(i))
        permuted_labels = create_permutation(labels)
        correct_maxcluster, confidence_maxcluster, interaction_maxcluster,_ ,_ ,_ = get_significant_samples(merged_trajectories, permuted_labels )
        correct_distribution.append(correct_maxcluster)
        confidence_distribution.append(confidence_maxcluster)
        interaction_distribution.append(interaction_maxcluster)
    return correct_distribution, confidence_distribution, interaction_distribution

def run_permutation_analysis(labels, merged_trajectories):
    correct_maxcluster, confidence_maxcluster, interaction_maxcluster, correct_significantsamples, confidence_significantsamples, interaction_significantsamples = get_significant_samples(merged_trajectories, labels)
    correct_distribution, confidence_distribution, interaction_distribution = create_F_distributions(labels, merged_trajectories)

    occurrences_more_than_F = np.where(correct_distribution > correct_maxcluster)
    correct_permutation_p_val = np.sum(occurrences_more_than_F)/1000

    occurrences_more_than_F = np.where(confidence_distribution > confidence_maxcluster)
    confidence_permutation_p_val = np.sum(occurrences_more_than_F)/1000

    occurrences_more_than_F = np.where(interaction_distribution > interaction_maxcluster)
    interaction_permutation_p_val = np.sum(occurrences_more_than_F)/1000

    print(correct_permutation_p_val)
    print(confidence_permutation_p_val)
    print(interaction_permutation_p_val)

    # all_trajectories, labels = generate_statistics_matrices(group_sizes,2,1)
# print(labels)
# print(np.shape(labels))




##################################
##################################
# plot the F-values for the different effects in different subplots
fig, axes = plt.subplots(3, 1, figsize=(6, 12))  # Create subplots for each effect
effect_labels = ["Correct", "Confidence", "Correct by Confdience"]

for i, effects in enumerate(["A", "B", "A:B"]):
    ax = axes[i]  # Get the current subplot

    pthresh = 0.05
    f_thresh = f_threshold_mway_rm(n_replications, factor_levels, effects, pthresh)

    tail = 1
    n_permutations = 100
    F_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
        np.swapaxes(data, 0, 1),
        stat_fun=stat_fun,
        threshold=f_thresh,
        tail=tail,
        n_permutations=n_permutations,
        buffer_size=None,
        out_type="mask",
    )

    good_clusters = np.where(cluster_p_values < 0.05)[0]
    print(len(clusters))

    print(len(good_clusters))
    print(good_clusters)
    print(np.squeeze(good_clusters))
    print(clusters[np.squeeze(good_clusters)])
    good_clusters = clusters[np.squeeze(good_clusters)]
    my_range = range(*good_clusters[0].indices(1000))  # Access the first element of the tuple
    print(list(my_range))
    F_obs_plot = F_obs.copy()

    # Create a boolean mask for elements outside the range
    mask = np.ones_like(F_obs_plot, dtype=bool)
    mask[my_range] = False
    # Assign np.nan to elements outside the range
    F_obs_plot[mask] = np.nan

    # Plot the F-values
    ax.plot(F_obs, color="blue", label="Original")
    ax.plot(F_obs_plot, color="orange", label="Modified")

    ax.set_title(effect_labels[i])
    ax.legend()

fig.tight_layout()
plt.show()

##############################"
##############################
# get the number of trajectories of each kind
correct_confidences = trajectory_properties[np.where((trajectory_properties[:, 1] == 4) & (trajectory_properties[:, 5] == 1) & (trajectory_properties[:, 6] == 1)  &  (trajectory_properties[:, 9] == 1))[0],7]
incorrect_confidences = trajectory_properties[np.where((trajectory_properties[:, 1] == 4) & (trajectory_properties[:, 5] == 1) & (trajectory_properties[:, 6] == 1)  & (trajectory_properties[:, 9] == 0))[0],7]


# Create a figure and subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# Plot the histogram for correct confidences
axs[0].hist(correct_confidences, bins=[1, 2, 3, 4, 5], color='blue')
axs[0].set_title('Correct Confidences')
axs[0].set_xticks([1, 2, 3, 4])
axs[0].set_ylim([0, 1000])

# Plot the histogram for incorrect confidences
axs[1].hist(incorrect_confidences, bins=[1, 2, 3, 4, 5], color='red')
axs[1].set_title('Incorrect Confidences')
axs[1].set_xticks([1, 2, 3, 4])
axs[1].set_ylim([0, 1000])
# Adjust spacing between subplots
plt.tight_layout()
for confidence in range(1,5):
    print(len(np.where(incorrect_confidences == confidence)[0]))
    print(len(np.where(correct_confidences == confidence)[0]))
smallest_group = len(np.where(incorrect_confidences == 4)[0])
print(smallest_group)
# Display the plot
plt.show()


##########################
##########################
# backup for cluster permutations without two-way anova

# now make permutation test with the obstruction trajectories
obstructed_indices = np.where((trajectory_properties[:, 1] == 4)  & (trajectory_properties[:, 5] == 1)  & (trajectory_properties[:, 6] == 1) & (trajectory_properties[:, 9] == 1))[0]
unobstructed_indices = np.where((trajectory_properties[:, 1] == 4)  & (trajectory_properties[:, 5] == 1) & (trajectory_properties[:, 6] == 1) & (trajectory_properties[:, 9] == 0))[0]


obstructed_trajectories = distance_to_consensus_start_locked[obstructed_indices,:]
unobstructed_trajectories = distance_to_consensus_start_locked[unobstructed_indices,:]

# delete nan values
nan_rows_obstructed = np.isnan(obstructed_trajectories).any(axis=1)
obstructed_trajectories = obstructed_trajectories[~nan_rows_obstructed]
nan_rows_unobstructed = np.isnan(unobstructed_trajectories).any(axis=1)
unobstructed_trajectories = unobstructed_trajectories[~nan_rows_unobstructed]


X = [obstructed_trajectories, unobstructed_trajectories]
print(np.shape(X))
#threshold =6

T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(X,
                                                                           out_type="mask",
                                                                           n_permutations=100,
                                                                           threshold=threshold,
                                                                           tail=0,
                                                                           )
print(clusters)
times = range(1000)
for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] <= 0.05:
        h = plt.axhline(y= -0.02, xmin=times[c.start]/1000, xmax=times[c.stop - 1]/1000, alpha=0.3, color='blue', linewidth = 4)

# now do the same test with confidence

X = []
for confidence in [1,2,3,4]:
    confidence_indices = np.where((trajectory_properties[:, 1] == 4) & (trajectory_properties[:, 5] == 1) & (trajectory_properties[:,6] == 1) & (trajectory_properties[:, 7] == confidence))[0]
    confidence_trajectories = distance_to_consensus_start_locked[confidence_indices, :]

    # delete nan values
    nan_rows_confidence = np.isnan(confidence_trajectories).any(axis=1)
    confidence_trajectories = confidence_trajectories[~nan_rows_confidence]

    X.append(confidence_trajectories)

print(np.shape(X))
#threshold =3

T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(X,
                                                                           out_type="mask",
                                                                           n_permutations=permutations,
                                                                           threshold=threshold,
                                                                           tail=0,
                                                                           )
print(clusters)
times = range(1000)

for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] <= 0.05:
        h = plt.axhline(y = -0.06 , xmin = times[c.start]/1000, xmax = times[c.stop - 1]/1000, alpha=0.8, color = 'orange', linewidth = 4)

plt.xlim(0, 1000)
#plt.ylim(-0.1,1.1)
plt.tight_layout()  # Adjust spacing between subplots
#plt.show()
plt.savefig('obstructed_distance_clusters.png')
plt.clf()
plt.figure()



##############"
#############
correct_confidences = trajectory_properties[np.where((trajectory_properties[:, 1] == 4) & (trajectory_properties[:, 5] == 1) & (trajectory_properties[:, 6] == 1)  &  (trajectory_properties[:, 9] == 1))[0],7]
incorrect_confidences = trajectory_properties[np.where((trajectory_properties[:, 1] == 4) & (trajectory_properties[:, 5] == 1) & (trajectory_properties[:, 6] == 1)  & (trajectory_properties[:, 9] == 0))[0],7]


# Create a figure and subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# Plot the histogram for correct confidences
axs[0].hist(correct_confidences, bins=[1, 2, 3, 4, 5], color='blue')
axs[0].set_title('Correct Confidences')
axs[0].set_xticks([1, 2, 3, 4])
axs[0].set_ylim([0, 2500])

# Plot the histogram for incorrect confidences
axs[1].hist(incorrect_confidences, bins=[1, 2, 3, 4, 5], color='red')
axs[1].set_title('Incorrect Confidences')
axs[1].set_xticks([1, 2, 3, 4])
axs[1].set_ylim([0, 1000])
# Adjust spacing between subplots
plt.tight_layout()
for confidence in range(1,5):
    print(len(np.where(incorrect_confidences == 4)[0]))
    print(len(np.where(correct_confidences == 4)[0]))
smallest_group = len(np.where(incorrect_confidences == 4)[0])
print(smallest_group)
# Display the plot
plt.show()
# there are 326 values for the smallest group: confidence 4 with incorrect opinions
# try out a two-way anova: