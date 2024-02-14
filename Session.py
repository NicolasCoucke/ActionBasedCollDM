################
# script to create data objects
# to be used in 'main.py'
################

import numpy as np
from Trial import TrialClass
import matplotlib
import matplotlib.pyplot as plt
import type_2_ROC
matplotlib.use('Agg')


class SessionClass:

    def __init__(self, session):
        self.Session_number = session
        self.PlayerSize = 2
        self.Condition_order = 0
        self.Trial_conditions = []
        self.Trial_numbers = []
        self.Trial_successes = []
        self.Trial_completion_times = []
        self.TrialList = []
        self.Trial_difficulties = []
        self.Trial_accuracies = []
        self.Trial_MCS_accuracies = []
        self.PuckTrial_accuracies = []
        self.Player_matrix_defined = False
        self.psychometric_input = []
        self.Puck_psychometric_input = []
        self.Puck_trial_correct_levels = []
        self.trial_correct_levels = []
        self.Puck_trial_total_levels = []
        self.trial_total_levels = []
        self.guidance_matrix = []
        self.Puck_guidance_matrix = []

        self.Average_influence = []
        self.Puck_Average_influence = []
        self.Average_confidences = []
        self.Puck_Average_confidences = []
        self.Number_discarded_trials = []
        self.Puck_Number_discarded_trials = []

        self.auroc2 = []
        self.Puck_auroc2 = []

        self.invalid_answers = []
        self.different_from_MCS = 0

    def Generate_trial_info(self, TrialList):
        self.Trial_info = np.zeros((len(self.TrialList),4))
        self.Puck_trial_correct_levels = np.zeros((4,))
        self.trial_correct_levels = np.zeros((4,))
        self.Puck_trial_total_levels = np.zeros((4,))
        self.trial_total_levels = np.zeros((4,))

        t=0
        for trial in TrialList:
            self.Trial_conditions.append(trial.Condition)
            self.Trial_numbers.append(trial.TrialNumber)
            self.Trial_successes.append(trial.Success)
            self.Trial_completion_times.append(trial.CompletionTime)
            self.Trial_difficulties.append(trial.Difficulty)
            if trial.Condition == 1:
                self.trial_total_levels[trial.Difficulty - 4] += 1
                if trial.Success == True:
                    self.trial_correct_levels[trial.Difficulty - 4] += 1
            else:
                self.Puck_trial_total_levels[trial.Difficulty - 4] += 1
                if trial.Success == True:
                    self.Puck_trial_correct_levels[trial.Difficulty - 4] += 1
         #   print(self.Puck_trial_correct_levels)
           # print(self.Puck_trial_total_levels)
           # print(self.trial_correct_levels)
           # print(self.trial_total_levels)

            if self.Trial_conditions[0] == 1:
                self.Condition_order = 1
            elif self.Trial_conditions[0] == 2:
                self.Condition_order = 2
            print(self.Condition_order)

        # get guiding per confidence levels
        Puck_guidance_counter_matrix = np.zeros((2, 4))
        Trial_guidance_counter_matrix = np.zeros((2, 4))
        Puck_guidance_total_matrix = np.zeros((2, 4))
        Trial_guidance_total_matrix = np.zeros((2, 4))
        # get average player confidences
        confidence_summer = np.zeros((self.PlayerSize,))
        confidence_counter = np.zeros((self.PlayerSize,))
        invalid_counter = np.zeros((self.PlayerSize,))

        Puck_confidence_summer = np.zeros((self.PlayerSize,))
        Puck_confidence_counter = np.zeros((self.PlayerSize,))
        Puck_invalid_counter = np.zeros((self.PlayerSize,))

        influence_summer = np.zeros((self.PlayerSize,))
        influence_counter = np.zeros((self.PlayerSize,))

        Puck_influence_summer = np.zeros((self.PlayerSize,))
        Puck_influence_counter = np.zeros((self.PlayerSize,))

        for trial in self.TrialList:
            for p in range(self.PlayerSize):
                if not ((trial.Confidences[p] > 0) & (trial.Confidences[p] < 5)) :
                    if trial.Condition == 1:
                        invalid_counter[p] =+1
                    else:
                        Puck_invalid_counter[p] = +1
                    break
                else:
                    conf = trial.Confidences[p]
                    if trial.Condition == 1:
                        confidence_summer[p]+= conf
                        confidence_counter[p]+=1
                    else:
                        Puck_confidence_summer[p] += conf
                        Puck_confidence_counter[p]+= 1


                if trial.Answers[p] == True:
                    cor = 1
                else:
                    cor = 0
                if trial.Condition == 1:
                    Trial_guidance_total_matrix[cor, conf-1] += 1
                    if (trial.Guiding[p] == 1):
                        Trial_guidance_counter_matrix[cor, conf-1] += 1
                        influence_summer[p]+=1
                    influence_counter[p]+=1
                elif trial.Condition == 2:
                    Puck_guidance_total_matrix[cor, conf-1] += 1
                    if (trial.Guiding[p] == 1):
                        Puck_guidance_counter_matrix[cor, conf-1] += 1
                        Puck_influence_summer[p] += 1
                    Puck_influence_counter[p] += 1

        self.guidance_matrix = Trial_guidance_counter_matrix / Trial_guidance_total_matrix
        self.Puck_guidance_matrix = Puck_guidance_counter_matrix / Puck_guidance_total_matrix
        #print(self.guidance_matrix)
        #print(self.Puck_guidance_matrix)

        self.Average_influence = influence_summer/influence_counter
        self.Puck_Average_influence = Puck_influence_summer/Puck_influence_counter
        self.Average_confidences = confidence_summer/confidence_counter
        self.Puck_Average_confidences = Puck_confidence_summer/Puck_confidence_counter
        self.Number_discarded_trials = invalid_counter
        self.Puck_Number_discarded_trials = Puck_invalid_counter
        #print(self.Average_influence)
        #print(self.Average_confidences)


        #get auroc2 for participants
        self.auroc2 = np.zeros((self.PlayerSize))
        self.Puck_auroc2 = np.zeros((self.PlayerSize))
        for p in range(self.PlayerSize):
            player_correct = []
            player_confidences = []
            Puck_player_correct = []
            Puck_player_confidences = []
            #get data needed for auroc
            for trial in self.TrialList:
                if not ((trial.Confidences[p] > 0) & (trial.Confidences[p] < 5)):
                    continue
                if trial.Condition == 1:
                    player_confidences.append(trial.Confidences[p])
                    if  trial.Answers[p] == True:
                        player_correct.append(1)
                    else:
                        player_correct.append(0)
                else:
                    Puck_player_confidences.append(trial.Confidences[p])
                    if trial.Answers[p] == True:
                        Puck_player_correct.append(1)
                    else:
                        Puck_player_correct.append(0)

            #calculate auroc
            #print(player_correct)
            #print(player_confidences)
            player_auroc = type_2_ROC.type2roc(np.array(player_correct), np.array(player_confidences), 4)
            Puck_player_auroc = type_2_ROC.type2roc(Puck_player_correct, Puck_player_confidences, 4)
            self.auroc2[p] = player_auroc
            self.Puck_auroc2[p] = Puck_player_auroc
        #print(self.auroc2)
        #print(self.Puck_auroc2)



    def Session_statistics(self):

            # get number of invalid answers per participant
            self.invalid_answers = np.zeros((self.PlayerSize,))

            #get accuracies per difficulty level
            Puck_counter_matrix = np.zeros((4,))
            Trial_counter_matrix = np.zeros((4,))
            for trial in self.TrialList:
                if trial.Condition == 1:

                    detect_invalid = False
                    # find trials with invalid answers
                    for p in range(self.PlayerSize):
                        if (trial.ActualAnswers[p] not in [1, 2, 3, 4]) or (trial.Confidences[p] not in [1, 2, 3, 4]):
                            self.invalid_answers[p] += 1
                            detect_invalid = True

                    # if ther is an invalid answer then do not take this trial into account
                    if detect_invalid == True:
                        continue

                    Trial_counter_matrix[trial.Difficulty - 1]+=1
                    #group guess
                    if trial.Success == True:
                        self.Trial_accuracies[0, trial.Difficulty-1] +=1
                    for i in range(len(trial.Answers)):
                        if trial.Answers[i] == True:
                            self.Trial_accuracies[i+1,trial.Difficulty-1] += 1

                    # maximal confidence heursitic guess
                    if trial.MCS_Success == True:
                        self.Trial_MCS_accuracies[trial.Difficulty-1] += 1

                    if trial.MCS_chosen == False:
                        self.different_from_MCS +=1


                if trial.Condition == 2:
                    Puck_counter_matrix[trial.Difficulty - 1]+=1
                    #group guess
                    if trial.Success == True:
                        self.PuckTrial_accuracies[0, trial.Difficulty-1] +=1
                    for i in range(len(trial.Answers)):
                        if trial.Answers[i] == True:
                            self.PuckTrial_accuracies[i+1,trial.Difficulty-1] += 1




            #print(self.Trial_accuracies)
            #print(Trial_counter_matrix)
            for i in range(self.PlayerSize+1):
                self.Trial_accuracies[i,:] = self.Trial_accuracies[i,:]/Trial_counter_matrix
                self.PuckTrial_accuracies[i,:] = self.PuckTrial_accuracies[i,:] / Puck_counter_matrix
            self.Trial_MCS_accuracies = self.Trial_MCS_accuracies/Trial_counter_matrix


    def DefinePlayerMatrix(self, PlayerSize):
        self.Trial_accuracies = np.zeros((PlayerSize+1,4))
        self.PuckTrial_accuracies = np.zeros((PlayerSize+1,4))
        self.Player_matrix_defined = True
        self.Trial_MCS_accuracies = np.zeros((4,))

    def ParseSession(self, data):
        index = 0
        #loop through file and create a new trial when you find one
        while (index < data.size):
            string = data[index]
            if "Trial" in string:
                if self.Player_matrix_defined == False:
                    if "A4" in data[index+1]:
                        self.PlayerSize = 4
                    elif "A3" in data[index+1]:
                        self.PlayerSize = 3
                    else:
                        self.PlayerSize = 2
                    self.DefinePlayerMatrix(self.PlayerSize)
                trial = TrialClass(self.PlayerSize) #creates new trial object
                index = trial.ParseTrial(data, index) #extracts trial info and puts in that object
                trial.cut_off_zero_parts()
                trial.get_guiding()
                #trial.Generate_One_Trial_Plot(trial)

                #ConfidenceMeasures.get_first_distance(trial)
                #ConfidenceMeasures.get_AUC(trial)
                #trial.Generate_One_Trial_Plot(trial)
                self.TrialList.append(trial) #add the trial object to the session object
                if trial.TrialNumber == 2:
                    STOP

            else:
                index+=1
        self.Generate_trial_info(self.TrialList)


    def PlotTrialResults(self):
        #fig = plt.figure((4,6))
        plt.title('session ' + str(self.Session_number))
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11, 4))
        ax1.set_title('Separate avatars')
        ax1.plot(self.Trial_accuracies[0,:], linewidth = 4, alpha = 0.6)
        ax1.set_ylim([0,1])
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('difficulty')

        #ax1.set_xticks([0, 1, 2, 3],["1", "2", "3", "4"])
        #ax1.xticks([0, 1, 2, 3], [1, 2, 3, 4])

       # print(self.PlayerSize)
        for i in range(self.PlayerSize):
            ax1.plot(self.Trial_accuracies[i+1,:])

        ax2.set_title('Combined avatar')
        ax2.plot(self.PuckTrial_accuracies[0, :], linewidth=5, alpha = 0.6)
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('accuracy')
        ax2.set_xlabel('difficulty')

        #print(self.PlayerSize)

        for i in range(self.PlayerSize):
            ax2.plot(self.PuckTrial_accuracies[i + 1, :])

        #ax2.set_xticks([0, 1, 2, 3],["1", "2", "3", "4"])
        ax2.legend(["Group", "Player 1", "Player 2", "Player 3", "Player 4"])

        # plt.show()
        plt.savefig('session' + str(self.Session_number), dpi='figure', format=None, metadata=None,
                bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)