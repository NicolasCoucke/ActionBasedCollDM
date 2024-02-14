######
# Script to create trial objects as part of processed data dictionary
# Used by Session.py script
######


import numpy as np
import matplotlib.pyplot as plt
class TrialClass:

    def __init__(self, _PlayerSize):
        self.TrialNumber = 0
        self.Condition = 0
        self.Difficulty = 0
        self.CorrectLocation = 0
        self.ConsensusLocation = 0
        self.Success = False
        self.MCS_chosen = False

        self.MCS_Success = False

        self.CompletionTime = 0
        self.time = []
        self.PlayerSize = _PlayerSize

        self.Players_x= []
        self.Players_y= []

        self.Puck_x= []
        self.Puck_y= []

        self.ActualAnswers = []
        self.Answers = []
        self.Confidences = []

        self.Guiding = []


        while len(self.Answers) < self.PlayerSize:
            self.Answers.append(-1)
        while len(self.ActualAnswers) < self.PlayerSize:
            self.ActualAnswers.append(-1)
        while len(self.Guiding) < self.PlayerSize:
            self.Guiding.append(-1)
        while len(self.Confidences) < self.PlayerSize:
            self.Confidences.append(-1)

    def DefineCircles(self, correct, ax):
        circleDiams = [12, 13, 14, 15]
        CirclesPositions = np.array([[-4, 4], [4, 4], [-4, -4], [4, -4]])
        for circle in range(4):
            if circle == correct:
                color = 'g'
                radius = 1.2
            else:
                color = 'r'
                radius = 1.2
            circle1 = plt.Circle((CirclesPositions[circle]), radius, color=color, fill=False)
            ax.add_patch(circle1)

    def get_game_time(self, cell):
        self.time.append(float(cell))

    def get_next_cell(self, string, begin_comma_index):
        # finds the next comma in line
        end_comma_index = string.find(",", begin_comma_index + 1)
        if end_comma_index == -1: #if you don't find comma's anymore then you're at the end of the string
            end_comma_index = len(string)
        cell = string[begin_comma_index + 1:end_comma_index]
        return cell, end_comma_index


    def get_position_player(self, string, start_position, playerNumber):
        if len(self.Players_x) < playerNumber:
            player_x = []
            player_y = []
            self.Players_x.append(player_x)
            self.Players_y.append(player_y)
        position = string[string.find("(", start_position) + 1:string.find(")", start_position)]
        posindex = position.find(",")

        Xpos = float(position[0:posindex])
        Ypos = float(position[posindex + 1:])

        self.Players_x[playerNumber-1].append(Xpos)
        self.Players_y[playerNumber-1].append(Ypos)
        end_position = string.find(")", start_position + 1) + 1

        return end_position

    def get_position_puck(self, string, start_position):
        position = string[string.find("(", start_position) + 1:string.find(")", start_position)]
        posindex = position.find(",")
        Xpos = float(position[0:posindex])
        Ypos = float(position[posindex + 1:])
        self.Puck_x.append(Xpos)
        self.Puck_y.append(Ypos)
        end_position = string.find(")", start_position + 1) + 1
        return end_position

    def get_consensus_location(self):
        print('get consensus')
        DistToLocation = np.zeros((4,))
        CirclesPositions = np.array([[-4, 4], [4, 4], [-4, -4], [4, -4]])
        for i in range(4):
            circle_x = CirclesPositions[i][0]
            circle_z = CirclesPositions[i][1]
            eucl_distance = np.sqrt(np.square(self.Players_x[0][-1] - circle_x) + np.square(
                self.Players_y[0][-1]  - circle_z))
            DistToLocation[i] =  eucl_distance
        self.ConsensusLocation = np.argmin(DistToLocation)+1

    def get_guiding(self,):
        self.get_consensus_location()
      #  print("consensusloc " + str(trial.ConsensusLocation))
        #print("correctLoc " + str(trial.CorrectLocation))

        for i in range(len(self.Answers)):
         #  print("answer " + str(trial.ActualAnswers[i]))
           #print("cons " + str(trial.ConsensusLocation))
           if self.ActualAnswers[i] == self.ConsensusLocation:
                self.Guiding[i] = 1
           elif self.ActualAnswers[i] == -1:
                self.Guiding[i] = -1
           else:
                self.Guiding[i] = 0
        #   print("guide " + str(i+1) + " " + str(trial.Guiding[i]))


    def Generate_One_Trial_Plot(self,trial):
        CirclesPositions = np.array([[-4, 4], [4, 4], [-4, -4], [4, -4]])
        circle_x = CirclesPositions[trial.CorrectLocation - 1][0]
        circle_z = CirclesPositions[trial.CorrectLocation - 1][1]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        for playernumber in range(len(trial.Players_x)):
            # print(playernumber)
            ax1.plot(trial.Players_x[playernumber], trial.Players_y[playernumber])
            eucl_distance = np.sqrt(np.square(trial.Players_x[playernumber] - circle_x * np.ones(
                len(trial.Players_x[playernumber]), )) + np.square(
                trial.Players_y[playernumber] - circle_z * np.ones(len(trial.Players_y[playernumber]), )))
            ax2.plot(range(len(eucl_distance)), eucl_distance)

        if trial.Condition == 2:
            ax1.plot(trial.Puck_x, trial.Puck_y, linewidth=8, alpha=0.5, color='gray')

        trial.DefineCircles(trial.CorrectLocation - 1, ax1)
        ax1.set_xlim([-5, 5])
        ax1.set_ylim([-5, 5])
        ax2.set_ylim([0, 12])
        # if trial.Success:
        #     plt.title("correct")
        # else:
        #     plt.title("wrong")
        if trial.Condition == 1:
            plt.title("Trial " + str(trial.TrialNumber) + " Seperate " + str(trial.Success))
        else:
            plt.title("Trial " + str(trial.TrialNumber) + " Combined " + str(trial.Success))
        if len(trial.Answers) == 2:
            plt.legend(["player 1 " + "A " + str(trial.Answers[0]) + " C " + str(trial.Confidences[0]),
                        "player 2 " + "A " + str(
                            trial.Answers[1]) + " C " + str(
                            trial.Confidences[1])] )
        if len(trial.Answers) == 3:
            plt.legend(["player 1 " + "A " + str(trial.Answers[0]) + " C " + str(trial.Confidences[0]),
                        "player 2 " + "A " + str(
                            trial.Answers[1]) + " C " + str(
                            trial.Confidences[1]), "player 3 " + "A " + str(
                    trial.Answers[2]) + " C " + str(
                    trial.Confidences[2]) ])
        if len(trial.Answers) == 4:
            plt.legend(["player 1 " + "A " + str(trial.Answers[0]) + " C " + str(trial.Confidences[0]),
                        "player 2 " + "A " + str(
                            trial.Answers[1]) + " C " + str(
                            trial.Confidences[1]), "player 3 " + "A " + str(
                    trial.Answers[2]) + " C " + str(
                    trial.Confidences[2]), "player 4 " + "A " + str(
                    trial.Answers[3]) + " C " + str(
                    trial.Confidences[3])])

        plt.show()



    trial = 0

    time = []
    Player_1_x = []
    Player_1_y = []

    Player_2_x = []
    Player_2_y = []



    def get_MCS_success(self):

        # get the summed confidence associated with each answer
        answer_confidences = []
        for location in range(4):
            location_confidence = 0
            for p in range(len(self.ActualAnswers)):
                if self.ActualAnswers[p] == location+1:
                    if not np.isnan(self.Confidences[p]):
                        location_confidence += self.Confidences[p]
            answer_confidences.append(location_confidence)


        # check if the answer chosen with most summed confidence is the one chosen
        max_answer_confidences = np.argmax(answer_confidences)+1
        if max_answer_confidences == self.CorrectLocation:
            self.MCS_Success = True


        self.get_consensus_location()
        if max_answer_confidences == self.ConsensusLocation:
            self.MCS_chosen = True






    def ParseTrial(self, data, startindex):
        #gets all the data from trial
        index = startindex
        string = data[index]
       # check which condition (puck or not)
        if "Puck" in string:
            self.Condition = 2
        else:
            self.Condition = 1
        # get the trial number, difficulty and correct position
        self.TrialNumber = string.split(",")[0].split("l ")[1]
        strindex = string.find(",", 0)
        cell, strindex = self.get_next_cell(string, strindex)

        self.CorrectLocation = int(cell)


        #print("correct" + str(self.CorrectLocation))
        #cell, strindex = self.get_next_cell(string, strindex)
        self.Difficulty = int(string[strindex+1:])
        #print(" diff " + str(self.Difficulty))
        #self.CorrectLocation = cell
        index+=1 #skip to the next line and get the answers from the keypads
        string = data[index]
        cell, strindex = self.get_next_cell(string, 0)
        cell = string[0:strindex]
        while cell.find(":") != -1 :
            try:
                reply = int(cell[cell.find(":")+2])
            except:
                reply = np.nan #-1
            #print(reply)
            if "A" in cell:
                if self.CorrectLocation == reply:
                    self.Answers[int(cell[cell.find("A")+1])-1] = True
                else:
                    self.Answers[int(cell[cell.find("A") + 1]) - 1] = False
                self.ActualAnswers[int(cell[cell.find("A")+1])-1] = reply
                #self.Answers.append(reply)
            elif "C" in cell:
                self.Confidences[int(cell[cell.find("C")+1])-1] = reply
                #self.Confidences.append(reply)
            if strindex == -1:
                break
            cell, strindex = self.get_next_cell(string, strindex)
        #start getting the positiondata while checking for the end of the trial; if the end is reached then check if it was succesfull or not and how long it took
        index+=1
        while (index < data.size):  # float(data[index+1][0]) >= float(data[index][0])):
            string = data[index]
            if "WRONG" in string:
                self.Success = False
                self.CompletionTime = max(self.time)
                while ("Trial" not in string and index+1 < data.size): #move until next trial
                    index += 1
                    string = data[index]
                self.get_MCS_success()
                return index
            elif "CORRECT" in string:
                self.Success = True
                self.CompletionTime = max(self.time)
                while("Trial" not in string and index+1 < data.size): #make sure it's a real succes and not one that precedes a failure
                    index+=1
                    string = data[index]
                self.get_MCS_success()
                return index
            else:
                strindex = 0
                strindex = string.find(",", 0)
                # fix here (crashes because only a zero in the file)
                cell = string[0:strindex]
                self.get_game_time(cell)
                #cell, strindex = self.get_next_cell(string, strindex)

                if self.Condition == 2:
                    strindex = self.get_position_puck(string, strindex)
                playerNumber = 1

                while string.find("(", strindex) != -1: #as long as you still find positions
                   # print("posiiton of " + str(playerNumber))
                    strindex = self.get_position_player(string, strindex, playerNumber)
                    playerNumber+=1


                index += 1
        return index



    def cut_off_zero_parts(self):
        all_player_t = []
        for p in range(len(self.Players_x)):
            t = 0
            while (self.Players_x[p][t] == 0) & (self.Players_x[p][t] == 0):
                t+=1
            all_player_t.append(t)
        first_t = np.max(all_player_t)
        for p in range(len(self.Players_x)):
            cutof_x = self.Players_x[p][first_t:]
            self.Players_x[p] = cutof_x
            cutof_y = self.Players_y[p][first_t:]
            self.Players_y[p] = cutof_y
        new_CompletionTime = self.CompletionTime - self.time[first_t]
        self.CompletionTime = new_CompletionTime









