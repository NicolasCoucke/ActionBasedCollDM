#########
# Script to calculate the type 2 ROC as a measure of metacognitive sensitivity
# Based on Fleming & Lau (2014)
#########

import numpy as np


def type2roc(correct, conf, Nratings):
    #correct: vector of tirals (0 for error, 1 for correct)
    #conf: vector of ntrials with confratings
    #Nratings: confidence levels available

    i = Nratings+1

    H2 = np.zeros((i,))
    FA2 = np.zeros((i,))
    for c in range(Nratings):
        for it in range(len(conf)):
            if (conf[it] == c+1) and correct[it]:
                H2[i - 2]+=1
            if (conf[it] == c+1) and not correct[it]:
                FA2[i - 2]+=1
        H2[i-2]+= 0.5
        FA2[i-2]+= 0.5
        i-=1

    H2 = H2/np.sum(H2)
    FA2 = FA2/np.sum(FA2)
    cum_H2 = np.pad(np.cumsum(H2), (1, 0), 'constant')
    cum_FA2 = np.pad(np.cumsum(FA2), (1, 0), 'constant')

    k = []
    for c in range(Nratings):
        k.append(np.square(cum_H2[c+1] - cum_FA2[c]) - np.square(cum_H2[c] - cum_FA2[c+1]))

    auroc2 = 0.5 + 0.25 * np.sum(k)

    return auroc2