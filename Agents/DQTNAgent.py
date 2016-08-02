"""
Author: Robert Post

"""

import os
import cPickle
import time
import numpy as np
import sys
import AgentMemory


class DQTNAgent(object):
    def __init__(self, rng, functionApproximators, disjointDQN ,epsilonStart, epsilonMin,
                 epsilonDecaySteps, evalEpsilon, numReplayExperiences, experimentDirectory,
                 minExperiencesToTrain, updateFrequency, discountRate, kReturnLength, transferTaskModule):

        self.transferTaskModule         = transferTaskModule
        self.disjointDQN                = disjointDQN
        self.functionApproximators      = functionApproximators
        self.allActionsList             = self.transferTaskModule.getTotalActionsList()
        self.numActions                 = len(self.allActionsList)
        self.numReplayExperiences       = numReplayExperiences
        self.experimentDirectory        = experimentDirectory
        self.minExperiencesToTrain      = minExperiencesToTrain
        self.updateFrequency            = updateFrequency
        self.rng                        = rng
        self.discountRate               = discountRate
        self.kReturnLength              = kReturnLength

        self.numTasks                   = self.transferTaskModule.getNumTasks()
        self.currentValidActions        = self.transferTaskModule.getActionsForCurrentTask()
        self.currentTaskIndex           = self.transferTaskModule.currentTaskIndex

        self.inputState                 = self.functionApproximators[0].inputState
        self.phiLength                  = self.inputState[0]
        self.memoryStateShape           = self.inputState[1:]

        self.agentTrainingMemory        = AgentMemory.AgentMemory(
                                                self.rng, 
                                                self.memoryStateShape, 
                                                self.phiLength, 
                                                self.numReplayExperiences, 
                                                self.discountRate, 
                                                self.numTasks
                                            )

        # set the max size of the evaluation memory to be just big enough to fit a phi. This isn't used for training
        self.agentEvaluationMemory      = AgentMemory.AgentMemory(
                                                self.rng, 
                                                self.memoryStateShape, 
                                                self.phiLength, 
                                                self.phiLength * 2,
                                                self.discountRate, 
                                                self.numTasks)


        self.underEvaluation            = False
        self.episodeStepCounter         = 0
        self.batchCounter               = 0
        self.totalStepCounter           = 0
        self.lossArray                  = np.array([])

        self.lastState                  = None
        self.lastAction                 = None

        self.epsilonStart               = epsilonStart
        self.epsilonMin                 = epsilonMin
        self.epsilonDecaySteps          = epsilonDecaySteps
        self.evalEpsilon                = evalEpsilon
        self.epsilon                    = self.epsilonStart

        if self.epsilonDecaySteps != 0:
            self.epsilonRate = ((self.epsilonStart - self.epsilonMin) / self.epsilonDecaySteps)
        else:
            self.epsilonRate = 0



    def startEpisode(self, observation):
        self.episodeStepCounter  = 0
        self.batchCounter        = 0
        self.lossArray           = np.array([])
        self.currentTaskIndex    = self.transferTaskModule.currentTaskIndex
        self.currentValidActions = self.transferTaskModule.getActionsForCurrentTask()
        self.lastState           = observation

        firstActionIndex         = self.rng.randint(0, len(self.currentValidActions))
        self.lastAction          = self.currentValidActions[firstActionIndex]
        return self.lastAction 


    def endEpisode(self, reward):
        self.episodeStepCounter   += 1

        if not self.underEvaluation:
            self.agentTrainingMemory.addSample(self.lastState, self.lastAction, np.clip(reward, -1, 1), True, self.currentTaskIndex)


    def runTrainingBatch(self):
        nextTaskSampled     = np.random.randint(0, self.numTasks)
        sampledTasksActions = self.transferTaskModule.getActionsForTask(nextTaskSampled)

        experienceBatch = self.agentTrainingMemory.getRandomExperienceBatch(self.functionApproximators[0].batchSize, kReturnLength = self.kReturnLength, taskIndex = nextTaskSampled)
        
        if experienceBatch == None:
            return 0

        batchStates, batchActions, batchRewards, batchGammas, batchNextStates, batchNextActions, batchTerminals = experienceBatch

        if self.disjointDQN:
            experienceStateShape = (self.functionApproximators[0].batchSize, self.phiLength) + self.memoryStateShape
            zeroState = np.zeros(experienceStateShape, dtype='uint8')        

            for i in xrange(len(self.functionApproximators)):
                if i == nextTaskSampled:
                    #Train the network of the next sampled task with its sampled data
                    res = self.functionApproximators[i].trainNetwork(batchStates, batchActions, batchRewards, batchGammas, batchNextStates, batchNextActions, batchTerminals, nextTaskSampled, sampledTasksActions)
                else:
                    #other networks get trained with zeros
                    self.functionApproximators[i].trainNetwork(zeroState, batchActions, batchRewards, batchGammas, zeroState, batchNextActions, batchTerminals, nextTaskSampled, sampledTasksActions)
        else:
            res = self.functionApproximators[0].trainNetwork(batchStates, batchActions, batchRewards, batchGammas, batchNextStates, batchNextActions, batchTerminals, nextTaskSampled, sampledTasksActions)


        return res


    def chooseAction(self, epsilon, currentPhi):
        if self.episodeStepCounter < self.phiLength or self.rng.rand() < epsilon:
            actionIndex = self.rng.randint(0, len(self.currentValidActions))
            action = self.currentValidActions[actionIndex]
        else:
            if self.disjointDQN:
                approximator = self.functionApproximators[self.currentTaskIndex]
            else:
                approximator = self.functionApproximators[0]

            qValues = approximator.computeQValues(currentPhi, self.currentTaskIndex)
            reducedQValues = [qValues[i] for i in self.currentValidActions]
            bestActionIndex = np.argmax(reducedQValues)
            action = self.currentValidActions[bestActionIndex]

        return action


    def step(self, reward, observation):
        self.episodeStepCounter += 1
        self.totalStepCounter   +=1

        isTerminal = False
        currentPhi = None
        reward = np.clip(reward, -1, 1)

        if self.underEvaluation:
            self.agentEvaluationMemory.addSample(self.lastState, self.lastAction, reward, isTerminal, self.currentTaskIndex)
            currentPhi = self.agentEvaluationMemory.getCurrentPhi(observation)
            epsilonToUse = self.evalEpsilon
        else:
            self.agentTrainingMemory.addSample(self.lastState, self.lastAction, reward, isTerminal, self.currentTaskIndex)
            currentPhi = self.agentTrainingMemory.getCurrentPhi(observation)
            if len(self.agentTrainingMemory) > self.minExperiencesToTrain:
                self.epsilon = max(self.epsilonMin, self.epsilon - self.epsilonRate)
            
            epsilonToUse = self.epsilon

        actionIndex = self.chooseAction(epsilonToUse,currentPhi)


        if not self.underEvaluation and len(self.agentTrainingMemory) > self.minExperiencesToTrain and self.episodeStepCounter % self.updateFrequency == 0:
            loss = self.runTrainingBatch()
            self.batchCounter += 1
            self.lossArray = np.append(self.lossArray, loss)

        self.lastAction = actionIndex
        self.lastState  = observation

        return self.allActionsList [ actionIndex ]
