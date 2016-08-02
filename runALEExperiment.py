'''
 Author: Robert Post
'''
import sys

# sys.path.append("Utilities/")
# sys.path.append("FunctionApproximation/")
# sys.path.append("Agents/")
# sys.path.append("FunctionApproximation/TransferArchitecture")
                                       


import logging
import os
import argparse
import ale_python_interface
import cPickle
import numpy as np
import theano
import lasagne
import cv2
import time

from Utilities import RLParameters
from Utilities import ALEEnvironment
from Utilities import TransferTaskModule
from FunctionApproximation import DeepQTransferNetwork
from FunctionApproximation import DeepNetworks
from Agents import DQTNAgent


class ALEExperiment(object):
    def __init__(self, parameters):
        self.parameters = parameters
        
        logging.shutdown()
        rootlog = logging.getLogger()
        rootlog.handlers = []

        if self.parameters["logToFile"] == True:
            logging.basicConfig(filename = self.parameters["experimentDirectory"] + "/experiment.log", format='%(levelname)s:\t%(message)s', level=logging.INFO)
        else:
            logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.INFO)

        sys.setrecursionlimit(10000)
        self.rng = np.random.RandomState(self.parameters["seed"])
        lasagne.random.set_rng(self.rng)


        self.ale = ale_python_interface.ALEInterface()
        ALEEnvironment.initializeALEParameters(
            self.ale, 
            self.parameters["seed"], 
            self.parameters["repeatActionProbability"],
            self.parameters["displayScreen"] 
            )


        self.transferTaskModule = TransferTaskModule.TransferTaskModule(
            self.ale, 
            parameters["rom"],
            parameters["baseRomPath"],
            parameters["actionSet"],
            parameters["romDelimiter"],
            parameters["romPartsDelimiter"],
            parameters["flavorTaskDelimiter"],
            parameters["flavorOptionDelimiter"]
        )

        self.allActionsList = self.transferTaskModule.getTotalActionsList()
        self.numTotalActions = len(self.allActionsList)
        self.numTasks = self.transferTaskModule.getNumTasks()
        self.stateWidth, self.stateHeight = self.ale.getScreenDims()
        self.bufferCount = 0
        self.screenBuffer = np.zeros((2, self.stateHeight, self.stateWidth),dtype=np.uint8)
        self.lifeLost = False
        self.inputState = (self.parameters["phiLength"], self.parameters["resizedHeight"], self.parameters["resizedWidth"])

        logging.info( "Actions List:" + str(self.allActionsList))
        logging.info( "Number of total tasks:" + str(self.numTasks) + " across " + str(self.transferTaskModule.getNumGames()) + " games.")

        self.learningFilename = self.parameters["experimentDirectory"] + "/learning.csv"
        self.resultsFilenames = []

        if not os.path.exists(self.learningFilename):   
            with open(self.learningFilename, "a") as learningFile:
                learningFile.write('mean_loss,epsilon\n')

        #Create a results file for every task.
        experimentOKFlag = True
        for i in range(self.numTasks):
            taskFilename = self.parameters["experimentDirectory"] + '/task_' + str(i)+'_results.csv'
            self.resultsFilenames.append(taskFilename)

            if os.path.exists(taskFilename) and self.parameters['startingEpoch'] > 0:
                
                with open(taskFilename, "r") as taskResultFile:
                    fileContents = taskResultFile.readlines()

                    if len(fileContents) == 1 and self.parameters['startingEpoch'] > 0:
                        experimentOKFlag = False
                        continue

                    lastLine = fileContents[-1]
                    lastEvalEpoch = int(lastLine.split(',')[0].strip())
                    if lastEvalEpoch != self.parameters['startingEpoch']:
                        #The program ended in the middle of an experiment
                        #somewhere between creating the network and finishing the evaluation of all tasks
                        #When this happens delete the last network file and remove the evaluation lines that corresponded to that epoch 
                        experimentOKFlag = False
                
            else:
                taskResultsFile = open(taskFilename, 'a', 0)
                epochFormated = "{0:7}".format("Epoch,")
                numEpisodesFormated = "{0:14}".format("Num Episodes,")
                totalRewardFormated = "{0:14}".format("Total Reward,")
                averageRewardFormatted = "{0:16}".format("Average Reward,")
                rewardStdFormatted = "{0:16}".format("Reward Std,")
                meanQValFormatted = "{0:20}".format("Mean best Q Value")
                resultsHeader = epochFormated + numEpisodesFormated + totalRewardFormated + averageRewardFormatted + rewardStdFormatted + meanQValFormatted + "\n"
                taskResultsFile.write(resultsHeader)
                taskResultsFile.flush()
                taskResultsFile.close()


        if not experimentOKFlag and self.parameters['startingEpoch'] > 0:
            os.remove(self.parameters['nnFile'])
            self.parameters['nnFile'] = self.parameters['nnFile'].replace(str(self.parameters['startingEpoch']), str(self.parameters['startingEpoch'] - 1))
            self.parameters['startingEpoch'] = self.parameters['startingEpoch'] - 1
            for taskFilename in self.resultsFilenames:
                taskResultFile = open(taskFilename, "r")
                fileContents = taskResultFile.readlines()

                if len(fileContents) == 1:
                    continue

                lastLine = fileContents[-1]
                lastEvalEpoch = int(lastLine.split(',')[0].strip())
                taskResultFile.close()
                #If there is a missmatch between the last evaluated epoch number and the network file the experiment ended in a weird state
                #Either during network file creation and the file is broken or between evaluating tasks 
                #either way remove it above then remove the recorded evaluation results from the tasks that did an evaluation at that epoch 
                if lastEvalEpoch == self.parameters['startingEpoch'] + 1:
                    taskResultFile = open(taskFilename, "w")
                    taskResultFile.write(str("".join(fileContents[0:-1])))
                    taskResultFile.close()

        functionApproximators = []
        for i in xrange(self.numTasks):

            functionApproximator = DeepQTransferNetwork.DeepQTransferNetwork(
                self.rng, 
                self.numTotalActions,
                self.parameters["batchSize"], 
                self.inputState,
                self.parameters["discountRate"], 
                self.parameters["learningRate"], 
                self.parameters["rmsRho"], 
                self.parameters["rmsEpsilon"], 
                self.parameters["momentum"], 
                self.parameters["networkUpdateDelay"], 
                self.parameters["useSARSAUpdate"], 
                self.parameters["kReturnLength"],
                self.parameters["transferExperimentType"] , 
                self.numTasks, 
                self.parameters["networkType"],
                clipDelta = self.parameters["clipDelta"]
            )

            functionApproximators.append(functionApproximator)

            if self.parameters["disjointDQN"] == False:
                break


        if len(self.parameters["nnFile"]) > 0:
            #Load the network data
            paramArray = [f.qValueNetwork for f in functionApproximators]
            DeepNetworks.loadNetworkParams(paramArray, self.parameters["nnFile"], self.parameters["loadWeightsFlipped"])
            functionApproximator.resetNextQValueNetwork()
            logging.info("Loaded network file: {0}".format(self.parameters["nnFile"]))


        self.agent = DQTNAgent.DQTNAgent(self.rng,functionApproximators,
                                  self.parameters["disjointDQN"],
                                  self.parameters["epsilonStart"],
                                  self.parameters["epsilonEnd"],
                                  self.parameters["epsilonDecaySteps"],
                                  self.parameters["evalEpsilon"],
                                  self.parameters["replayMemorySize"],
                                  self.parameters["experimentDirectory"],
                                  self.parameters["replayStartSize"],
                                  self.parameters["updateFrequency"],
                                  self.parameters["discountRate"],
                                  self.parameters["kReturnLength"],
                                  self.transferTaskModule
                                  )

        if self.parameters["startingEpoch"] == 0:
            logging.info("Saving initialization parameters to network file 0")

            paramArray = [f.qValueNetwork for f in self.agent.functionApproximators]
            DeepNetworks.saveNetworkParams(paramArray, self.parameters["experimentDirectory"] + '/network_0.pkl')
            # DeepNetworks.saveNetworkParams(self.agent.functionApproximator.qValueNetwork, self.parameters["experimentDirectory"] + '/network_0.pkl')




    def run(self):
        for epoch in range(self.parameters["startingEpoch"] + 1, self.parameters["numEpochs"] + 1):
            self.runTrainingEpoch(epoch, self.parameters["stepsPerTrainingEpoch"], self.parameters["episodesPerTrainingEpoch"] )
            
            paramArray = [f.qValueNetwork for f in self.agent.functionApproximators]
            DeepNetworks.saveNetworkParams(paramArray, self.parameters["experimentDirectory"] + '/network_' + str(epoch) + '.pkl')
            # DeepNetworks.saveNetworkParams(self.agent.functionApproximator.qValueNetwork, self.parameters["experimentDirectory"] + '/network_' + str(epoch) + '.pkl')

            if (self.parameters["stepsPerEvaluationEpoch"] > 0 or self.parameters["episodesPerEvaluationEpoch"] > 0)  and epoch % self.parameters["evaluationFrequency"] == 0:
                self.agent.underEvaluation = True
                self.runEvaluationEpoch(epoch, self.parameters["stepsPerEvaluationEpoch"], self.parameters["episodesPerEvaluationEpoch"])
                self.agent.underEvaluation = False


    def runTrainingEpoch(self, epoch, maxNumSteps, maxEpisodes):
        self.lifeLost       = False
        self.livesRemaining = False

        assert maxNumSteps >= 0 or maxEpisodes >= 0, "One of max steps and max episodes parameters must be set"
        if maxNumSteps >= 0:
            numStepsRemaining   = maxNumSteps
        else:
            numStepsRemaining   = float('inf')

        if maxEpisodes >=0:
            numEpisodesRemaining= maxEpisodes
        else:
            numEpisodesRemaining= float('inf')


        totalEpisodeCounter = 0
        avgLoss             = 0

        logging.info("Starting Training epoch: " + str(epoch) + " Steps Left: " + str(numStepsRemaining) + " Episodes Left: " + str(numEpisodesRemaining))
        while numStepsRemaining > 0 and numEpisodesRemaining > 0:
            totalEpisodeCounter += 1

            if not self.livesRemaining:
                lowestSamplesTask = self.agent.agentTrainingMemory.getLowestSampledTask()
                self.transferTaskModule.changeToTask(lowestSamplesTask)
                self.livesRemaining = True

            episodeReward, numStepsTaken, episodeTotalTime = self.runEpisode(numStepsRemaining)
            numStepsRemaining    -= numStepsTaken
            numEpisodesRemaining -= 1

            with open(self.learningFilename, 'a') as f:
                avgLoss = np.mean(self.agent.lossArray)
                out = "{0},{1}\n".format(avgLoss, self.agent.epsilon)
                f.write(out)
                f.flush()

            epochFormated = "{0:<3}".format(epoch)
            taskFormatted = "{0:<3}".format(lowestSamplesTask)
            totalEpisodeCounterFormatted = "{0:<10}".format(totalEpisodeCounter)
            episodeRewardFormatted = "{0:<10}".format(episodeReward)
            numStepsRemainingFormatted = "{0:<10}".format(numStepsRemaining)
            numEpisodesRemainingFormatted = "{0:<10}".format(numEpisodesRemaining)
            logging.info("Training Epoch: " + epochFormated + " currentTask: " + taskFormatted + " Episode " + totalEpisodeCounterFormatted + " received reward: " + episodeRewardFormatted + " Average Steps Per Second: {0:6.2f}".format(numStepsTaken / episodeTotalTime) + " Steps remaining: " + numStepsRemainingFormatted + " episodes remaining: " + numEpisodesRemainingFormatted + " Average Loss during episode: {0:.4f}".format(avgLoss))

        logging.info( "Finished Training epoch: " + str(epoch))



    def runEvaluationEpoch(self, epoch, maxNumSteps, maxEpisodes):
        assert maxNumSteps >= 0 or maxEpisodes >= 0, "One of max steps and max episodes parameters must be set"
        for currentEpisodeTask in xrange(self.numTasks):

            totalEpochReward    = 0.0
            totalEpisodeCounter = 0
            self.lifeLost       = False
            epochRewards        = np.array([], dtype='float32')
            self.transferTaskModule.changeToTask(currentEpisodeTask)


            if maxNumSteps >= 0:
                numStepsRemaining   = maxNumSteps
            else:
                numStepsRemaining   = float('inf')

            if maxEpisodes >=0:
                numEpisodesRemaining= maxEpisodes
            else:
                numEpisodesRemaining= float('inf')

            logging.info("Starting Evaluation epoch: " + str(epoch) + " currentTask: " + str(currentEpisodeTask) + " Steps Left: " + str(numStepsRemaining) + " Episodes Left: " + str(numEpisodesRemaining))

            while numStepsRemaining > 0 and numEpisodesRemaining > 0:
                totalEpisodeCounter += 1
                episodeReward, numStepsTaken, episodeTotalTime = self.runEpisode(numStepsRemaining)
                numStepsRemaining    -= numStepsTaken
                numEpisodesRemaining -= 1

                epochFormated = "{0:<3}".format(epoch)
                taskFormatted = "{0:<3}".format(currentEpisodeTask)
                totalEpisodeCounterFormatted = "{0:<10}".format(totalEpisodeCounter)
                episodeRewardFormatted = "{0:<10}".format(episodeReward)
                numStepsRemainingFormatted = "{0:<10}".format(numStepsRemaining)
                numEpisodesRemainingFormatted = "{0:<10}".format(numEpisodesRemaining)
                logging.info("Evaluation Epoch: " + epochFormated + " currentTask: " + taskFormatted + " Episode " + totalEpisodeCounterFormatted + " received reward: " + episodeRewardFormatted + " Average Steps Per Second: {0:6.2f}".format(numStepsTaken / episodeTotalTime) + " Steps remaining: " + numStepsRemainingFormatted + " Episodes Remaining: " + numEpisodesRemainingFormatted)


                if (numStepsRemaining > 0 and numEpisodesRemaining >= 0) or totalEpisodeCounter == 1:
                    #Only count the last episode in the evaluation if we run out of time and its the only episode
                    totalEpochReward += episodeReward
                    epochRewards = np.append(epochRewards, episodeReward)
                else:
                    #We prematurely ended the episode so we dont include its reward and decrease the num episode by 1 to only return the average over completed episodes
                    #(unless only 1 episode was run and ended prematurely)
                    totalEpisodeCounter -= 1
            
            #Calculate the average best Q value from data from this task 
            holdoutSize = self.parameters["numHoldoutQValues"]
            holdoutData = None
        
            if len(self.agent.agentTrainingMemory) > holdoutSize:
                holdoutData = self.agent.agentTrainingMemory.getRandomExperienceBatch(holdoutSize, kReturnLength = self.parameters["kReturnLength"], taskIndex = currentEpisodeTask)
                if holdoutData != None:
                    holdoutData = holdoutData[0]


            holdoutSum = 0.0
            if holdoutData is not None:
                for i in range(holdoutSize):

                    if self.parameters["disjointDQN"]:
                        holdoutSum += np.max(self.agent.functionApproximators[currentEpisodeTask].computeQValues(holdoutData[i, ...], currentEpisodeTask))
                    else:
                        holdoutSum += np.max(self.agent.functionApproximators[0].computeQValues(holdoutData[i, ...], currentEpisodeTask))


            currentTaskFilename = self.resultsFilenames[currentEpisodeTask]
            currentTaskResultsFile = open(currentTaskFilename, 'a', 0)
            
            averageReward   = totalEpochReward / float(totalEpisodeCounter)
            averageBestQVal = holdoutSum / holdoutSize


            epochFormated  = (str(epoch) + ",").ljust(7)
            numEpisodesFormated = (str(totalEpisodeCounter) + ",").ljust(14)
            totalRewardFormated = (str(totalEpochReward) + ",").ljust(14)
            averageRewardFormatted = ("{0:.3f}".format(np.mean(epochRewards)) + ",").ljust(16)
            rewardStdFormatted = ("{0:.3f}".format(np.std(epochRewards)) + ",").ljust(16)
            meanQValFormatted = ("{0:.6}".format(averageBestQVal)).ljust(20) + "\n"


            resultsLine = epochFormated + numEpisodesFormated + totalRewardFormated + averageRewardFormatted + rewardStdFormatted + meanQValFormatted #+ averageRewardFormatted2 + rewardStdFormatted
            currentTaskResultsFile.write(resultsLine)
            currentTaskResultsFile.flush()
            currentTaskResultsFile.close()

            logging.info( "Finished Evaluation epoch: " + str(epoch) + " for task: " + str(currentEpisodeTask)+" with average reward: {0:<10.3f}".format(averageReward))





    def runEpisode(self, maxNumSteps):
        if not self.lifeLost or self.ale.game_over():
            self.ale.reset_game()

            if self.parameters["maxStartingNoOps"] > 0:
                numNoOpsExecuted = self.rng.randint(0, self.parameters["maxStartingNoOps"]+1)
                for i in range(numNoOpsExecuted):
                    self.performAction(0)



        self.performAction(0)
        self.performAction(0)

        startingLives = self.ale.lives()
        action = self.agent.startEpisode(self.getStateObservation())
        startTime = time.time()
        numStepsTaken = 0
        episodeReward  = 0
        while True:

            rewardPool = 0
            for i in range(self.parameters["frameSkip"]):
                rewardPool += self.performAction(action)

            reward = rewardPool
            episodeReward += reward

            self.lifeLost = (self.parameters["deathEndsTrainingEpisode"] and not self.agent.underEvaluation and self.ale.lives() < startingLives)
            naturalEpisodeEnd = self.ale.game_over() or self.lifeLost
            stopEvaluationEpisodeEarly = (self.agent.underEvaluation and numStepsTaken >= self.parameters["maxStepsPerEvaluationEpisode"])

            numStepsTaken += 1

            if naturalEpisodeEnd or numStepsTaken >= maxNumSteps or stopEvaluationEpisodeEarly:
                self.agent.endEpisode(reward)
                totalTime = time.time() - startTime

                if self.ale.game_over():
                    self.livesRemaining = False

                break

            action = self.agent.step(reward, self.getStateObservation())
        return episodeReward, numStepsTaken, totalTime



    def performAction(self, action):
        reward = self.ale.act(action)
        self.screenBuffer[self.bufferCount % 2, ...] = self.ale.getScreenGrayscale().reshape((self.stateHeight, self.stateWidth))
        self.bufferCount += 1
        return reward


    def getStateObservation(self):
        maxedImage = np.maximum(self.screenBuffer[0, ...], self.screenBuffer[1, ...])
        scaledImage = cv2.resize( maxedImage, (self.parameters["resizedHeight"], self.parameters["resizedWidth"]), interpolation=cv2.INTER_LINEAR)
        return scaledImage



if __name__ == "__main__":
    parameters = RLParameters.parseDQTNParametersFromArguments(sys.argv[1:], __doc__)
    aleExperiment = ALEExperiment(parameters)
    aleExperiment.run()
