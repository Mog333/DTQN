"""
Author: Robert Post
Based on code from Nathan Sprague
from: https://github.com/spragunr/deep_q_rl
"""

import DeepNetworks
import lasagne
import numpy as np
import theano
import theano.tensor as T
import cPickle
import imp

class DeepQTransferNetwork(object):
    def __init__(self, rng, numActions, batchSize, inputState,  
            discountRate, learningRate, rho, rms_epsilon, momentum, networkUpdateDelay, useSARSAUpdate, kReturnLength,
            transferExperimentType = "fullShare", numTransferTasks = 1,
            networkType = "conv", clipDelta = 1.0, inputScale = 255.0):
        
        self.rng                = rng
        self.batchSize          = batchSize

        self.inputState         = inputState
        self.inputScale         = inputScale
        self.numActions         = numActions
        self.discountRate       = discountRate
        self.learningRate       = learningRate
        self.rho                = rho
        self.rms_epsilon        = rms_epsilon
        self.momentum           = momentum
        self.networkUpdateDelay = networkUpdateDelay
        self.useSARSAUpdate     = useSARSAUpdate
        self.kReturnLength      = kReturnLength
        self.networkType        = networkType
        self.clipDelta          = clipDelta
        self.updateCounter      = 0
        self.numTransferTasks   = numTransferTasks
        self.transferExperimentType = transferExperimentType


        states           = T.tensor4("states")
        nextStates       = T.tensor4("nextStates")
        rewards          = T.col("rewards")
        gammas           = T.col("rewards")
        actions          = T.icol("actions")
        nextActions      = T.icol("nextActions")
        terminals        = T.icol("terminals")
        validActionsList = T.ivector("validActions")


        self.statesShared      = theano.shared(np.zeros((self.batchSize,) + self.inputState, dtype=theano.config.floatX))
        self.singleStateShared = theano.shared(np.zeros((1,) + self.inputState, dtype=theano.config.floatX))
        self.nextStatesShared  = theano.shared(np.zeros((self.batchSize,) + self.inputState, dtype=theano.config.floatX))
        self.rewardsShared     = theano.shared(np.zeros((self.batchSize, 1), dtype=theano.config.floatX), broadcastable=(False, True))
        self.gammasShared      = theano.shared(np.zeros((self.batchSize, 1), dtype=theano.config.floatX), broadcastable=(False, True))
        self.actionsShared     = theano.shared(np.zeros((self.batchSize, 1), dtype='int32'), broadcastable=(False, True))
        self.nextActionsShared = theano.shared(np.zeros((self.batchSize, 1), dtype='int32'), broadcastable=(False, True))
        self.terminalsShared   = theano.shared(np.zeros((self.batchSize, 1), dtype='int32'), broadcastable=(False, True))

        self.hiddenTransferLayers = []

        # self.qValueNetwork, transferLayers  = DeepNetworks.buildDeepQTransferNetwork(
        #     self.batchSize, self.inputState, self.numActions, self.transferExperimentType , self.numTransferTasks,convImplementation=self.networkType)
        
        self.qValueNetwork, transferLayers  = DeepNetworks.buildTransferNetwork(
            self.transferExperimentType, 
            self.batchSize, 
            self.inputState, 
            self.numActions, 
            self.numTransferTasks,
            convImplementation=self.networkType
        )
        

        for layer in transferLayers:
            self.hiddenTransferLayers.append(layer)


        qValues = lasagne.layers.get_output(self.qValueNetwork, states / self.inputScale)

        if self.networkUpdateDelay > 0:
            # self.nextQValueNetwork, nextTransferLayers = DeepNetworks.buildDeepQTransferNetwork(
            #     self.batchSize, self.inputState, self.numActions, self.transferExperimentType, self.numTransferTasks, convImplementation = self.networkType)
            
            self.nextQValueNetwork, nextTransferLayers  = DeepNetworks.buildTransferNetwork(
                self.transferExperimentType, 
                self.batchSize, 
                self.inputState, 
                self.numActions, 
                self.numTransferTasks,
                convImplementation=self.networkType
            )


            for layer in nextTransferLayers:
                self.hiddenTransferLayers.append(layer)

            self.resetNextQValueNetwork()
            nextQValues = lasagne.layers.get_output(self.nextQValueNetwork, nextStates / self.inputScale)
        else:
            nextQValues = lasagne.layers.get_output(self.qValueNetwork, nextStates / self.inputScale)
            nextQValues = theano.gradient.disconnected_grad(nextQValues)


        if self.useSARSAUpdate:
            target = rewards + (T.ones_like(terminals) - terminals) * gammas * nextQValues[T.arange(self.batchSize), nextActions.reshape((-1,))].reshape((-1, 1))
        else:
            target = rewards + (T.ones_like(terminals) - terminals) * gammas * T.max(nextQValues[:, validActionsList], axis = 1, keepdims = True)

        targetDifference = target - qValues[T.arange(self.batchSize), actions.reshape((-1,))].reshape((-1, 1))

        if self.clipDelta > 0:
            quadraticPart = T.minimum(abs(targetDifference), self.clipDelta)
            linearPart = abs(targetDifference) - quadraticPart
            loss = 0.5 * quadraticPart ** 2 + self.clipDelta * linearPart
        else:
            loss = 0.5 * targetDifference ** 2

        loss = T.sum(loss)

        networkParameters = lasagne.layers.helper.get_all_params(self.qValueNetwork)
        updates = DeepNetworks.deepmind_rmsprop(loss, networkParameters, self.learningRate, self.rho, self.rms_epsilon)

        if self.momentum > 0:
            updates.lasagne.updates.apply_momentum(updates, None, self.momentum)

        lossGivens = {
            states: self.statesShared,
            nextStates: self.nextStatesShared,
            rewards:self.rewardsShared,
            gammas: self.gammasShared,
            actions: self.actionsShared,
            nextActions: self.nextActionsShared,
            terminals: self.terminalsShared
        }

        self.__trainNetwork = theano.function([validActionsList], [loss], updates=updates, givens=lossGivens, on_unused_input='warn')
        self.__computeQValues = theano.function([], qValues[0], givens={states: self.singleStateShared}, on_unused_input='warn')


        #theano.printing.pydotprint(self.__computeQValues, "/home/robpost/Desktop/RLRP/testOutput.png")


    def trainNetwork(self, stateBatch, actionBatch, rewardBatch, gammaBatch, nextStateBatch, nextActionBatch, terminalBatch, currentBatchsTask, currentValidActions):
        self.statesShared.set_value(stateBatch)
        self.nextStatesShared.set_value(nextStateBatch)
        self.actionsShared.set_value(actionBatch)
        self.nextActionsShared.set_value(nextActionBatch)
        self.rewardsShared.set_value(rewardBatch)
        self.gammasShared.set_value(gammaBatch)
        self.terminalsShared.set_value(terminalBatch)

        self.changeNetworkToTask(currentBatchsTask)

        if self.networkUpdateDelay > 0 and self.updateCounter % self.networkUpdateDelay == 0:
            self.resetNextQValueNetwork()

        loss = self.__trainNetwork(currentValidActions)
        self.updateCounter += 1
        return np.sqrt(loss)

    def computeQValues(self, state, currentTask):
        self.singleStateShared.set_value(state.reshape((1,) + state.shape))
        self.changeNetworkToTask(currentTask)
        return self.__computeQValues()


    def changeNetworkToTask(self, taskID):
        for transferLayer in self.hiddenTransferLayers:
            if transferLayer is not None:
                transferLayer.setSwitchIndex(taskID)


    def resetNextQValueNetwork(self):
        networkParameters = lasagne.layers.helper.get_all_param_values(self.qValueNetwork)
        lasagne.layers.helper.set_all_param_values(self.nextQValueNetwork, networkParameters)
