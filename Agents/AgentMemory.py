"""
Author: Robert Post
UOfAlberta 2015

This Class stores samples of experience - Only the most recent x samples are stored.
Samples are stores in a circular array.
Memory is used to construct random batches for the DQN.
"""

import time
import numpy as np
import theano

floatX = theano.config.floatX

class AgentMemory(object):

    def __init__(self, rng, stateShape, phiLength=4, memorySize=10000, discountRate = 1.0, numTasks = 1):
        """ 
        Arguments:
            stateShape - tuple containing the dimensions of the experiences being stored
            phiLength - number of images in a state.
            memorySize - The number of experiences that can be stored

            An experience is a single frame, a state is several frames (phiLength) in a single numpy tensor
        """

        self.currentMemoryIndex     = 0
        self.numberOfExperiences    = 0
        self.memorySize             = memorySize
        self.stateShape             = stateShape
        self.phiLength              = phiLength
        self.numTasks               = numTasks
        self.rng                    = rng
        self.discountRate           = discountRate
        self.taskSampleCount        = np.zeros(self.numTasks, dtype='int32')
        self.totalTaskSampleCount   = np.zeros(self.numTasks, dtype='int32')

        self.stateMemory            = np.zeros((self.memorySize,) + self.stateShape , dtype = 'uint8')
        self.rewardMemory           = np.zeros(self.memorySize, dtype = floatX)
        self.actionMemory           = np.zeros(self.memorySize, dtype='int32')
        self.terminalMemory         = np.zeros(self.memorySize, dtype='int32')
        self.taskMemory             = -1 * np.ones(self.memorySize, dtype='int32')




    def addFrame(self, frame, memoryIndex = None):
        assert( memoryIndex == None or ( (memoryIndex < memorySize) and (memoryIndex >= 0) ) )
        if memoryIndex == None:
            memoryIndex = self.currentMemoryIndex

        assert(self.stateShape[0] == frame.shape[0])
        assert(self.stateShape[1] == frame.shape[1])

        self.stateMemory[memoryIndex, ...] = frame

    def addExperience(self, reward, action, terminal = 0, taskIndex = 0, memoryIndex = None):
        assert( memoryIndex == None or ( (memoryIndex < memorySize) and (memoryIndex >= 0) ) )
        if memoryIndex == None:
            memoryIndex = self.currentMemoryIndex

        self.actionMemory[memoryIndex]     = action
        self.rewardMemory[memoryIndex]     = reward
        self.terminalMemory[memoryIndex]   = terminal

        if self.taskMemory[memoryIndex] != -1:
            #Overwritting another memory
            self.taskSampleCount[self.taskMemory[memoryIndex]] -= 1

        self.taskSampleCount[taskIndex]      += 1
        self.totalTaskSampleCount[taskIndex] += 1
        self.taskMemory[memoryIndex]          = taskIndex

        self.currentMemoryIndex = (self.currentMemoryIndex  + 1) % self.memorySize
        self.numberOfExperiences += 1


    def addSample(self, image, action, reward, terminal, taskIndex):
        self.addFrame(image)
        self.addExperience(reward, action, terminal, taskIndex)

    def getCurrentPhi(self, image):
        phiIndices = np.arange(self.currentMemoryIndex - self.phiLength + 1, self.currentMemoryIndex) % self.memorySize

        phi = np.empty((self.phiLength,) + self.stateShape, dtype = floatX)
        phi[0:self.phiLength - 1] = self.stateMemory[phiIndices]
        phi[-1] = image
        return phi

    def getPhiIndices(self, index = None):
        assert index < self.memorySize
        assert index < self.numberOfExperiences

        if index == None:
            index = self.currentMemoryIndex

        startingIndex = (index - self.phiLength + 1) % self.memorySize
        phiIndices = [(startingIndex + i) % self.memorySize for i in xrange(self.phiLength)]
        return phiIndices

    def getPhi(self, index = None):
        phiIndices = self.getPhiIndices(index)
        phi = np.array([self.stateMemory[i] for i in phiIndices])
        return phi

    def getRandomExperienceBatch(self, batchSize, kReturnLength = 1, taskIndex = None):
        if kReturnLength < 1:
            return None

        if batchSize > self.numberOfExperiences - self.phiLength + 1:
            return None

        if self.taskSampleCount[taskIndex] < batchSize:
            return None

        if taskIndex is not None and self.taskSampleCount[taskIndex] < batchSize:
            return None

        experienceStateShape = (batchSize, self.phiLength) + self.stateShape

        batchStates     = np.zeros(experienceStateShape, dtype='uint8')
        batchNextStates = np.zeros(experienceStateShape, dtype='uint8')
        batchRewards    = np.zeros((batchSize, 1),       dtype=floatX)
        batchActions    = np.zeros((batchSize, 1),       dtype='int32')
        batchNextActions= np.zeros((batchSize, 1),       dtype='int32')
        batchTerminals  = np.zeros((batchSize, 1),       dtype='bool')
        batchGammas     = np.zeros((batchSize, 1),       dtype=floatX)

        count = 0
        maxIndex = min(self.numberOfExperiences, self.memorySize)

        while count < batchSize:

            index = self.rng.randint(0, maxIndex - 1)
            phiIndices = self.getPhiIndices(index)
            #Picked a sample too close to start of episode - sample state crosses episode boundary
            if True in [self.terminalMemory[i] for i in phiIndices]:
                continue

            #Sample is not of the current task
            if taskIndex != None and self.taskMemory[index] != taskIndex:
                continue

            #There is a region of experience we dont want to sample from due to filling in new experience in the replay
            #This area is the region between the current memory index minus the desired return length and
            #the current memory index plus the phi length 
            #And memories slightly over the current index will have their phi states invalidated by going between new and old memories
            #And memories kReturnLength behind the current index cant be sampled as they dont have k steps to form a full k step return

            upperBound = self.currentMemoryIndex + self.phiLength
            lowerBound = self.currentMemoryIndex - kReturnLength
            if upperBound % self.memorySize < upperBound:
                #looped over end of circular buffer by finding starting acceptable index thats above the upper bound
                if index >= lowerBound or index <= upperBound % self.memorySize:
                    continue
            else:
                if lowerBound % self.memorySize > lowerBound:
                #Looped from start to end of circular buffer by subtracting kReturnLength when finding the lower bound
                    if index >= lowerBound % self.memorySize or index <= upperBound:
                        continue

                elif index <= upperBound and index >= lowerBound:
                    continue


            currentReturn = 0.0
            currentDiscount = 1.0
            currentIndex = index
            for i in xrange(0, kReturnLength):
                currentIndex = (index + i) % self.memorySize
                currentReturn += currentDiscount * self.rewardMemory[currentIndex]
                currentDiscount *= self.discountRate

                endIndex = (currentIndex + 1) % self.memorySize

                if self.terminalMemory[endIndex] == True:
                    break

            batchStates[count]     = self.getPhi(index)
            batchNextStates[count] = self.getPhi(endIndex)
            batchRewards[count]    = currentReturn
            batchGammas[count]     = currentDiscount
            batchActions[count]    = self.actionMemory[index]
            batchNextActions[count]= self.actionMemory[endIndex]
            batchTerminals[count]  = self.terminalMemory[endIndex]

            count += 1

        return batchStates, batchActions, batchRewards, batchGammas, batchNextStates, batchNextActions, batchTerminals


    def getLowestSampledTask(self):
        return np.argmin(self.taskSampleCount)

    def __len__(self):
        """ Return the total number of available data items. """
        return max(0, min(self.numberOfExperiences - self.phiLength, self.memorySize - self.phiLength))

def main():
    m = AgentMemory((3,2), 4, 10)
    
    for i in xrange(4):
        frame = np.random.randn(3,2)
        m.addFrame(frame)
        m.addExperience(1, 0, False)

    print "StateMemory:"
    print m.stateMemory
    print "\nPhi:\n"
    print m.getPhi()    

    for i in xrange(15):
        frame = np.random.randn(3,2)
        m.addFrame(frame)
        m.addExperience(1, 0, False)


    frame = np.random.randn(3,2)
    m.addFrame(frame)
    m.addExperience(0, 1, True)

    print "\nCurrent Memory Index:" + str(m.currentMemoryIndex) + "\n"

    frame = np.random.randn(3,2)
    m.addFrame(frame)

    print m.stateMemory
    print "\nCurrent Phi crossing end of memory:\n" + str(m.getPhi())

    return m



if __name__ == "__main__":
    main()
