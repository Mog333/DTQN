import sys
from random import randrange
import random
from ale_python_interface import ALEInterface
import ALEEnvironment



'''
    example rom strings
    3 tasks: pong space invaders easy and space invaders hard / fast bombs 
    pong:(0,0)^space_invaders:(0,0)&(1,7)
    pong^space_invaders
    pong.bin

'''


class TransferTaskModule():
    def __init__(self, ale, romString, baseRomPath, actionSet = "minimal", romDelimiter='^', romPartsDelimiter = ':', flavorTaskDelimiter='#', flavorOptionDelimiter=','):

        self.ale = ale
           
        if actionSet == "full":
            actionSetFunction = ale.getLegalActionSet
        else:
            actionSetFunction = ale.getMinimalActionSet


        gameCounter = 0
        self.gameInfo = []
        self.numTasks = 0
        self.allActionsList = set()

        for rom in romString.split(romDelimiter):
            romParts = rom.split(romPartsDelimiter)
            romName = romParts[0]
            romPath = baseRomPath + "/" + romName
            if not romPath.endswith('.bin'):
                romPath += ".bin"
            
            if len(romParts) == 2:
                romOptionsList = romParts[1].split(flavorTaskDelimiter)
                romOptionsList = [eval(option) for option in romOptionsList]
            else:
                romOptionsList = [(0,0)]

            ale.loadROM(romPath)
            romActionsList = actionSetFunction()
            self.allActionsList = self.allActionsList.union(set(romActionsList))
            
            availableDiffs = ale.getAvailableDifficulties()
            availableModes = ale.getAvailableModes()
            numModes = len(availableModes)
            numDifficulties = len(availableDiffs)

            #Validate flavor options mode and difficulty are available for this rom
            for romOption in romOptionsList:
                diff,mode = romOption
                if diff < 0 or mode < 0:
                    raise Exception("You entered a negative mode or difficulty.")

                if diff >= numDifficulties or mode >= numModes:
                    exceptionString = "\nYou entered a mode or difficulty that is out of bounds."
                    exceptionString += "\ngame: " + romName + " has " + numDifficulties + " difficulties and " + numModes + " modes."
                    raise Exception(exceptionString)

            numFlavors = len(romOptionsList)
            self.numTasks += numFlavors

            romDict = {"romName": romName, "romPath":romPath, "numFlavors":numFlavors, "modesList":availableModes, "difficultiesList":availableDiffs, "actionsList":romActionsList, "flavorList":romOptionsList}
            gameCounter += 1
            self.gameInfo.append(romDict)


        self.allActionsList = sorted(list(self.allActionsList))

        for game in self.gameInfo:
            game["actionIndices"] = []
            for action in game["actionsList"]:
                if action in self.allActionsList:
                    game["actionIndices"].append(self.allActionsList.index(action))
            
        # self.currentGameIndex = -1
        self.currentTaskIndex = -1
        self.changeToTask(0)

    def getNumTasks(self):
        return self.numTasks

    def getNumGames(self):
        return len(self.gameInfo)

    def getNumTotalActions(self):
        return len(self.allActionsList)

    def getTotalActionsList(self):
        return self.allActionsList

    def getActionsForCurrentTask(self):
        return self.getActionsForTask(self.currentTaskIndex)

    def getActionsForTask(self, taskIndex):
        gameDict = self.__getGameDictAndTaskIndex(taskIndex)[0]
        return gameDict["actionIndices"]

    def getGameInfoForCurrentTask(self):
        return self.getGameInfoForTask(self.currentTaskIndex)

    def getGameInfoForTask(self, taskIndex):
        return self.__getGameDictAndTaskIndex(taskIndex)[0]


    def __getGameDictAndTaskIndex(self, taskIndex = None):
        if taskIndex == None:
            taskIndex = self.currentTaskIndex

        assert taskIndex >= 0

        taskCounter = taskIndex
        gameDict = None 
        for game in self.gameInfo:
            if taskCounter < game["numFlavors"]:
                gameDict = game
                break

            taskCounter -= game["numFlavors"]

        assert(gameDict != None)
        return gameDict, taskCounter


    def changeToTask(self, newTaskNumber):
        assert newTaskNumber >= 0 and newTaskNumber < self.getNumTasks()

        if newTaskNumber == self.currentTaskIndex:
            return

        #newTaskIndex is relative to the number of tasks in the selected gameDict
        gameDict, newTaskIndex= self.__getGameDictAndTaskIndex(newTaskNumber)
        self.currentTaskIndex = newTaskNumber

        rom = gameDict["romPath"]
        diffIndex = gameDict["flavorList"][newTaskIndex][0]
        modeIndex = gameDict["flavorList"][newTaskIndex][1]

        diff = gameDict["difficultiesList"][diffIndex]
        mode = gameDict["modesList"][modeIndex]

        # print "changing to task: " + str(newTaskNumber)
        # print "Tasks action indices: " +str(self.getActionsForCurrentTask())

        self.ale.loadROM(rom)
        self.ale.setMode(mode)
        self.ale.setDifficulty(diff)
        self.ale.reset_game()



if __name__ == "__main__":
    baseRomPath = "../../../../ALE/roms/"
    ale = ALEInterface()
    ALEEnvironment.initializeALEParameters(ale, 1, 0.00, False)



    def testInspectTransferModule(t):
        print "NumGames: " + str(t.getNumGames())
        print "NumTasks: " + str(t.getNumTasks())
        print "TotalActionsList: " + str(t.getTotalActionsList())

        for x in range(t.getNumTasks()):
            print "Changing to task: " + str(x)
            t.changeToTask(x)
            print "Current Game Info Dict: " +  str(t.getGameInfoForCurrentTask())
            print "CurrentTaskActions: " + str(t.getActionsForCurrentTask())

    t = TransferTaskModule(ale, "pong", baseRomPath, "minimal")
    testInspectTransferModule(t)
    print "\n"

    t = TransferTaskModule(ale, "pong:(0,0)^space_invaders:(0,0)#(1,0)", baseRomPath, "minimal")
    testInspectTransferModule(t)
    print "\n"


    t = TransferTaskModule(ale, "pong^space_invaders", baseRomPath, "full")
    testInspectTransferModule(t)
    print "\n"


    t = TransferTaskModule(ale, "hero^pong^freeway:(0,0)", baseRomPath, "minimal")
    testInspectTransferModule(t)
    print "\n"


    t = TransferTaskModule(ale, "hero:(0,0)#(0,1)^pong^freeway:(0,0)#(1,1)", baseRomPath, "minimal")
    testInspectTransferModule(t)
    print "\n"