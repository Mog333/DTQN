'''
Author Robert Post

Reinforcement Learning Parameters for the DQTN

'''


import json
import os
import sys
import argparse
import logging


Parameters = {
    #Experiment Parameters
    "numEpochs": [200, "The number of training epochs and evaluation epochs in the experiment."],
    "startingEpoch": [0, "The current epoch number to start on, for restarting experiments"],
    "stepsPerTrainingEpoch": [250000, "The number of frames the agent sees in a training epoch"],
    "stepsPerEvaluationEpoch": [-1, "The number of frames the agent sees in a evaluation epoch"],
    "episodesPerTrainingEpoch": [-1, "The number of episodes to complete for a single training epoch"],
    "episodesPerEvaluationEpoch": [30, "The number of episodes to complete for a single evaluation epoch"],
    
    "experimentDirectory": ["./", "The directory to store results and network files"],
    "parameterFile":["","The parameter file to load parameters from instead of defaults"],
    "evaluationFrequency": [1, "This constant divides the number of epochs we evaluate in. Higher is less often"],
    "epsilonDecaySteps": [1000000, "The number of steps to decay epsilon over"],
    "epsilonEnd": [0.1, "The final epsilon value after decaying (During training)"],
    "epsilonStart": [1.0, "The starting epsilon value for decaying"],
    "evalEpsilon": [0.05, "The epsilon value used when evaluating"],
    "numHoldoutQValues": [3200, "The number of samples used to calculate the average Q value after evaluation"],
    "maxStepsPerEvaluationEpisode": [60 * 60 * 5 / 4, "The maximum number of steps an evaluation episode can run; default is 5 minutes"],
    "reduceEpochLengthByNumFlavors": [False, "Used to reduce the length of an experiment "],
    "seed": [123456, "Seed used in random number generation"],
    "logToFile": [True, "Sets the python logging to log to a file or to the standard output"],
    "flavorOptionDelimiter":[',', "Delimiter separating mode / difficulty (or other options?) default is , ex: (0,0)"],
    "flavorTaskDelimiter":['#', "Delimiter separating options for multiple tasks with the same rom (but different mode/difficulties). ex: (0,0)&(0,1)"],
    "romDelimiter": ['^', "Delimiter used to separate modes and difficulties between roms in a mode/diff string. Default is ^. ex: pong^space_invaders"],
    "romPartsDelimiter":[':', "Delimter separating the rom from the flavor options for that rom. Default is :. ex: pong:(0,0)"],

    #ALE Parameters
    "baseRomPath": ["../../ALE/roms/", "Where to find the rom files for the ALE"],
    "rom": ["breakout.bin", "The rom to load (with or without the .bin extension)"],
    "deathEndsTrainingEpisode": [True, "Signal to end episodes early in the ALE when the player loses a life during training"],
    "deathEndsEvaluationEpisode": [False, "Signal to end episodes early in the ALE when the player loses a life during evaluation"],
    "displayScreen": [False, "Signal to the ALE to display the game as its being played using SDL"],
    "repeatActionProbability": [0.0, "ALE flag to have a chance to repeat the last action to simulate the innaccuraties of a human playing with a joystick and to introduce randomness"],
    "difficultyString": ["", "String representation of which difficulties of the rom(s) to play"],
    "modeString": ["", "String representation of which modes of the rom(s) to play"],
    "resizedHeight": [84, "The height the ALE images are resized to for input into the agent"],
    "resizedWidth": [84, "The width the ALE images are resized to for input into the agent"],
    "actionSet":["minimal", "Which action set to use, minimal or full"],


    #Agent Parameters
    "maxStartingNoOps": [30, "The number of noops taken by the agent at the start of an episode"],
    "frameSkip": [4, "How often an agent gets a datapoint / state from the ALE. When >1 the last action selected by the agent is repeated over these skipped frames. Rewards are also pooled over these frames."],
    "kReturnLength": [1, "How many steps into the future for calculating the lambda return"],
    "replayMemorySize": [1000000, "The number of experiences to store in the agents memory"],
    "replayStartSize": [50000, "The minimum number of memories before the agent starts training"],
    "phiLength": [4, "The number of previous game screens / states concatenated together to form the true state used by the RL algorithm"],
    


    #Deep Learning / Neural Network Parameters
    "batchSize": [32, "The number of samples in a batch"],
    "discountRate": [0.99, "Discount rate in RL algoriths"],
    "learningRate": [0.00025, "Learning rate in Deep Learning algorithms"],
    "loadWeightsFlipped": [False, "Flag to load weights for conv filters flipped when loaded a model"],
    "momentum": [0, "Momentum used in updating a parameter"],
    "networkType": ["conv", "The type of convolution function used. dnn or cuda or conv"],
    "networkUpdateDelay": [2500, "How often to set the target network weights to be equal to the q value network"],
    "rmsEpsilon": [0.01, "Epsilon param in the rms update equation"],
    "rmsRho": [0.95, "rho param in the rms update equation"],
    "transferExperimentType": ["DQNNet", "Type of network architecture, allowed values: {DQNNet, PolicySwitchNet, PolicyPartialSwitchNet, RepresentationSwitchNet, FirstRepresentationSwitchNet, TaskTransformationNet} To understand these architectures look at their corresponding files under FunctionApproximation/TransferArchitecture"],
    "updateFrequency": [4, "Number of frames seen before running a training pass"],
    "useSARSAUpdate": [False, "Flag to set the network target update rule to use a sarsa like update by looking at the next action taken rather than the best action taken for computing q value differences"],
    "nnFile": ["", "The neural network file to load a existing model"],
    "clipDelta": [1.0,"The value to clip the target different in the learning algorithm at."],
    "disjointDQN":[False, "Used to setup multiple networks, one for each task that are completely separate. Then all are used during traing, except after sampling a task only that task gets the actual data the other network get a batch of zeros. Computing Q values for actions only use the current tasks network"]
}



def parseDQTNParametersFromArguments(args, description):
    """
    Args is a list of string arguments from the commandline to be processed,
    description is the doct string for the help argparse help message


        Order of operation:
        parse command line arguments
        if a specified parameter file exists load it
        if one isnt specified but one exists in the specified experiment directory, load it
        overwrite loaded or default parameters with command line arguments
        Make sure the rom is in the right format
        Make sure the experiment directory and results file exist or make them
        Write current parameters to the params file in the experiment directory

        THE NEW PARAMETERS CAN OVERWRITE THE OLD FILE WHEN SPECIFIED ON THE COMMAND LINE
        useful when you want to increase the epochs or steps per test or something....
   
    """

    parser = argparse.ArgumentParser(description=description)

    defaultParams = Parameters

    for key in defaultParams:
        t = type(defaultParams[key][0])
        if t == bool:
            t = str
        parser.add_argument('--' + str(key), dest=str(key), type=t, help = defaultParams[key][1])

    parsedArgsDict = vars(parser.parse_args(args))

    if parsedArgsDict["parameterFile"] != None and parsedArgsDict["parameterFile"].endswith("json"):
        params = readParametersFromJSON(parsedArgsDict["parameterFile"])
        if params == False:
            params = defaultParams
    else:
        #Autoload parameter files in an experiment directory
        if parsedArgsDict["experimentDirectory"] != None and os.path.exists(parsedArgsDict["experimentDirectory"] + "/parameters.json"):
            params = readParametersFromJSON(parsedArgsDict["experimentDirectory"] + "/parameters.json")
            if params == False:
                params = defaultParams
        else:
            params = defaultParams


    #Replace default / loaded arguments with new ones from command line
    for key in parsedArgsDict:
        if parsedArgsDict[key] != None:
            if type(parsedArgsDict[key]) == str:
                #Parse boolean string flags to actual bools
                if parsedArgsDict[key].lower() == "true":
                    parsedArgsDict[key] = True
                elif parsedArgsDict[key].lower() == "false":
                    parsedArgsDict[key] = False

            params[key][0] = parsedArgsDict[key]


    #Check for the existence of an experiment directory and create it
    if not os.path.isdir(params["experimentDirectory"][0]):
        os.mkdir(params["experimentDirectory"][0])


    parametersPath = params["experimentDirectory"][0] + "/parameters.json"
    writeParametersToJSON(params, parametersPath)

    #look for the highest network file in the experiment directory, for restarting experiments that were canceled without commandline changes
    if len(params["nnFile"][0]) == 0:
        contents = os.listdir(params["experimentDirectory"][0])

        networkFiles = []
        for handle in contents:
            if handle.startswith("network") and handle.endswith(".pkl"):
                networkFiles.append(handle)

         
        if len(networkFiles) > 0:
            #Found a previous experiments network files, now find the highest epoch number
            highestNNFile = networkFiles[0]
            highestNetworkEpochNumber = int(highestNNFile[highestNNFile.index("_") + 1 : highestNNFile.index(".")])
            for networkFile in networkFiles:
                networkEpochNumber =  int(networkFile[networkFile.index("_") + 1 : networkFile.index(".")])
                if networkEpochNumber > highestNetworkEpochNumber:
                    highestNNFile = networkFile
                    highestNetworkEpochNumber = networkEpochNumber

            logging.info("Highest Network File found: {0}".format(highestNetworkEpochNumber))
            params["nnFile"][0] = params["experimentDirectory"][0] + "/" + str(highestNNFile)
            params["startingEpoch"][0] = highestNetworkEpochNumber

            #When restarting a experiment dont anneal the epsilon down
            #This causes your actions to produce a better starting memory which can affect training
            #TO DO 5 is an arbitrary number - do something about it? 
            if highestNetworkEpochNumber >= 5:
                params["epsilonStart"][0] = params["epsilonEnd"][0]



    #Remove the help string to slightly simplify using the parameters dictionary
    paramsWithoutHelpString = {}
    for key in params:
        paramsWithoutHelpString[key] = params[key][0]

    #Parse rom parameter to a list of roms for multigame experiments
    # romList = []
    # for rom in params["rom"][0].split(","):
    #     if not rom:
    #         continue

    #     if not rom.endswith(".bin"):
    #         rom += ".bin"
    #     romList.append(paramsWithoutHelpString["baseRomPath"] + "/" + rom)

    # paramsWithoutHelpString["roms"] = romList 


    return paramsWithoutHelpString



def readParametersFromJSON(jsonFileString):
    parameters = {}

    try:
        f = open(jsonFileString, 'r')
        parameters = json.load(f)
        f.close()
    except IOError as e:
        print "IOError: Error({0}): {1}".format(e.errno, e.strerror)
        return False

    return parameters
    

def writeParametersToJSON(parameters, jsonFileString):
    try:
        with open(jsonFileString, 'w') as  f:
            paramString = json.dumps(parameters, sort_keys = True, indent=4, separators=(',', ': '))
            f.write(paramString)
            return True
    except IOError as e:
        print "Error Dumping Parameters to JSON File\nIOError: Error({0}): {1}".format(e.errno, e.strerror)
        return False


if __name__ == "__main__":
    p = parseDQTNParametersFromArguments(sys.argv[1:], __doc__)
    for key in p:
        print key,": ", p[key]
