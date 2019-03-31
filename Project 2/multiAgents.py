# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import math
import statistics

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        food = []
        for x in range(newFood.width):
            for y in range(newFood.height):
                if newFood[x][y] == True:
                    food.append( (x,y) )
        distance = []
        for i in range(len(food)):
            distance.append(manhattanDistance(newPos,food[i]))
        minDistance = min(distance, default=0)    
        if minDistance is not 0:
            successorGameState.data.score += (1/minDistance)
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, 1)

    def maxValue(self, gameState, depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        v = -math.inf
        for action in gameState.getLegalActions(0):
            current = self.minValue(gameState.generateSuccessor(0, action), 1, depth)
            v = max(v, current)
            if v is current:
                best = action
        if depth is 1:
            return best
        else:
            return v

    def minValue(self, gameState,agentIndex, depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        v = math.inf
        for action in gameState.getLegalActions(agentIndex):
            if (agentIndex+1) is gameState.getNumAgents():
                if depth is self.depth:
                    v = min(v, self.evaluationFunction(gameState.generateSuccessor(agentIndex, action)))
                else:
                    v = min(v, self.maxValue(gameState.generateSuccessor(agentIndex, action), depth+1))
            else:
                v = min(v, self.minValue(gameState.generateSuccessor(agentIndex, action), agentIndex+1,depth))
        return v
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    "*** YOUR CODE HERE ***"
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.maxValue(gameState, 1, -math.inf, math.inf)

    def maxValue(self, gameState, depth, maxV, minV):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        v = -math.inf
        for action in gameState.getLegalActions(0):
            current = self.minValue(gameState.generateSuccessor(0, action), 1, depth, maxV, minV)
            v = max(v, current)
            if v is current:
                best = action
            if v > minV:
                return v
            maxV = max(maxV, v)
        if depth is 1:
            return best
        else:
            return v

    def minValue(self, gameState, agentIndex, depth, maxV, minV):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        v = math.inf
        for action in gameState.getLegalActions(agentIndex):
            if (agentIndex + 1) is gameState.getNumAgents():
                if depth is self.depth:
                    v = min(v, self.evaluationFunction(gameState.generateSuccessor(agentIndex, action)))
                else:
                    v = min(v, self.maxValue(gameState.generateSuccessor(agentIndex, action), depth + 1, maxV, minV))
            else:
                v = min(v, self.minValue(gameState.generateSuccessor(agentIndex, action),agentIndex + 1, depth, maxV, minV))
            if v < maxV:
                return v
            minV = min(minV, v)
        return v
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, 1)

    def maxValue(self, gameState, depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        v = -math.inf
        for action in gameState.getLegalActions(0):
            current = self.expectiValue(gameState.generateSuccessor(0, action), 1, depth)
            v = max(v, current)
            if v is current:
                best = action
        if depth is 1:
            return best
        else:
            return v

    def expectiValue(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        v = 0
        choice = 0
        for action in gameState.getLegalActions(agentIndex):
            choice += 1
            if (agentIndex + 1) is gameState.getNumAgents():
                if depth is self.depth:
                    v += self.evaluationFunction(gameState.generateSuccessor(agentIndex, action))
                else:
                    v += self.maxValue(gameState.generateSuccessor(agentIndex, action), depth + 1)
            else:
                v += self.expectiValue(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth)
        return v/choice
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostPos = currentGameState.getGhostPositions()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    
    food = []
    for x in range(newFood.width):
        for y in range(newFood.height):
            if newFood[x][y] is True:
                food.append( (x,y) )
    distance = []
    for i in range(len(food)):
        distance.append(manhattanDistance(newPos,food[i]))
    minDistance = min(distance, default=0)   
    ghostDistance = []
    for i in range(len(newGhostPos)):
        ghostDistance.append(manhattanDistance(newPos,newGhostPos[i]))
    ghostDis = min(ghostDistance, default=0)  
    
    if minDistance is not 0:
        currentGameState.data.score += (1/minDistance)
        
    if newScaredTimes[0] > 5:
        currentGameState.data.score += (40/ghostDis)
        
    return currentGameState.getScore()
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
