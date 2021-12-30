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


from util import manhattanDistance
from game import Directions
import random, util
import functools

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
        # Collect legal moves and child states
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

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        

        "*** YOUR CODE HERE ***"
        score = childGameState.getScore()
        newGhostPositions = childGameState.getGhostPositions()#oi thesois ton ghosts
        food_available = newFood.asList()#lista me ta fagita
        nearest_food = float("inf")
        nearest_ghost = float("inf")
        
        num_food = len(food_available)#arithmos ton diathesimon dots gia fagoma
        
        "to pacman prota pigainei sto kontinotero dot"
        if num_food:
            for food in food_available:#vriskoume to pio kontino dot
                nearest_food = min(nearest_food, manhattanDistance(newPos, food))
        
        "oso pio konta einai to pacman se dot toso megalitero einai kai to score gia na to entharinoume na kinithei pros to kontinotero dot"
        score += 1.0/float(nearest_food) 

        for ghost in newGhostPositions:
            nearest_ghost =  min(nearest_ghost, manhattanDistance(newPos, ghost))
        if (nearest_ghost < 2): #ean vrethei poli konta se ghost epistrefoume -inf oste na min kinithei pros to ghost
            return -float('inf')

        score -= 1.0/(nearest_ghost)
        return score

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

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        action, cost = self.max_value(gameState=gameState, agentIndex=0, depth=0)
        return action
        util.raiseNotDefined()


    def max_value(self, gameState, agentIndex, depth):
        optimal_action = ("max", -float('inf'))
        legal_actions = gameState.getLegalActions(agentIndex)
        num_agends = gameState.getNumAgents()
        new_depth = depth + 1
        if (agentIndex==0):#an o agent itan o pacman tote einai seira tou protou ghost na kinithei
            player = agentIndex+1
        elif (agentIndex==num_agends-1):#ean kounithikan ola ta ghost tote einai seira tou pacman
            player = 0
        else:#an den exoun kounithei ola ta ghosts tote einai seira tou epomenou ghost
            player = agentIndex+1
        for action in legal_actions:
            succesor_state = gameState.getNextState(agentIndex, action)
            new_val = self.minimax(gameState=succesor_state, agentIndex=player, depth=new_depth)
            succ_action = (action, new_val)
            optimal_action = max(optimal_action, succ_action, key=lambda x:x[1])
        return optimal_action

    def min_value(self, gameState, agentIndex, depth):
        optimal_action = ("min", float('inf'))
        legal_actions = gameState.getLegalActions(agentIndex)
        num_agends = gameState.getNumAgents()
        new_depth = depth + 1
        if (agentIndex==0):#an o agent itan o pacman tote einai seira tou protou ghost na kinithei
            player = agentIndex+1
        elif (agentIndex==num_agends-1):#ean kounithikan ola ta ghost tote einai seira tou pacman
            player = 0
        else:#an den exoun kounithei ola ta ghosts tote einai seira tou epomenou γηοστ
            player = agentIndex+1
        for action in legal_actions:
            succesor_state = gameState.getNextState(agentIndex, action)
            new_val = self.minimax(gameState=succesor_state, agentIndex=player, depth=new_depth)
            succ_action = (action, new_val)
            optimal_action = min(optimal_action, succ_action, key=lambda x:x[1])
        return optimal_action

    def minimax(self, gameState, agentIndex, depth):
        if (depth >= self.depth*gameState.getNumAgents() or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        if agentIndex is 0:
            action, cost = self.max_value(gameState, agentIndex, depth)
            return cost
        else:
            action, cost = self.min_value(gameState, agentIndex, depth)
            return cost

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -float('inf')
        beta = float('inf')
        action, cost = self.max_value(gameState=gameState, agentIndex=0, depth=0, alpha=alpha, beta=beta)
        return action
        util.raiseNotDefined()

    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
        if (depth >= self.depth*gameState.getNumAgents() or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        if agentIndex is 0:
            action, cost = self.max_value(gameState, agentIndex, depth, alpha, beta)
            return cost
        else:
            action, cost = self.min_value(gameState, agentIndex, depth, alpha, beta)
            return cost


    def max_value(self, gameState, agentIndex, depth, alpha, beta):
        optimal_action = ("max", -float('inf'))
        legal_actions = gameState.getLegalActions(agentIndex)
        new_depth = depth + 1 
        num_agends = gameState.getNumAgents()
        if (agentIndex==0):#an o agent itan o pacman tote einai siera tou ghost na kinithei
            player = agentIndex+1
        elif (agentIndex==num_agends-1):#ean kounithikan ola ta ghost tote einai seira tou pacman
            player = 0
        else:#an den exoun kounithei ola ta ghosts tote einai seira tou epomenou
            player = agentIndex+1
        for action in legal_actions:
            succesor_state = gameState.getNextState(agentIndex, action)
            new_val = self.alphabeta(gameState=succesor_state, agentIndex=player, depth=new_depth, alpha=alpha, beta=beta)
            succ_action = (action, new_val)
            optimal_action = max(optimal_action, succ_action, key=lambda x:x[1])
            act, val = optimal_action
            if (val > beta):
                return optimal_action
            alpha  = max(alpha, val)
        return optimal_action
    
    def min_value(self, gameState, agentIndex, depth, alpha, beta):
        optimal_action = ("min", float('inf'))
        legal_actions = gameState.getLegalActions(agentIndex)
        new_depth = depth + 1
        num_agends = gameState.getNumAgents()
        if (agentIndex==0):#an o agent itan o pacman tote einai siera tou ghost na kinithei
            player = agentIndex + 1
        elif (agentIndex==num_agends-1):#ean kounithikan ola ta ghost tote einai seira tou pacman
            player = 0
        else:#an den exoun kounithei ola ta ghosts tote einai seira tou epomenou
            player = agentIndex + 1
        for action in legal_actions:
            succesor_state = gameState.getNextState(agentIndex, action)
            new_val = self.alphabeta(gameState=succesor_state, agentIndex=player, depth=new_depth, alpha=alpha, beta=beta)
            succ_action = (action, new_val)
            optimal_action = min(optimal_action, succ_action, key=lambda x:x[1])
            act, val = optimal_action
            if (val < alpha):
                return optimal_action
            beta  = min(beta, val)
        return optimal_action


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
        action, cost = self.max_value(gameState=gameState, agentIndex=0, depth=0)
        return action
        util.raiseNotDefined()

    def max_value(self, gameState, agentIndex, depth):
        optimal_action = ("max", -float('inf'))
        legal_actions = gameState.getLegalActions(agentIndex)
        new_depth = depth + 1
        num_agends = gameState.getNumAgents()
        if (agentIndex==0):#an o agent itan o pacman tote einai siera tou ghost na kinithei
            player = agentIndex + 1
        elif (agentIndex==num_agends-1):#ean kounithikan ola ta ghost tote einai seira tou pacman
            player = 0
        else:#an den exoun kounithei ola ta ghosts tote einai seira tou epomenou
            player = agentIndex + 1
        for action in legal_actions:
            succesor_state = gameState.getNextState(agentIndex, action)
            new_val = self.expectimax(gameState=succesor_state, agentIndex=player, depth=new_depth)
            succ_action = (action, new_val)
            optimal_action = max(optimal_action, succ_action, key=lambda x:x[1])
        return optimal_action

    def expected_value(self, gameState, agentIndex, depth):
        optimal_action = []
        legal_actions = gameState.getLegalActions(agentIndex)
        new_depth = depth + 1
        num_agends = gameState.getNumAgents()
        if (agentIndex==0):#an o agent itan o pacman tote einai siera tou ghost na kinithei
            player = agentIndex + 1
        elif (agentIndex==num_agends-1):#ean kounithikan ola ta ghost tote einai seira tou pacman
            player = 0
        else:#an den exoun kounithei ola ta ghosts tote einai seira tou epomenou
            player = agentIndex + 1
        for action in legal_actions:
            succesor_state = gameState.getNextState(agentIndex, action)
            new_val = self.expectimax(gameState=succesor_state, agentIndex=player, depth=new_depth)
            optimal_action.append(new_val)
        expected_value = sum(optimal_action)/len(optimal_action)
        return expected_value


    def expectimax(self, gameState, agentIndex, depth):
        if (depth >= self.depth*gameState.getNumAgents() or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        if agentIndex is 0:
            action, cost = self.max_value(gameState, agentIndex, depth)
            return cost
        else:
            cost = self.expected_value(gameState, agentIndex, depth)
            return cost

    

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    i nea evaluation function lamvanei ipopsi:
        - tin apostasi sto pio kontino food-dot, 
        -tin apostasi sto pio kontino ghost, 
        -tin apostasi sto pio kontino capsule,
        -ton xrono pou to ghost einai scared

    """
    "*** YOUR CODE HERE ***"
    from functools import reduce
    
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    score = currentGameState.getScore()
    GhostPositions = currentGameState.getGhostPositions()
    food_available = Food.asList()
    nearest_food = float("inf")
    nearest_ghost = float("inf")
    nearest_capsule = float('inf')
    num_food = len(food_available)
    capsules_available = currentGameState.getCapsules()
    num_capsules = len(capsules_available)
    pac_pos=currentGameState.getPacmanPosition()
    scared_ghost_total_time = 0

    if currentGameState.isLose():
        return -float('inf')
    if currentGameState.isWin():
        return float('inf')

    "to pacman prota pigainei sto kontinotero dot"
    if num_food:
        for food in food_available:#vriskoume to pio kontino dot
            nearest_food = min(nearest_food, manhattanDistance(pac_pos, food))

    score += 1.0/(nearest_food)

    if num_capsules:
        for capsule in capsules_available:
            nearest_capsule = min(nearest_capsule, manhattanDistance(pac_pos, capsule))
        
    "oso pio konta einai to pacman se dot toso megalitero einai kai to score gia na to entharinoume na kinithei pros to kontinotero dot"
    score += 1.0/(nearest_capsule)

    for i in ScaredTimes:
        scared_ghost_total_time += i
    score += scared_ghost_total_time

    for ghost in GhostStates:
        nearest_ghost =  min(nearest_ghost, manhattanDistance(pac_pos, ghost.getPosition()))
    if (nearest_ghost < 2): #ean vrethei poli konta se ghost epistrefoume -inf oste na min kinithei pros to ghost
        return -float('inf')

    score -= 1.0/(nearest_ghost)

    return score
   
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
