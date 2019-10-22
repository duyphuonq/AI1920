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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        #print legalMoves
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
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
        #print successorGameState
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        #print newGhostStates
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print newScaredTimes

        "*** YOUR CODE HERE ***"
        foodPos = newFood.asList()
        foodCount = len(foodPos)
        minDis = 10**6
        if foodCount == 0:
          minDis = 0
        else:
          distance = min([manhattanDistance(foodPos[i], newPos) + foodCount*100 for i in range(foodCount)])
          if distance < minDis:
            minDis = distance
        score = -minDis
        for i in range(len(newGhostStates)):
          ghostPos = successorGameState.getGhostPosition(i+1)
          if manhattanDistance(newPos,ghostPos)<=1 :
            score -= 10**6
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

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        #print numAgents
        scores = []
        def _minimax(s, iterCount):
          if iterCount >= self.depth*numAgents or s.isWin() or s.isLose():
            return self.evaluationFunction(s)
          if iterCount%numAgents != 0:    #is ghost
            result = 10**10
            for action in s.getLegalActions(iterCount%numAgents):
              successors = s.generateSuccessor(iterCount%numAgents, action)
              result = min(result, _minimax(successors, iterCount+1))
            return result
          else:   #is Pacman
            result = -10**10
            for action in s.getLegalActions(iterCount%numAgents):
              successors = s.generateSuccessor(iterCount%numAgents, action)
              result = max(result, _minimax(successors, iterCount+1))
              if iterCount == 0:
                scores.append(result)
            return result
        result = _minimax(gameState, 0)
        return gameState.getLegalActions(0)[scores.index(max(scores))]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        scores = []
        def _alphaBeta(s, iterCount, alpha, beta):
          if iterCount >= self.depth*numAgents or s.isWin() or s.isLose():
            return self.evaluationFunction(s)
          if iterCount%numAgents != 0:    #Ghost min
            result = 10**10
            choose_min = iterCount%numAgents
            for action in s.getLegalActions(choose_min):
              successors = s.generateSuccessor(choose_min, action)
              #print "hihi"
              result = min(result, _alphaBeta(successors, iterCount+1, alpha, beta))
              beta = min(beta, result)
              if beta < alpha:
                break
            return result
          else:
            result = -10**10
            choose_max = iterCount%numAgents
            for action in s.getLegalActions(choose_max):
              successors = s.generateSuccessor(choose_max, action)
              #print "haha"
              result = max(result, _alphaBeta(successors, iterCount+1, alpha, beta))
              alpha = max(alpha, result)
              if iterCount == 0:
                scores.append(result)
              if beta < alpha:
                break
            return result
        result = _alphaBeta(gameState, 0, -10**20, 10**20)
        return gameState.getLegalActions(0)[scores.index(max(scores))]

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
        numAgents = gameState.getNumAgents()
        ActionScore = []

        def _expectMinimax(s, iterCount):
          if iterCount >= self.depth*numAgents or s.isWin() or s.isLose(): #leaf node
            return self.evaluationFunction(s)
          if iterCount%numAgents != 0: #is Ghost 
            successorScore = []
            for a in s.getLegalActions(iterCount%numAgents):
              new_gameState = s.generateSuccessor(iterCount%numAgents,a)
              result = _expectMinimax(new_gameState, iterCount+1)
              successorScore.append(result)
            averageScore = sum([ float(x)/len(successorScore) for x in successorScore])
            return averageScore
          else: #is Pacman
            result = -10**10
            for a in s.getLegalActions(iterCount%numAgents):
              new_gameState = s.generateSuccessor(iterCount%numAgents,a)
              result = max(result, _expectMinimax(new_gameState, iterCount+1))
              if iterCount == 0:
                ActionScore.append(result)
            return result
      
        result = _expectMinimax(gameState, 0)
        return gameState.getLegalActions(0)[ActionScore.index(max(ActionScore))]
        
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    def _ghostHunting(gameState):
      score = 0
      for ghost in gameState.getGhostStates():
        disGhost = manhattanDistance(gameState.getPacmanPosition(), ghost.getPosition())
        if ghost.scaredTimer > 0:
          score += pow(max(8 - disGhost, 0), 2)
        else:
          score -= pow(max(7 - disGhost, 0), 2)
      return score

    def _foodGobbling(gameState):
      disFood = []
      for food in gameState.getFood().asList():
        disFood.append(1.0/manhattanDistance(gameState.getPacmanPosition(), food))
      if len(disFood)>0:
        return max(disFood)
      else:
        return 0

    def _pelletNabbing(gameState):
      score = []
      for Cap in gameState.getCapsules():
        score.append(50.0/manhattanDistance(gameState.getPacmanPosition(), Cap))
      if len(score) > 0:
        return max(score)
      else:
        return 0
    score = currentGameState.getScore()
    return score + _ghostHunting(currentGameState) \
                  + _foodGobbling(currentGameState) \
                    + _pelletNabbing(currentGameState)

# Abbreviation
better = betterEvaluationFunction

