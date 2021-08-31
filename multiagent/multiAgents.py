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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        food_list = currentGameState.getFood().asList()
        alive_ghost_positions = [ghost_state.getPosition() for ghost_state in newGhostStates if
                                 ghost_state.scaredTimer == 0]

        if newPos in alive_ghost_positions:
            return -1
        if newPos in food_list:
            return 1

        def get_closest_distance(li):
            if len(li) == 0:
                return 1
            else:
                return min([util.manhattanDistance(newPos, item) for item in li])

        closest_food_pos = get_closest_distance(food_list)
        closest_ghost_pos = get_closest_distance(alive_ghost_positions)

        return 1.0 / closest_food_pos - 1.0 / closest_ghost_pos


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        ghost_num = gameState.getNumAgents() - 1

        def should_eval_terminate(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        def minimize(state, depth, agent_index):
            if should_eval_terminate(state, depth):
                return self.evaluationFunction(state)

            legal_actions = state.getLegalActions(agent_index)
            min_value = float('inf')
            for action in legal_actions:
                successor_state = state.generateSuccessor(agent_index, action)

                if agent_index == ghost_num:
                    min_value = min(maximize(successor_state, depth + 1), min_value)

                else:
                    min_value = min(minimize(successor_state, depth, agent_index + 1), min_value)

            return min_value

        def maximize(state, depth):
            if should_eval_terminate(state, depth):
                return self.evaluationFunction(state)

            legal_actions = state.getLegalActions(0)
            max_value = float('-inf')
            optimal_action = Directions.STOP
            for action in legal_actions:
                successor_state = state.generateSuccessor(0, action)
                eval_value = minimize(successor_state, depth, 1)
                if max_value < eval_value:
                    max_value = eval_value
                    optimal_action = action
            if depth > 0:
                return max_value
            return optimal_action

        return maximize(gameState, 0)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
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
        ghost_num = gameState.getNumAgents() - 1

        def should_eval_terminate(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        def expected_value(state, depth, agent_index):
            if should_eval_terminate(state, depth):
                return self.evaluationFunction(state)

            legal_actions = state.getLegalActions(agent_index)
            expected = 0
            prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                successor_state = state.generateSuccessor(agent_index, action)

                if agent_index == ghost_num:
                    expected += prob * maximize(successor_state, depth + 1)

                else:
                    expected += prob * expected_value(successor_state, depth, agent_index + 1)

            return expected

        def maximize(state, depth):
            if should_eval_terminate(state, depth):
                return self.evaluationFunction(state)

            legal_actions = state.getLegalActions(0)
            max_value = float('-inf')
            optimal_action = Directions.STOP
            for action in legal_actions:
                successor_state = state.generateSuccessor(0, action)
                eval_value = expected_value(successor_state, depth, 1)
                if max_value < eval_value:
                    max_value = eval_value
                    optimal_action = action
            if depth > 0:
                return max_value
            return optimal_action

        return maximize(gameState, 0)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pac_pos = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    food_num = currentGameState.getNumFood()
    ghost_state = currentGameState.getGhostStates()
    capsule_num = len(currentGameState.getCapsules())
    game_score = currentGameState.getScore()

    alive_ghost_positions = [ghost_state.getPosition() for ghost_state in ghost_state if
                             ghost_state.scaredTimer == 0]
    scared_ghost_positions = [ghost_state.getPosition() for ghost_state in ghost_state if
                              ghost_state.scaredTimer != 0]

    def get_closest_distance(li):
        if not li:
            return 0
        else:
            return min([util.manhattanDistance(pac_pos, item) for item in li])

    closest_alive_ghost_dis = 5 if get_closest_distance(alive_ghost_positions) <= 0 else get_closest_distance(
        alive_ghost_positions)
    closest_scared_ghost_dis = get_closest_distance(scared_ghost_positions)
    closest_food_dis = get_closest_distance(food_list)

    score = game_score - 1.5 * closest_food_dis - 2 * (
            1.0 / closest_alive_ghost_dis) - 2 * closest_scared_ghost_dis - 20 * capsule_num - 4 * food_num
    return score


# Abbreviation
better = betterEvaluationFunction
