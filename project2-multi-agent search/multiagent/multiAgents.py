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


from audioop import avg
from turtle import position
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

        # print("legal moves: {}".format(legalMoves))


        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        ###
        #For action in legal moves:
            # Required_list=[]
            # # a variable that store return value
            # Temp =self.evaluationFunction(gameState, action)
            # Required_list.append(temp)
        ###

        bestScore = max(scores)
        # find the best index from the best score
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
        newFoodCoor = newFood.asList()

        print("\n\n")
        print("\nsuccessorGameState: {}".format(successorGameState))
        print("new pos: {} \nnew food: \n{} \nnew ghost states: {}".format(newPos, newFoodCoor, newScaredTimes))



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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.pacmanPath = []

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

        #####
        # brain storming
        # use self.depth, self.evaluationFunction
        # which one is min / max agent?

        # for 1 ghost
        # get the value of top node by searching thru depth n
        minimax(evaluation func, depth, maxAgent):
            if depth == 0:
                return evaluation of this state

            if maxAgent:
                eval = -inf
                for node in childNodes:
                    val = minimax(depth-1, false)
                    eval = max(eval, val)
            else:
                eval = +inf
                for node in childNodes:
                    val = minimax(depth-1, true)
                    eval = max(eval, val)

        """
        "*** YOUR CODE HERE ***"

        # pacPos = gameState.getPacmanPosition()
        # pacState = gameState.getPacmanState()
        # print("pacman. pos: {}, state: {}".format(pacPos, pacState))
        # print("gameState: \n{}".format(gameState))
        print("number of agents: {}".format(gameState.getNumAgents()))

        _, action = self.minimax(gameState, self.depth, 0)

        # return the action list of pacman
        # print("\n\npath returned: {}".format(self.pacmanPath[::-1]))
        return action

        #     path = self.minimaxSearch(gameState, agentIndex=0, depth=self.depth)[1]
        #     # print("path: {}".format(path))
        #     return path

    # single ghost single pacman version
    # max agent = pacman, min agent = ghost
    def minimax(self, gameState, depth, agent):
        # divider = ''
        # for i in range(0, depth):
        #     divider += '     '
        # for i in range(0, depth):
        #     divider += '-----'
        # print("\n" + divider + "start\n")

        print("minimax called, depth = {}, agent = {}".format(depth, agent))

        if depth == 0 or gameState.isWin() or gameState.isLose():
            print("\n\nterminal state reached, return: {}\n\n".format(self.evaluationFunction(gameState)))
            return (self.evaluationFunction(gameState), Directions.STOP) #evaluation of this state

        if agent == 0:
            print("pacman")
            maxScore = float('-inf')
            actionToTake = Directions.STOP

            # loop thru possible actions
            legalActions = gameState.getLegalActions(0)
            # print("legal actions: {}".format(legalActions))

            for action in legalActions:
                nextGameState = gameState.generateSuccessor(0, action)
                newAgentIndex = (agent+1) % gameState.getNumAgents()
                val, useless = self.minimax(nextGameState, depth, newAgentIndex)

                # print("\n\ndepth: {}, val: {}, returned action: {}\n\n".format(depth, val, useless))

                # pick the choice with highest score
                if val > maxScore:
                    # print("maxScore updated to {} from {}".format(val, maxScore))
                    maxScore = val
                    actionToTake = action

            # for i in range(0, depth):
            #     divider += '-----'
            # print("\n" + divider + "end\n")

            print("returning max result: {}, {}".format(maxScore, actionToTake))
            return (maxScore, actionToTake)
        else:
            print("ghost{}".format(agent))
            minScore = float('inf')
            actionToTake = Directions.STOP

            legalActions = gameState.getLegalActions(agent)
            # print("legal actions: {}".format(legalActions))

            # if not legalActions:
            #     # always have stop
            #     # idk if returning 0 for no legal action is correct

            #     print("\n\n no legal actions avail, return: {}\n\n".format(self.evaluationFunction(gameState)))
            #     return self.evaluationFunction(gameState) #evaluation of this state

            for action in legalActions:
                nextGameState = gameState.generateSuccessor(agent, action)

                newAgentIndex = (agent+1) % gameState.getNumAgents()
                # print("newAgentIndex = {}".format(newAgentIndex))

                # pick the choice with highest score
                if newAgentIndex == 0:
                    val, useless = self.minimax(nextGameState, depth-1, newAgentIndex)
                else:
                    # Important: A single level of the search is considered
                    # to be one Pacman move and all the ghosts’ responses,
                    # so depth 2 search will involve Pacman and each ghost moving twice.
                    val, useless = self.minimax(nextGameState, depth, newAgentIndex)

                # print("\n\ndepth: {}, val: {}, returned action: {}\n\n".format(depth, val, useless))

                if val < minScore:
                    minScore = val
                    actionToTake = action
                # minScore = min(minScore, val)

            # if minScore == float('-inf'):
            #     print("\n\n\n\n fix this bug \n\n\n\n\n")

            # for i in range(0, depth):
            #     divider += '-----'
            # print("\n" + divider + "end\n")

            # print("returning min result: {}, {}".format(minScore, actionToTake))
            return (minScore, actionToTake)

            # # find the min number
            # for node in childNodes:
            #     val = self.minimax(depth-1, true)
            #     eval = max(eval, val)

        # # testing
        # legalActions = gameState.getLegalActions(1)
        # for action in legalActions:
        #     successors = gameState.generateSuccessor(1, action)
        #     print("ghost 1 successors: {}".format(successors))

        # legalActions = gameState.getLegalActions(2)
        # for action in legalActions:
        #     successors = gameState.generateSuccessor(1, action)
        #     print("ghost 2 successors: {}".format(successors))

    # def minimaxSearch(self, gameState, agentIndex, depth):
    #     if depth == 0 or gameState.isLose() or gameState.isWin():
    #         ret = self.evaluationFunction(gameState), Directions.STOP
    #     elif agentIndex == 0:
    #         ret = self.maximizer(gameState, agentIndex, depth)
    #     else:
    #         ret = self.minimizer(gameState, agentIndex, depth)
    #     return ret

    # def minimizer(self, gameState, agentIndex, depth):
    #     actions = gameState.getLegalActions(agentIndex)
    #     # print("agent number: {}".format(gameState.getNumAgents()))
    #     if agentIndex == gameState.getNumAgents() - 1:
    #         next_agent, next_depth = 0, depth - 1
    #     else:
    #         next_agent, next_depth = agentIndex + 1, depth
    #     min_score = 1e9
    #     min_action = Directions.STOP
    #     for action in actions:
    #         successor_game_state = gameState.generateSuccessor(agentIndex, action)
    #         new_score = self.minimaxSearch(successor_game_state, next_agent, next_depth)[0]
    #         if new_score < min_score:
    #             min_score, min_action = new_score, action
    #     return min_score, min_action

    # def maximizer(self, gameState, agentIndex, depth):
    #     actions = gameState.getLegalActions(agentIndex)
    #     if agentIndex == gameState.getNumAgents() - 1:
    #         next_agent, next_depth = 0, depth - 1
    #     else:
    #         next_agent, next_depth = agentIndex + 1, depth
    #     max_score = -1e9
    #     max_action = Directions.STOP
    #     for action in actions:
    #         successor_game_state = gameState.generateSuccessor(agentIndex, action)
    #         new_score = self.minimaxSearch(successor_game_state, next_agent, next_depth)[0]
    #         if new_score > max_score:
    #             max_score, max_action = new_score, action
    #     return max_score, max_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        _, action = self.minimax(gameState, self.depth, agentIndex=0, alpha=float('-inf'), beta=float('inf'))
        return action

    def minimax(self, gameState, depth, agentIndex, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        if agentIndex == 0:
            maxScore = float('-inf')
            optimalAction = Directions.STOP

            # find possible actions & assume at least have 1 legal action ()
            legalActions = gameState.getLegalActions(agentIndex)

            for action in legalActions:
                # find next game state
                nextState = gameState.generateSuccessor(agentIndex, action)

                # update agent index
                newAgentIndex = agentIndex + 1

                # minimax on the child node
                score, _ = self.minimax(nextState, depth, agentIndex+1, alpha, beta) # 0+1

                # update max score & optimal action
                if score > maxScore:
                    maxScore = score
                    optimalAction = action

                # update alpha
                alpha = max(alpha, score)

                # if alpha > beta:
                #     break
                if score > beta:
                    return score, action

            return maxScore, optimalAction
        else:
            minScore = float('inf')
            optimalAction = Directions.STOP

            legalActions = gameState.getLegalActions(agentIndex)

            for action in legalActions:
                # find next game state
                nextGameState = gameState.generateSuccessor(agentIndex, action)

                # assume when agent of index != ghost btw.
                # Important: A single level of the search is considered
                # to be one Pacman move and all the ghosts’ responses,
                # so depth 2 search will involve Pacman and each ghost moving twice.
                newAgentIndex = (agentIndex+1) % gameState.getNumAgents()

                if newAgentIndex == 0:
                    score, _ = self.minimax(nextGameState, depth-1, newAgentIndex, alpha, beta)
                else:
                    score, _ = self.minimax(nextGameState, depth, newAgentIndex, alpha, beta)

                # update max score & optimal action
                if score < minScore:
                    minScore = score
                    optimalAction = action

                # update beta
                beta = min(beta, score)

                # if alpha >= beta:
                #     break
                if score < alpha:
                    return score, action

            return minScore, optimalAction

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
        _, action = self.minimax(gameState, self.depth, agentIndex=0)
        return action

    def minimax(self, gameState, depth, agentIndex):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        if agentIndex == 0:

            # find possible actions & assume at least have 1 legal action ()
            legalActions = gameState.getLegalActions(agentIndex)

            maxScore = float('-inf')
            optimalAction = Directions.STOP

            for action in legalActions:
                # find next game state
                nextState = gameState.generateSuccessor(agentIndex, action)

                # update agent index
                newAgentIndex = agentIndex + 1

                # minimax on the child node
                score, _ = self.minimax(nextState, depth, newAgentIndex) # 0+1

                # update max score & optimal action
                if score > maxScore:
                    maxScore = score
                    optimalAction = action

            return maxScore, optimalAction

            # # obtain avg scores when it's not the top node
            # if depth != self.depth:
            #     avgScore = float('-inf')
            #     for action in legalActions:
            #         # find next game state
            #         nextState = gameState.generateSuccessor(agentIndex, action)

            #         # update agent index
            #         newAgentIndex = agentIndex + 1

            #         # minimax on the child node
            #         score, _ = self.minimax(nextState, depth, agentIndex+1) # 0+1

            #         # # update max score & optimal action
            #         # if score > maxScore:
            #         #     maxScore = score
            #         #     optimalAction = action

            #         avgScore += score

            #     avgScore /= len(legalActions)

            #     # only the top node returns action
            #     return avgScore, _
            # else:
            #     maxScore = float('-inf')
            #     optimalAction = Directions.STOP

            #     for action in legalActions:
            #         # find next game state
            #         nextState = gameState.generateSuccessor(agentIndex, action)

            #         # update agent index
            #         newAgentIndex = agentIndex + 1

            #         # minimax on the child node
            #         score, _ = self.minimax(nextState, depth, agentIndex+1) # 0+1

            #         # update max score & optimal action
            #         if score > maxScore:
            #             print('maxScore updated: {}'.format(maxScore))
            #             maxScore = score
            #             optimalAction = action

            #     return maxScore, optimalAction
        else:
            # avgScore = float('inf')
            # optimalAction = Directions.STOP

            legalActions = gameState.getLegalActions(agentIndex)
            successors = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]

            newAgentIndex = (agentIndex+1) % gameState.getNumAgents()
            newDepth = (depth-1) if newAgentIndex == 0 else depth
            scores = [self.minimax(state, newDepth, newAgentIndex)[0] for state in successors]

            print('scores = {}'.format(scores))

            return sum(scores)/len(legalActions), Directions.STOP # STOP is a placeholder, not gonna be used

            # for action in legalActions:
            #     # find next game state
            #     nextGameState = gameState.generateSuccessor(agentIndex, action)

            #     # assume when agent of index != ghost btw.
            #     # Important: A single level of the search is considered
            #     # to be one Pacman move and all the ghosts’ responses,
            #     # so depth 2 search will involve Pacman and each ghost moving twice.
            #     newAgentIndex = (agentIndex+1) % gameState.getNumAgents()
            #     newDepth = (depth-1) if newAgentIndex == 0 else depth
            #     score, _ = self.minimax(nextGameState, newDepth, newAgentIndex)

            #     # if newAgentIndex == 0:
            #     #     score, _ = self.minimax(nextGameState, depth-1, newAgentIndex)
            #     # else:
            #     #     score, _ = self.minimax(nextGameState, depth, newAgentIndex)

            #     avgScore += score

            # avgScore /= len(legalActions)

            # return avgScore, optimalAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """



# Abbreviation
better = betterEvaluationFunction
