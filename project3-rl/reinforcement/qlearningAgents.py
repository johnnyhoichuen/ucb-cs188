# qlearningAgents.py
# ------------------
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


from numpy import diff
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
from collections import defaultdict

# class QLearningAgent(ReinforcementAgent):
#     """
#       Q-Learning Agent
#       Functions you should fill in:
#         - computeValueFromQValues
#         - computeActionFromQValues
#         - getQValue
#         - getAction
#         - update
#       Instance variables you have access to
#         - self.epsilon (exploration prob)
#         - self.alpha (learning rate)
#         - self.discount (discount rate)
#       Functions you should use
#         - self.getLegalActions(state)
#           which returns legal actions for a state
#     """

#     def __init__(self, **args):
#         "You can initialize Q-values here..."
#         ReinforcementAgent.__init__(self, **args)
#         "*** YOUR CODE HERE ***"
#         self.Q = defaultdict(lambda: defaultdict(float))

#     def getQValue(self, state, action):
#         """
#           Returns Q(state,action)
#           Should return 0.0 if we have never seen a state
#           or the Q node value otherwise
#         """
#         "*** YOUR CODE HERE ***"
#         return self.Q[state][action]

#     def computeValueFromQValues(self, state):
#         """
#           Returns max_action Q(state,action)
#           where the max is over legal actions.  Note that if
#           there are no legal actions, which is the case at the
#           terminal state, you should return a value of 0.0.
#         """
#         "*** YOUR CODE HERE ***"
#         legalActions = self.getLegalActions(state)
#         if not legalActions:
#             return 0.0
#         return max(self.getQValue(state, action) for action in legalActions)

#     def computeActionFromQValues(self, state):
#         """
#           Compute the best action to take in a state.  Note that if there
#           are no legal actions, which is the case at the terminal state,
#           you should return None.
#         """
#         "*** YOUR CODE HERE ***"
#         legalActions = self.getLegalActions(state)
#         if not legalActions:
#             return None
#         # § Break ties randomly
#         value = self.computeValueFromQValues(state)
#         actions = [action for action in legalActions
#                    if self.getQValue(state, action) == value]
#         return random.choice(actions)

#     def getAction(self, state):
#         """
#           Compute the action to take in the current state.  With
#           probability self.epsilon, we should take a random action and
#           take the best policy action otherwise.  Note that if there are
#           no legal actions, which is the case at the terminal state, you
#           should choose None as the action.
#           HINT: You might want to use util.flipCoin(prob)
#           HINT: To pick randomly from a list, use random.choice(list)
#         """
#         "*** YOUR CODE HERE ***"
#         legalActions = self.getLegalActions(state)
#         if random.random() < self.epsilon:
#             return random.choice(legalActions)
#         return self.getPolicy(state)

#     def update(self, state, action, nextState, reward):
#         """
#           The parent class calls this to observe a
#           state = action => nextState and reward transition.
#           You should do your Q-Value update here
#           NOTE: You should never call this function,
#           it will be called on your behalf
#         """
#         "*** YOUR CODE HERE ***"
#         Q = self.Q
#         estimated_return = reward + self.discount * self.computeValueFromQValues(nextState)
#         Q[state][action] += self.alpha * (estimated_return - Q[state][action])

#     def getPolicy(self, state):
#         return self.computeActionFromQValues(state)

#     def getValue(self, state):
#         return self.computeValueFromQValues(state)


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qValues = {}
        # self.qValues = defaultdict(lambda: defaultdict(float))
        # self.qValues = defaultdict(float)

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # # print('getQValue: {}, {}'.format(state, action))
        if not (state, action) in self.qValues:
          self.qValues[(state, action)] = 0.0
          return 0.0
        # # return self.qValues[(state, action)] if (state, action) in self.qValues else 0.0
        return self.qValues[(state, action)]

        # ref code
        # return self.qValues[state][action]

    def setQValue(self, state, action, qVal):
        self.qValues[(state, action)] = qVal
        # print('setting q val: {}'.formatpython autograder.py -q q7(type(self.qValues[(state, action)])))
        # self.qValues[state][action] = qVal

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """

        # print('computeValueFromQValues: {}'.format(state))

        legalActions = self.getLegalActions(state)

        # terminal state
        if not legalActions:
          return 0.0

        # own code
        # # then find max q value
        # maxQ = 0.0
        # for action in legalActions:
        #   # find q val of each action
        #   qVal = self.getQValue(state, action)
        #   if qVal > maxQ:
        #     maxQ = qVal
        # return maxQ

        # ref code
        return max(self.getQValue(state, action) for action in legalActions)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.

          Note: For computeActionFromQValues, you should break ties randomly
          for better behavior. The random.choice() function will help. In a
          particular state, actions that your agent hasn’t seen before still
          have a Q-value, specifically a Q-value of zero, and if all of the
          actions that your agent has seen before have a negative Q-value,
          an unseen action may be optimal.
        """

        # print('computeActionFromQValues: {}'.format(state))

        legalActions = self.getLegalActions(state)

        if not legalActions:
          return None

        # for action in legalActions:
        #   # argmax of a (Q values)
        #   # q = sum of T(R + gamma*Q)
        #   print('st')
        #   qVal

        value = self.computeValueFromQValues(state)

        # actions = []
        # for action in legalActions:
        #   if self.getQValue(state, action) == value:
        #     actions.append(action)
        #     print('action appended to argmax actions!!!')

        actions = [action for action in legalActions if self.getQValue(state, action) == value]

        # print('actions in computeActionFromQValues: {}'.format(actions))
        return random.choice(actions)

        # maxQ = 0.0
        # optimalAction = Directions.STOP
        # for action in legalActions:
        #   # find q val of each action
        #   qVal = self.getQValue(state, action)
        #   if qVal > maxQ:
        #     maxQ = qVal
        #     optimalAction = action

        # # or you can use self.computeValueFromQValues and compare with all the Q val

        # # HOW???
        # # Note: For computeActionFromQValues, you should break ties randomly
        # # for better behavior. The random.choice() function will help.

        # return optimalAction

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """

        # # return random action
        # if util.flipCoin(self.epsilon):
        #   legalActions = self.getLegalActions(state)

        #   if not legalActions:
        #     return Directions.STOP

        #   return random.choice(legalActions)
        # else:
        #   return self.computeActionFromQValues(state)

        legalActions = self.getLegalActions(state)
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """

        # find old q value of the state. If not exist then init
        # if not (state, action) in self.qValues:
        #   self.qValues[(state, action)] = 0.0

        # qVal = (1-alpha)*oldQval + alpha*sample = old qVal + alpha*(sample - oldQval)
        sample = reward + self.discount*(self.computeValueFromQValues(nextState))
        # self.qValues[(state, action)] = (1-self.alpha)*self.computeValueFromQValues(state) + self.alpha*(sample)
        newQval = (1-self.alpha)*self.computeValueFromQValues(state) + self.alpha*(sample)
        self.setQValue(state, action, newQval)

        # self.qValues[(state, action)] += self.alpha * (sample - self.getQValue(state, action))
        # self.qValues[state][action] += self.alpha * (sample - self.getQValue(state, action))


        # print('update: state: {}, action: {}. NextState: {}, reward: {}, new Q val: {}'
        #       .format(state, action, nextState, reward, self.getQValue(state, action)))

        # terminal state
        # if 'exit' in self.getLegalActions(state):
        #   print('Q table: ')
        #   for key in self.qValues:
        #     print('{}: {}'.format(key, self.qValues[key]))
        #   print('\n')

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

        self.count = 0

        # print('weight vector')
        # print(self.weights)

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        # sum = 0
        # # loop through all weights
        # for i, weight in enumerate(self.getWeights()):
        #   # how to get state and action from weight


        #   # featureValue
        #   featureValue = self.featExtractor.getFeatures(state, action)

        #   # add weight * feature value
        #   sum += self.getWeights[(state, action)] * featureValue

        # return sum


        featureValue = self.featExtractor.getFeatures(state, action)
        # print('feature value: ')
        # print(featureValue)

        # * is a dot product here
        return self.weights*featureValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """

        # new q value = (reward + gamma* max q(s',a')) - old q value
        newSample = reward + self.discount*self.computeValueFromQValues(nextState)
        diff = newSample - self.computeValueFromQValues(state)

        # update weights
        # for key, weight in self.getWeights.items():
        #   # state, action = key

          # if value = (state, action): get key
          # self.weights[i] = weight - self.alpha*diff*self
        features = self.featExtractor.getFeatures(state, action)

        # print('\n\n\n\n\n\nfeatures: {}'.format(features))
        # for feature in features:
        #   print(feature[0])
        #   print(feature[1])

        #   self.count += 1

        #   if self.count == 5:
        #     sys.exit("123")

        print('length of feature: {}'.format(len(features)))

        for i in features:
          self.weights[i] += self.alpha * diff * features[i]
        # estimated_return = reward + self.discount * self.computeValueFromQValues(nextState)
        # diff = estimated_return - self.getQValue(state, action)
        # features = self.featExtractor.getFeatures(state, action)
        # weights = self.weights
        # for k in features:
        #     weights[k] += self.alpha * diff * features[k]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging

            print('weight vector: ')
            for i, weight in enumerate(self.getWeights()):
              print(weight)

            pass
