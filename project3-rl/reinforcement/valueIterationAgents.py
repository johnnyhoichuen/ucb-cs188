# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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



from os import stat
import re
import mdp, util

from learningAgents import ValueEstimationAgent
import collections


# import sys
# sys.setrecursionlimit(100)



class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    # find the policy
    def runValueIteration(self):
        # V(s) = sum[prob*(immediate R + discounted V(s'))]

        for i in range(0, self.iterations):
            # init new values
            newValues = util.Counter()

            # loop states
            for state in self.mdp.getStates():
                if state == self.mdp.isTerminal(state):
                    newValues[state] = 0
                    break

                # update the value of the state
                # newValues[state] = self.getValue(state)

                maxVal = 1e-9
                for action in self.mdp.getPossibleActions(state):
                    val = self.getQValue(state, action)

                    # --slowest
                    # # this forces max func to run
                    # maxVal = max(val, maxVal)

                    # --medium
                    # this can avoid unnecessary update
                    if val > maxVal:
                        maxVal = val

                    ## simplified
                    # maxVal = max(maxVal, self.getQValue(state, action))
                newValues[state] = maxVal

                # --fastest
                # find out why this fails
                # newValues[state] = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])

            # values & newValues are very close/similar
            # if max(compare all the items in values & newValues) < epsilon:
            #   break

            self.values = newValues

            # get policies
            policies = [self.getPolicy(state) for state in self.mdp.getStates()]




        # while self.iterations != 0:
        #     self.iterations -= 1
        #     new_values = util.Counter() # store new value of a state
        #     update_flag = util.Counter() # store whether a state has been updated
        #     for state in self.mdp.getStates():
        #         best_action = self.computeActionFromValues(state)
        #         if best_action:
        #             new_value = self.computeQValueFromValues(state, best_action)
        #             new_values[state] = new_value
        #             update_flag[state] = 1
        #     for state in self.mdp.getStates():
        #         if update_flag[state]:
        #             self.values[state] = new_values[state]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """

        # if state == self.mdp.isTerminal(state):
        #     print("is terminal state")
        #     return 0

        # array = [1, 1]
        # for action in self.mdp.getPossibleActions(state):
        #     print("qVal: ", self.getQValue(state, action))
        #     array += [self.getQValue(state, action)]

        # if not self.mdp.getPossibleActions(state):
        #     print("empty action list")
        #     print("state: ", state)

        # return max(array)

        # ----------------

        # optValue = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
        # return optValue

        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        if self.mdp.isTerminal(state):
            return 0

        possibleOutcome = self.mdp.getTransitionStatesAndProbs(state, action)
        # print('possibleOutcome: {}'.format(possibleOutcome))

        # verbal version
        sum = .0
        for nextState, prob in possibleOutcome:
            reward = self.mdp.getReward(state, action, nextState)

            # find diff outcomes from 1 action

            # val for 1 action: T * (R + gamma * V(s'))
            val = prob*(reward + self.discount*self.getValue(nextState))
            # print("nextState, prob, reward, val: {}, {}, {}, {}".format(nextState, prob, reward, val))

            sum += val

        return sum

        # simplified version
        # val for 1 action: T * (R + gamma * V(s'))
        # nextState, prob = outcome
        # qValue = sum([outcome[1]*(self.mdp.getReward(state, action, outcome[0]) + self.discount*self.getValue(outcome[0])) \
        #     for outcome in possibleOutcome])

        # return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        if state == self.mdp.isTerminal(state): # or not self.mdp.getPossibleActions(state):
            return None

        optimalAction = ''
        maxVal = float('-inf')
        for action in self.mdp.getPossibleActions(state):
            val = self.getQValue(state, action)

            if val > maxVal:
                maxVal = val
                optimalAction = action

        # print('maxval = {}, optimalAction = {}'.format(maxVal, optimalAction))
        return optimalAction

        # actions = self.mdp.getPossibleActions(state)
        # if not actions:
        #     return None
        # best_action, best_reward = '', -1e9
        # for action in actions:
        #     state_prob = self.mdp.\
        #                  getTransitionStatesAndProbs\
        #                  (state, action)
        #     reward = 0
        #     for new_state, prob in state_prob:
        #         reward += prob * (self.mdp.getReward(state, action, new_state)+\
        #                          self.discount * self.getValue(new_state))
        #     if reward > best_reward:
        #         best_reward = reward
        #         best_action = action
        # return best_action


        # # argmax of Q(s,a) for all actions
        # _, optimalAction = max([(self.getQValue(state, action), action) for action in self.mdp.getPossibleActions(state)])
        # return optimalAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

