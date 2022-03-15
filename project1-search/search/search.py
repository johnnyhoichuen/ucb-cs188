# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from importlib.resources import path
import queue
from tracemalloc import start
from turtle import st
from matplotlib.cbook import Stack
from sqlalchemy import null
from game import Directions
import util
import sys

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # # temp, delete afterwards
    # from game import Directions
    # s = Directions.SOUTH
    # w = Directions.WEST

    # print("\n\n")
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    # # print("Start's cost of action:", problem.getCostOfActions([s]))
    # print("\n\n")

    # start with a node (and add it to stack)
    startNode = problem.getStartState()
    visitedNodes = []
    stack = util.Stack()

    result = recursiveDfs(visitedNodes, stack, startNode, problem)
    # print("\nfinal result of dfs: ", stack.list)
    return result.list


    # example found online:
    #
    # stack = util.Stack()
    # start_pos = problem.getStartState()
    # visited = set()
    # first_node = [(start_pos, None, 1), list([])]
    # stack.push(first_node)
    #
    # while not stack.isEmpty():
    #     cur_node = stack.pop()
    #     print("visiting", cur_node)

    #     position = cur_node[0][0]
    #     directions = cur_node[1]

    #     if problem.isGoalState(position):
    #         print("\nGoal found")
    #         print(directions)
    #         print(len(directions))
    #         print(type(directions))
    #         return directions

    #     if position not in visited:
    #         visited.add(position)
    #         for succ in problem.getSuccessors(position):
    #             # print("\ncur node[1]", directions)

    #             next_node = (succ, directions + [succ[1]])
    #             # print("\nprint next_node", next_node)
    #             # print("\nprint [succ[1]]", [succ[1]])
    #             stack.push(next_node)
    # return None

def recursiveDfs(visitedNodes, stack, node, problem):
    # print("visiting: ", node, " path: ", stack.list)
    visitedNodes.append(node)

    # successor format (pos, direction, cost)
    successors = problem.getSuccessors(node)

    if successors != null:
        # looping available options
        for next in successors:
            if next[0] not in visitedNodes:
                # add direction to path guide
                stack.push(next[1])
                result = recursiveDfs(visitedNodes, stack, next[0], problem)

                # destination found if a Stack is returned
                if type(result) is util.Stack:
                    return result

        if not stack.isEmpty():
            if problem.isGoalState(node):
                print("\nGoal found")
                print(stack.list)
                print(len(stack.list))
                return stack

            # all successors are visited, go back to previous node
            print("all successors were visited at", node, ", back to previous node")
            stack.pop()
            return None

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    queue = util.Queue()
    visitedNodes = []
    startState = problem.getStartState()

    queue.push([startState, []])

    while not queue.isEmpty():
        # nodeWithPath: [pos1, path to pos1]
        state, path = queue.pop()
        # print("\n\nat state: ", state)

        if problem.isGoalState(state):
            print("\nGoal found: {}", path)
            return path

        if not state in visitedNodes:
            # update visited notes
            visitedNodes.append(state)
            # print(state, " appended to visitedNodes")

            # print(problem.getSuccessors(state)) # don't do this more than neccessary
            # or the system will think you expanded the nodes too many times

            # loop thru possible next step
            for suc in problem.getSuccessors(state):
                # print("looping suc: ", suc)

                pathToSuc = (path + [suc[1]])
                # print("pathToSuc: ", pathToSuc)
                queue.push([suc[0], pathToSuc])


    ########
    # backup
    ########
    # while not queue.isEmpty():
    #     # nodeWithPath: [pos1, path to pos1]
    #     nodeWithPath = queue.pop()

    #     # print("nodeWithPath: ", nodeWithPath)
    #     # print("nodeWithPath[0]: ", nodeWithPath[0])
    #     # print("nodeWithPath[0][0]: ", nodeWithPath[0][0])
    #     successors = problem.getSuccessors(nodeWithPath[0][0])
    #     # print("successors: ", successors)

    #     # return is
    #     if problem.isGoalState(nodeWithPath[0]):
    #         print("\nGoal found: ")
    #         print(nodeWithPath[1])
    #         return nodeWithPath[1]

    #     # loop thru possible next step
    #     for suc in successors:
    #         # check if visited
    #         if not suc[0] in visitedNodes:
    #             # update the path with (pos, path to reach pos)
    #             pathToSuc = (nodeWithPath[1] + [suc[1]])
    #             # print("pathToSuc: ", pathToSuc)
    #             queue.push([suc[0], pathToSuc])

    #             # update visited notes
    #             visitedNodes.append(suc[0])


    # fringe = util.Queue()
    # start_pos = problem.getStartState()
    # closed = set()
    # first_node = ((start_pos, None, 1),[])
    # fringe.push(first_node)

    # while not fringe.isEmpty():
    #     cur_node = fringe.pop()
    #     if problem.isGoalState(cur_node[0][0]):
    #         return cur_node[1]

    #     if cur_node[0][0] not in closed:
    #         closed.add(cur_node[0][0])
    #         for succ in problem.getSuccessors(cur_node[0][0]):
    #             next_node = (succ,cur_node[1] + [succ[1]])
    #             print("cur node: ", cur_node[1])
    #             fringe.push(next_node)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    queue = util.PriorityQueue()
    visitedNodes = []
    # startState = problem.getStartState()

    # queue.push([[startState, None, 1], [], 0], 0) # both 0 here represents the priority to this path
    # visitedNodes.append(startState)

    # while not queue.isEmpty():
    #     # print("\nqueue list: ", queue.list)

    #     ########
    #     # nodeWithPath: [[pos1, path to pos1], cost]
    #     ########
    #     nodeWithPath = queue.pop()
    #     # print("\nnode with path: ", nodeWithPath)
    #     # print("nodeWithPath[0]: ", nodeWithPath[0])
    #     # print("nodeWithPath[0][0]: ", nodeWithPath[0][0])
    #     successors = problem.getSuccessors(nodeWithPath[0][0])

    #     if problem.isGoalState(nodeWithPath[0][0]):
    #         print("\nGoal found: ")
    #         print(nodeWithPath[1])
    #         return nodeWithPath[1]

    #     # # loop thru possible next step
    #     # if problem.getSuccessors(nodeWithPath[0][0]) not in visitedNodes:
    #     #     for suc in successors:
    #     #         # # check if visited
    #     #         # if not suc[0] in visitedNodes:
    #     #         # print("looping suc: ", suc[0])

    #     #         # update the path with (pos, path to reach pos)
    #     #         pathToSuc = (nodeWithPath[1] + [suc[1]])
    #     #         # print("pathToSuc: ", pathToSuc)
    #     #         # print("cost of path to current node: ", nodeWithPath[2])
    #     #         # print("cost of successor: ", suc[2])
    #     #         costToPath = nodeWithPath[2] + suc[2]
    #     #         queue.push([suc, pathToSuc, costToPath], costToPath)

    #     #         # print("delete afterwards, cost of path of successor: ", costToPath)
    #     #         # count2 -= 1

    #     #         # update visited notes
    #     #         visitedNodes.append(suc[0])

    #     # loop thru possible next step
    #     for suc in successors:
    #         # # check if visited
    #         if not suc[0] in visitedNodes:
    #             # print("looping suc: ", suc[0])

    #             # update the path with (pos, path to reach pos)
    #             pathToSuc = (nodeWithPath[1] + [suc[1]])
    #             # print("pathToSuc: ", pathToSuc)
    #             # print("cost of path to current node: ", nodeWithPath[2])
    #             # print("cost of successor: ", suc[2])
    #             costToPath = nodeWithPath[2] + suc[2]
    #             queue.push([suc, pathToSuc, costToPath], costToPath)

    #             # print("delete afterwards, cost of path of successor: ", costToPath)
    #             # count2 -= 1

    #             # update visited notes
    #             visitedNodes.append(suc[0])






    # ####
    # # refined codes
    # # passed the maze, did not pass the auto grader
    # ####
    # startNode = Node(problem.getStartState(), [], 0)
    # queue.push(startNode, startNode.costToPath) # both 0 here represents the priority to this path
    # visitedNodes.append(startNode)

    # while not queue.isEmpty():
    #     # print("\nqueue list: ", queue.list)

    #     ########
    #     # currentNode: [[pos1, path to pos1], cost]
    #     ########
    #     currentNode = queue.pop()
    #     # print("current node: ", currentNode.state)
    #     # print("successor of current node: ", problem.getSuccessors(currentNode.state))

    #     if problem.isGoalState(currentNode.state):
    #         print("\nGoal found: ")
    #         print(currentNode.path)
    #         return currentNode.path

    #     # loop thru possible next step
    #     for suc in problem.getSuccessors(currentNode.state):
    #         sucState, sucDir, sucCost = suc
    #         # print("suc: ", sucState, sucDir, sucCost)

    #         # # check if visited
    #         if not sucState in visitedNodes:
    #             # update visited notes
    #             visitedNodes.append(sucState)

    #             # update the path with (pos, path to reach pos)
    #             pathToSuc = (currentNode.path + [sucDir])
    #             # print("pathToSuc: ", pathToSuc)
    #             # print("cost of path to current node: ", nodeWithPath[2])
    #             # print("cost of successor: ", suc[2])
    #             costToPath = currentNode.costToPath + sucCost
    #             queue.push(Node(suc[0], pathToSuc, costToPath), costToPath)

    #             # print("delete afterwards, cost of path of successor: ", costToPath)
    #             # count2 -= 1

    # return list()





    ####
    # refined codes2
    # fixes, add to visited nodes only when
    ####
    startNode = Node(problem.getStartState(), [], 0)
    queue.push(startNode, startNode.costToPath) # both 0 here represents the priority to this path

    while not queue.isEmpty():
        currentNode = queue.pop()

        if problem.isGoalState(currentNode.state):
            print("\nGoal found: ")
            print(currentNode.path)
            return currentNode.path

        if currentNode.state not in visitedNodes:
            # update visited notes
            visitedNodes.append(currentNode.state)

            # loop thru possible next step
            for suc in problem.getSuccessors(currentNode.state):
                sucState, sucDir, sucCost = suc
                pathToSuc = (currentNode.path + [sucDir])
                costToPath = currentNode.costToPath + sucCost
                queue.push(Node(sucState, pathToSuc, costToPath), costToPath)

    return list()





    # working example

    closed = set()
    fringe = util.PriorityQueue()
    fringe.push(ExampleNode(problem.getStartState(), None, None), 0)
    while fringe.isEmpty() is not True:
        node = fringe.pop()
        if problem.isGoalState(node.state) is True:
            actions = list()
            print("node.action: ", node.action)
            while node.action is not None:
                actions.append(node.action)
                print("actions: ", actions)
                node = node.pred
                print("node.pred: ", node.pred)
            actions.reverse()
            print("action.reverse(): ", actions)
            return actions
        if node.state not in closed:
            closed.add(node.state)
            for s in problem.getSuccessors(node.state):
                fringe.push(ExampleNode(s[0], node, s[1], s[2]+node.priority),\
                            s[2]+node.priority)
    return list()

    return None

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    queue = util.PriorityQueue()
    visitedNodes = []
    startNode = Node(problem.getStartState(), [], 0)
    queue.push(startNode, startNode.costToPath) # both 0 here represents the priority to this path

    while not queue.isEmpty():
        currentNode = queue.pop()

        if problem.isGoalState(currentNode.state):
            print("\nGoal found: ")
            print(currentNode.path)
            # print("shortest path: ", queue.heap)
            # print("cost of shortest path: ", currentNode.costToPath)
            return currentNode.path

        if currentNode.state not in visitedNodes:
            # update visited notes
            visitedNodes.append(currentNode.state)

            # loop thru possible next step
            for suc in problem.getSuccessors(currentNode.state):
                sucState, sucDir, sucCost = suc
                pathToSuc = (currentNode.path + [sucDir])
                costToPath = currentNode.costToPath + sucCost
                # print("cost to path: ", costToPath)
                queue.push(Node(sucState, pathToSuc, costToPath), costToPath + heuristic(sucState, problem))

    return list()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

class Node:
    def __init__(self, state, path, costToPath):
        self.state = state
        self.path = path
        self.costToPath = costToPath

    def __repr__(self) -> str:
        return "State: {0}, path: {1}, cost to path: {2}".format(self.state, self.path, self.costToPath)

class ExampleNode:
    def __init__(self, state, pred, action, priority=0):
        self.state = state
        self.pred = pred
        self.action = action
        self.priority = priority
    def __repr__(self):
        return "State: {0}, Action: {1}".format(self.state, self.action)