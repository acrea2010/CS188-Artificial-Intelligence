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

import util

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
    return  [s, s]

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
    #util.raiseNotDefined()

    class Node:
        def __init__(self, location, directions, cost):
            self.location = location
            self.directions = directions
            self.cost = cost

    firstnode = Node(problem.getStartState(), [], 0)
    stack = util.Stack()
    stack.push(firstnode)

    visited = []

    while not stack.isEmpty():
        currentnode = stack.pop()
        visited.append(currentnode.location)

        if problem.isGoalState(currentnode.location):
            return currentnode.directions

        for nextnode in problem.getSuccessors(currentnode.location):

            location = nextnode[0]

            if location not in visited:
                directionlist = currentnode.directions[:]
                directionlist.append(nextnode[1])
                cost = currentnode.cost + nextnode[2]
                newnode = Node(location, directionlist, cost)
                stack.push(newnode)

    return []


# def breadthFirstSearch(problem):
#     """Search the shallowest nodes in the search tree first."""
#     "*** YOUR CODE HERE ***"
#
#     class Node:
#         def __init__(self, location, directions, cost):
#             self.location = location
#             self.directions = directions
#             self.cost = cost
#
#     firstnode = Node(problem.getStartState(), [], 0)
#     stack = util.Queue()
#     stack.push(firstnode)
#
#     visited = []
#
#     while not stack.isEmpty():
#         currentnode = stack.pop()
#         visited.append(currentnode.location)
#
#         if problem.isGoalState(currentnode.location):
#             return currentnode.directions
#
#         for nextnode in problem.getSuccessors(currentnode.location):
#
#             location = nextnode[0]
#
#             if location not in visited:
#                 directionlist = currentnode.directions[:]
#                 directionlist.append(nextnode[1])
#                 cost = currentnode.cost + nextnode[2]
#                 newnode = Node(location, directionlist, cost)
#                 stack.push(newnode)
#
#     return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    class Node:
        def __init__(self, state, direction):
            self.state = state
            self.direction = direction

    closed = set()
    fringe = util.Queue()
    startState = Node(problem.getStartState(), [])
    fringe.push(startState)
    while True:
        if fringe.isEmpty():
            return None
        node = fringe.pop()
        if problem.isGoalState(node.state):
            return node.direction
        if node.state not in closed:
            closed.add(node.state)
            for childNode in problem.getSuccessors(node.state):
                directionList = node.direction[:]
                directionList.append(childNode[1])
                newNode = Node(childNode[0],directionList)
                fringe.push(newNode)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    class Node:
        def __init__(self, location, directions, cost):
            self.location = location
            self.directions = directions
            self.cost = cost

    firstnode = Node(problem.getStartState(), [], 0)
    stack = util.PriorityQueue()
    stack.push(firstnode, 0)

    visited = []

    while not stack.isEmpty():
        currentnode = stack.pop()

        if problem.isGoalState(currentnode.location):
            return currentnode.directions

        if currentnode.location not in visited:
            visited.append(currentnode.location)

            for nextnode in problem.getSuccessors(currentnode.location):

                location = nextnode[0]
                directionlist = currentnode.directions[:]
                directionlist.append(nextnode[1])
                cost = currentnode.cost + nextnode[2]
                newnode = Node(location, directionlist, cost)
                stack.push(newnode, cost)

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    class Node:
        def __init__(self, location, directions, cost, heuristic):
            self.location = location
            self.directions = directions
            self.cost = cost
            self.heuristic = heuristic

    firstnode_heuristic = heuristic(problem.getStartState(), problem)
    firstnode = Node(problem.getStartState(), [], 0, firstnode_heuristic)
    stack = util.PriorityQueue()

    stack.push(firstnode, firstnode_heuristic)

    visited = []

    while not stack.isEmpty():
        currentnode = stack.pop()

        if problem.isGoalState(currentnode.location):
            return currentnode.directions

        if currentnode.location not in visited:
            visited.append(currentnode.location)

            for nextnode in problem.getSuccessors(currentnode.location):

                location = nextnode[0]
                directionlist = currentnode.directions[:]
                directionlist.append(nextnode[1])
                heuristic_cost = heuristic(nextnode[0], problem)
                cost = currentnode.cost + nextnode[2]
                totalcost = cost + heuristic_cost
                newnode = Node(location, directionlist, cost, heuristic_cost)
                stack.push(newnode, totalcost)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
