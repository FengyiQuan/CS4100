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


import mdp, util, copy

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            values = copy.deepcopy(self.values)

            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    possible_actions = self.mdp.getPossibleActions(state)
                    optimal = max(self.getQValue(state, action) for action in possible_actions)
                    values[state] = optimal
            self.values = values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = 0
        for next_state, probs in self.mdp.getTransitionStatesAndProbs(state, action):
            value += probs * (self.mdp.getReward(state, action, next_state) + self.discount * self.getValue(next_state))
        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        best_action = None
        max_value = float('-inf')
        for action in self.mdp.getPossibleActions(state):
            value = self.getQValue(state, action)
            if value > max_value:
                max_value = value
                best_action = action
        return best_action

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

    def __init__(self, mdp, discount=0.9, iterations=1000):
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
        states = self.mdp.getStates()
        states_len = len(states)
        for i in range(self.iterations):
            state = states[i % states_len]

            if not self.mdp.isTerminal(state):
                possible_actions = self.mdp.getPossibleActions(state)
                optimal = max(self.getQValue(state, action) for action in possible_actions)
                self.values[state] = optimal


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        predecessors = {}
        queue = util.PriorityQueue()
        states = self.mdp.getStates()

        # Compute predecessors of all states
        for state in states:
            if not self.mdp.isTerminal(state):
                possible_actions = self.mdp.getPossibleActions(state)
                for action in possible_actions:
                    for next_state, probs in self.mdp.getTransitionStatesAndProbs(state, action):
                        if next_state in predecessors:
                            predecessors[next_state].add(state)
                        else:
                            predecessors[next_state] = {state}

        # For each non-terminal state s
        for state in states:
            if not self.mdp.isTerminal(state):
                # Find the absolute value of the difference between the current value of s in self.values and the
                # highest Q-value across all possible actions
                current_value = self.getValue(state)
                possible_actions = self.mdp.getPossibleActions(state)
                optimal = max(self.getQValue(state, action) for action in possible_actions)
                diff = abs(current_value - optimal)
                # Push s into the priority queue with priority -diff
                queue.update(state, -diff)

        for i in range(self.iterations):
            if queue.isEmpty():
                break
            current = queue.pop()
            if not self.mdp.isTerminal(current):
                # Update the value of s (if it is not a terminal state) in self.values
                possible_actions = self.mdp.getPossibleActions(current)
                optimal = max(self.getQValue(current, action) for action in possible_actions)
                self.values[current] = optimal

            # For each predecessor p of s
            for predecessor in predecessors[current]:
                if not self.mdp.isTerminal(predecessor):
                    # Find the absolute value of the difference between the current value of p in self.values and
                    # the highest Q-value across all possible actions from p
                    current_value = self.getValue(predecessor)
                    possible_actions = self.mdp.getPossibleActions(predecessor)
                    optimal = max(self.getQValue(predecessor, action) for action in possible_actions)
                    diff = abs(current_value - optimal)

                    # If diff > theta, push p into the priority queue with priority -diff
                    if diff > self.theta:
                        queue.update(predecessor, -diff)
