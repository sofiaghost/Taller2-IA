from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_legal_actions(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_num_agents() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % num_agents. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """
        # TODO: Implement your code here
        return None


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Same as Minimax but with alpha-beta pruning.
    MAX node: prune when value > beta (strict).
    MIN node: prune when value < alpha (strict).
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using alpha-beta pruning.

        Tips:
        - Same structure as MinimaxAgent, but with alpha-beta pruning.
        - Alpha: best value MAX can guarantee (initially -inf).
        - Beta: best value MIN can guarantee (initially +inf).
        - MAX node: prune when value > beta (strict inequality, do NOT prune on equality).
        - MIN node: prune when value < alpha (strict inequality, do NOT prune on equality).
        - Update alpha at MAX nodes: alpha = max(alpha, value).
        - Update beta at MIN nodes: beta = min(beta, value).
        - Pass alpha and beta through the recursive calls.
        """
        # TODO: Implement your code here (BONUS)
        return None


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent with a mixed hunter model.

    Each hunter acts randomly with probability self.prob and greedily
    (worst-case / MIN) with probability 1 - self.prob.
    """

    def get_action(self, state: GameState) -> Directions | None:
        num_agents = state.get_num_agents()

        def expectimax(game_state: GameState, depth: int, agent_index: int):
            # Terminal state or depth limit
            if depth == self.depth or game_state.is_win() or game_state.is_lose():
                return self.evaluation_function(game_state)

            legal_actions = game_state.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(game_state)

            next_agent = (agent_index + 1) % num_agents
            next_depth = depth + 1 if next_agent == 0 else depth

            # Drone (MAX)
            if agent_index == 0:
                return max(
                    expectimax(game_state.generate_successor(agent_index, action), next_depth, next_agent)
                    for action in legal_actions
                )

            # Hunter (Chance Node)
            else:
                values = [
                    expectimax(game_state.generate_successor(agent_index, action), next_depth, next_agent)
                    for action in legal_actions
                ]
                min_value = min(values)
                mean_value = sum(values) / len(values)
                return (1 - self.prob) * min_value + self.prob * mean_value

        # Root decision: choose best action
        best_action = None
        best_value = float("-inf")

        for action in state.get_legal_actions(0):
            successor = state.generate_successor(0, action)
            value = expectimax(successor, 0, 1)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action
        """---(optimiza el algoritmo de python e identifica partes innecesarias o ineficientes. Simplifícalo manteniendo la misma lógica del algoritmo Expectimax)---
        def get_action(self, state: GameState) -> Directions | None:
        num_agents = state.get_num_agents()

        def expectimax(game_state: GameState, depth: int, agent_index: int):

            if game_state.is_win() or game_state.is_lose() or depth == self.depth:
                return self.evaluation_function(game_state)

            legal_actions = game_state.get_legal_actions(agent_index)

            next_agent = (agent_index + 1) % num_agents

            if next_agent == 0:
                next_depth = depth + 1  
            else:
                next_depth = depth

            # Dron (MAX)
            if agent_index == 0:
                value = float("-inf")

                for action in legal_actions:
                    successor = game_state.generate_successor(agent_index, action)
                    value = max(
                        value,
                        expectimax(successor, next_depth, next_agent),
                    )

                return value
            if agent_index == 0:

                value = float("-inf")

                for action in legal_actions:

                    successor = game_state.generate_successor(agent_index, action)

                    value = expectimax(successor, next_depth, next_agent)

                    if value > best_value:
                        best_value = value

                return best_value

            # Cazador(aleatorio o MIN)
            else:
                values = []

                for action in legal_actions:
                    successor = game_state.generate_successor(agent_index, action)
                    value = expectimax(successor, next_depth, next_agent)
                    values.append(value)

                min_value = min(values)
                mean_value = sum(values) / len(values)

                result = (1 - self.prob) * min_value + self.prob * mean_value
                return result

        # Nodo raiz, se elige la acción con el valor máximo de expectimax
        best_action = None
        best_value = float("-inf")

        for action in state.get_legal_actions(0):
            successor = state.generate_successor(0, action)
            value = expectimax(successor, 0, 1)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action"""

