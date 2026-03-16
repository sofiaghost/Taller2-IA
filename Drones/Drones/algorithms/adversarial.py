from __future__ import annotations

import math
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
        self.prob = float(prob)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        pass


class RandomAgent(MultiAgentSearchAgent):
    """Agent that chooses a legal action uniformly at random."""

    def get_action(self, state: GameState) -> Directions | None:
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent: drone = MAX, hunters = MIN.
    La profundidad se decrementa cada vez que el turno vuelve al dron.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Retorna la mejor acción para el dron usando Minimax.
        Itera sobre las acciones legales del dron (agent 0), evalúa cada
        sucesor iniciando el turno del primer cazador, y devuelve la acción
        de mayor valor.
        """
        best_action = None
        best_value = -math.inf

        for action in state.get_legal_actions(0):
            successor = state.generate_successor(0, action)
            # Primer cazador actúa después del dron
            value = self._minimax(successor, agent_index=1, depth=self.depth)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _minimax(self, state: GameState, agent_index: int, depth: int) -> float:
        """
        Función recursiva de Minimax.

        Args:
            state:       Estado actual del juego.
            agent_index: Índice del agente que actúa (0 = dron, ≥1 = cazador).
            depth:       Profundidad restante (se decrementa al volver a MAX).

        Returns:
            Valor heurístico o de utilidad del estado.
        """
        # --- Estados terminales ---
        if state.is_win() or state.is_lose():
            return self.evaluation_function(state)

        # --- Límite de profundidad (solo al inicio de turno del dron) ---
        if agent_index == 0 and depth == 0:
            return self.evaluation_function(state)

        if agent_index == 0:
            return self._max_value(state, depth)
        else:
            return self._min_value(state, agent_index, depth)

    def _max_value(self, state: GameState, depth: int) -> float:
        """Nodo MAX: el dron maximiza el valor."""
        value = -math.inf
        for action in state.get_legal_actions(0):
            successor = state.generate_successor(0, action)
            value = max(value, self._minimax(successor, agent_index=1, depth=depth))
        # Si no hay acciones legales, evaluar directamente
        if value == -math.inf:
            return self.evaluation_function(state)
        return value

    def _min_value(self, state: GameState, agent_index: int, depth: int) -> float:
        """
        Nodo MIN: el cazador minimiza el valor.
        Si es el último cazador, el siguiente turno es del dron y la
        profundidad se decrementa.
        """
        num_agents = state.get_num_agents()
        next_agent = (agent_index + 1) % num_agents
        next_depth = depth - 1 if next_agent == 0 else depth

        value = math.inf
        for action in state.get_legal_actions(agent_index):
            successor = state.generate_successor(agent_index, action)
            value = min(value, self._minimax(successor, agent_index=next_agent, depth=next_depth))

        # Si no hay acciones legales, evaluar directamente
        if value == math.inf:
            return self.evaluation_function(state)
        return value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Misma lógica que Minimax con poda.
    MAX: poda estricta cuando value > beta.
    MIN: poda estricta cuando value < alpha.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Retorna la mejor acción para el dron usando poda Alfa-Beta.
        """
        best_action = None
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf

        for action in state.get_legal_actions(0):
            successor = state.generate_successor(0, action)
            value = self._alphabeta(
                successor, agent_index=1, depth=self.depth, alpha=alpha, beta=beta
            )
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)

        return best_action

    def _alphabeta(
        self, state: GameState, agent_index: int, depth: int, alpha: float, beta: float
    ) -> float:
        """
        Función recursiva de Alfa-Beta.

        Args:
            state:       Estado actual.
            agent_index: Agente que actúa.
            depth:       Profundidad restante.
            alpha:       Mejor garantía para MAX (−∞ al inicio).
            beta:        Mejor garantía para MIN (+∞ al inicio).
        """
        if state.is_win() or state.is_lose():
            return self.evaluation_function(state)

        if agent_index == 0 and depth == 0:
            return self.evaluation_function(state)

        if agent_index == 0:
            return self._ab_max_value(state, depth, alpha, beta)
        else:
            return self._ab_min_value(state, agent_index, depth, alpha, beta)

    def _ab_max_value(
        self, state: GameState, depth: int, alpha: float, beta: float
    ) -> float:
        """Nodo MAX con poda β."""
        value = -math.inf
        for action in state.get_legal_actions(0):
            successor = state.generate_successor(0, action)
            value = max(
                value,
                self._alphabeta(successor, agent_index=1, depth=depth, alpha=alpha, beta=beta),
            )
            if value > beta:          # Poda estricta
                return value
            alpha = max(alpha, value)

        if value == -math.inf:
            return self.evaluation_function(state)
        return value

    def _ab_min_value(
        self, state: GameState, agent_index: int, depth: int, alpha: float, beta: float
    ) -> float:
        """Nodo MIN con poda α."""
        num_agents = state.get_num_agents()
        next_agent = (agent_index + 1) % num_agents
        next_depth = depth - 1 if next_agent == 0 else depth

        value = math.inf
        for action in state.get_legal_actions(agent_index):
            successor = state.generate_successor(agent_index, action)
            value = min(
                value,
                self._alphabeta(
                    successor, agent_index=next_agent, depth=next_depth, alpha=alpha, beta=beta
                ),
            )
            if value < alpha:         # Poda estricta
                return value
            beta = min(beta, value)

        if value == math.inf:
            return self.evaluation_function(state)
        return value


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


# ==============================================================================
# BLOQUE DE COMENTARIOS Y REFERENCIAS EN ESPAÑOL (IGNORADOS EN LA LÓGICA ARRIBA)
# ==============================================================================
"""
---(optimiza el algoritmo de python e identifica partes innecesarias o ineficientes. Simplifícalo manteniendo la misma lógica del algoritmo Expectimax)---
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

        return best_action

--- VERSIÓN SIMPLIFICADA REFERENCIADA EN CONVERSACIONES PREVIAS ---

import math
import random
import algorithms.evaluation as evaluation
from world.game import Agent
class MultiAgentSearchAgent(Agent):
def init(self, depth="2", _index=0, prob="0.0"):
self.index = 0
self.depth = int(depth)
self.prob = float(prob)
self.evaluation_function = evaluation.evaluation_function
def get_action(self, state):
# En lugar de un @abstractmethod, simplemente usaría un pass o raise
pass
class RandomAgent(MultiAgentSearchAgent):
def get_action(self, state):
acciones = state.get_legal_actions(self.index)
if len(acciones) > 0:
return random.choice(acciones)
return None
class MinimaxAgent(MultiAgentSearchAgent):
def get_action(self, state):
mejor_accion = None
mejor_valor = -math.inf
for accion in state.get_legal_actions(0):
sucesor = state.generate_successor(0, accion)
valor = self.minimax(sucesor, 1, self.depth)
if valor > mejor_valor:
mejor_valor = valor
mejor_accion = accion
return mejor_accion
def minimax(self, state, agent_index, depth):
# Casos base
if state.is_win() or state.is_lose() or (agent_index == 0 and depth == 0):
return self.evaluation_function(state)
# Turno del Dron (MAX)
if agent_index == 0:
valor = -math.inf
acciones = state.get_legal_actions(0)
if not acciones:
return self.evaluation_function(state)
for accion in acciones:
sucesor = state.generate_successor(0, accion)
valor = max(valor, self.minimax(sucesor, 1, depth))
return valor
# Turno de los Cazadores (MIN)
else:
# Lógica manual para calcular el siguiente turno
siguiente_agente = agent_index + 1
siguiente_profundidad = depth
if siguiente_agente == state.get_num_agents():
siguiente_agente = 0
siguiente_profundidad -= 1
valor = math.inf
acciones = state.get_legal_actions(agent_index)
if not acciones:
return self.evaluation_function(state)
for accion in acciones:
sucesor = state.generate_successor(agent_index, accion)
valor = min(valor, self.minimax(sucesor, siguiente_agente, siguiente_profundidad))
return valor
class AlphaBetaAgent(MultiAgentSearchAgent):
def get_action(self, state):
mejor_accion = None
mejor_valor = -math.inf
alfa = -math.inf
beta = math.inf
for accion in state.get_legal_actions(0):
sucesor = state.generate_successor(0, accion)
valor = self.alfa_beta(sucesor, 1, self.depth, alfa, beta)
if valor > mejor_valor:
mejor_valor = valor
mejor_accion = accion
alfa = max(alfa, mejor_valor)
return mejor_accion
def alfa_beta(self, state, agent_index, depth, alfa, beta):
if state.is_win() or state.is_lose() or (agent_index == 0 and depth == 0):
return self.evaluation_function(state)
# Turno MAX
if agent_index == 0:
valor = -math.inf
acciones = state.get_legal_actions(0)
if not acciones: return self.evaluation_function(state)
for accion in acciones:
sucesor = state.generate_successor(0, accion)
valor = max(valor, self.alfa_beta(sucesor, 1, depth, alfa, beta))
if valor > beta:
return valor
alfa = max(alfa, valor)
return valor
# Turno MIN
else:
siguiente_agente = agent_index + 1
siguiente_profundidad = depth
if siguiente_agente == state.get_num_agents():
siguiente_agente = 0
siguiente_profundidad -= 1
valor = math.inf
acciones = state.get_legal_actions(agent_index)
if not acciones: return self.evaluation_function(state)
for accion in acciones:
sucesor = state.generate_successor(agent_index, accion)
valor = min(valor, self.alfa_beta(sucesor, siguiente_agente, siguiente_profundidad, alfa, beta))
if valor < alfa:
return valor
beta = min(beta, valor)
return valor
class ExpectimaxAgent(MultiAgentSearchAgent):
# ... Mismo principio: combinaría la función chance y max en un solo método ...
pass

Prompt

1. Modularidad e inyección de dependencias (Clean Code):
"Refactoriza la función gigante minimax y alfa_beta. Separa la lógica del
jugador MAX y la del jugador MIN en métodos auxiliares privados como
_max_value y _min_value. Esto hará que el código sea más fácil de leer."
2. Tipado estricto (Type Hinting):
"Añade anotaciones de tipo (type hints) de Python a todas las funciones.
Usa cosas como -> float, state: GameState, y trae TYPE_CHECKING para
evitar importaciones circulares. Esto ayuda al autocompletado en el IDE."
3. Mejora de la herencia y orientación a objetos:
"Haz que la clase base MultiAgentSearchAgent herede de ABC (Abstract
Base Class) y usa el decorador @abstractmethod en get_action. Así
obligamos a que cualquier clase hija implemente ese método sí o sí."
4. Optimización de cálculos matemáticos básicos:
"En el turno de los cazadores, estoy sumando +1 para calcular el siguiente
agente y usando un if para reiniciar a 0. Cambia esto usando el operador
módulo (%) para que sea más elegante (ej: (agent_index + 1) % num_agents). 
"""
