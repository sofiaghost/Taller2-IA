from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from world.game_state import GameState


# ---------------------------------------------------------------------------
# Constantes de peso  (ajusta estos valores para calibrar el comportamiento)
# ---------------------------------------------------------------------------
W_DELIVERY      =  8.0   # Peso por distancia al punto de entrega más cercano (negativo)
W_HUNTER_NEAR   = 10.0   # Peso por distancia al cazador más cercano           (positivo)
W_HUNTER_FAR    =  2.0   # Peso por distancia media de TODOS los cazadores     (positivo)
W_PENDING       = 20.0   # Penalización por cada entrega pendiente             (negativo)
W_SCORE         =  1.0   # Peso de la puntuación acumulada del juego           (positivo)
W_URGENCY       = 25.0   # Bonus por entrega alcanzable antes que un cazador   (positivo)

DANGER_DIST_1   = -700.0  # Penalización: cazador a distancia 1 (adyacente)
DANGER_DIST_2   = -250.0  # Penalización: cazador a distancia 2
DANGER_DIST_3   =  -80.0  # Penalización: cazador a distancia 3

# Límite de distancia de cazador para el término lineal (evita inflar el score
# cuando no hay cazadores cerca)
MAX_HUNTER_DIST = 15.0


def evaluation_function(state: GameState) -> float:
    """
    Función de evaluación para estados no terminales.

    Tres objetivos principales:
      1. ACERCARSE a los puntos de entrega pendientes.
      2. ALEJARSE de los cazadores (penalización escalonada + término lineal).
      3. PENALIZAR la falta de progreso (entregas pendientes + score acumulado).

    Returns:
        float en [-999, 999].
    """
    # --- Terminales: utilidad exacta ---
    if state.is_win():
        return 999.0
    if state.is_lose():
        return -999.0

    # Importación local para evitar ciclos en el arranque del módulo
    from algorithms.utils import bfs_distance

    layout          = state.get_layout()
    drone           = state.get_drone_position()
    hunters         = state.get_hunter_positions()
    pending         = state.get_pending_deliveries()   # set / lista de (x, y)
    score           = state.get_score()

    # ==================================================================== #
    #  OBJETIVO 1 – Acercarse al punto de entrega más cercano              #
    # ==================================================================== #
    if pending:
        delivery_dists = [
            bfs_distance(layout, drone, d, hunter_restricted=False)
            for d in pending
        ]
        reachable = [d for d in delivery_dists if d != float("inf")]
        dist_delivery = min(reachable) if reachable else 50.0
    else:
        dist_delivery = 0.0   # Sin entregas pendientes → no penalizar

    delivery_component = -W_DELIVERY * dist_delivery

    # ==================================================================== #
    #  OBJETIVO 2 – Alejarse de los cazadores                              #
    #                                                                       #
    #  Dos sub-términos:                                                    #
    #    2a) Distancia al cazador MÁS CERCANO  (dominante, evitar trampa)  #
    #    2b) Distancia MEDIA de todos los cazadores (evitar cercos)        #
    #    2c) Penalización escalonada por peligro inminente                 #
    # ==================================================================== #
    danger_penalty    = 0.0
    hunter_component  = 0.0

    if hunters:
        hunter_dists = [
            bfs_distance(layout, h, drone, hunter_restricted=True)
            for h in hunters
        ]
        # Cazadores inalcanzables (montaña, niebla…) no suponen amenaza
        reachable_hunters = [d for d in hunter_dists if d != float("inf")]

        if reachable_hunters:
            dist_nearest  = min(reachable_hunters)
            dist_mean     = sum(reachable_hunters) / len(reachable_hunters)

            # 2a + 2b: términos lineales acotados
            capped_nearest = min(dist_nearest, MAX_HUNTER_DIST)
            capped_mean    = min(dist_mean,    MAX_HUNTER_DIST)
            hunter_component = (
                W_HUNTER_NEAR * capped_nearest
                + W_HUNTER_FAR  * capped_mean
            )

            # 2c: penalización escalonada (no lineal, para huir con urgencia)
            if   dist_nearest <= 1:
                danger_penalty = DANGER_DIST_1
            elif dist_nearest <= 2:
                danger_penalty = DANGER_DIST_2
            elif dist_nearest <= 3:
                danger_penalty = DANGER_DIST_3
        else:
            # Todos los cazadores bloqueados → situación segura
            hunter_component = W_HUNTER_NEAR * MAX_HUNTER_DIST

    else:
        # Sin cazadores: máximo valor del componente de seguridad
        hunter_component = W_HUNTER_NEAR * MAX_HUNTER_DIST

    # ==================================================================== #
    #  OBJETIVO 3 – Penalizar falta de progreso                            #
    #                                                                       #
    #  3a) Entregas pendientes: cada una pendiente es progreso no logrado  #
    #  3b) Puntuación acumulada del juego (refleja éxito histórico)        #
    #  3c) Bonus de urgencia: comprometerse con entregas seguras antes que  #
    #      un cazador llegue — evita oscilación por miedo excesivo         #
    # ==================================================================== #
    pending_component = -W_PENDING * (len(pending) if pending else 0)
    score_component   =  W_SCORE   * score

    urgency_bonus = 0.0
    if pending and hunters:
        for delivery in pending:
            d_drone = bfs_distance(layout, drone, delivery, hunter_restricted=False)
            if d_drone == float("inf"):
                continue
            # Distancia del cazador más rápido a esa entrega
            d_hunter_min = min(
                bfs_distance(layout, h, delivery, hunter_restricted=True)
                for h in hunters
            )
            # Si el dron llega antes (con margen de 1 paso), recompensar el compromiso
            if d_drone < d_hunter_min - 1:
                urgency_bonus += W_URGENCY

    progress_component = pending_component + score_component + urgency_bonus

    # ==================================================================== #
    #  Score final                                                          #
    # ==================================================================== #
    raw = (
        delivery_component   # objetivo 1
        + hunter_component   # objetivo 2 (lineal)
        + danger_penalty     # objetivo 2 (escalonado)
        + progress_component # objetivo 3
    )

    return max(-999.0, min(999.0, raw))

"""VERSIÓN SIMPLIFICADA 

from algorithms.utils import bfs_distance
def evaluation_function(state):
# Casos base
if state.is_win():
return 999.0
if state.is_lose():
return -999.0
# Obtener datos del estado
layout = state.get_layout()
dron = state.get_drone_position()
cazadores = state.get_hunter_positions()
entregas = state.get_pending_deliveries()
puntaje = state.get_score()
# --- 1. Distancia a las entregas ---
distancia_entrega = 50.0
if len(entregas) > 0:
distancias = []
for d in entregas:
dist = bfs_distance(layout, dron, d, False)
if dist != float("inf"):
distancias.append(dist)
if len(distancias) > 0:
distancia_entrega = min(distancias)
comp_entrega = -8.0 * distancia_entrega
# --- 2. Distancia a cazadores ---
peligro = 0.0
comp_cazador = 0.0
if len(cazadores) > 0:
dist_cazadores = []
for c in cazadores:
dist = bfs_distance(layout, c, dron, True)
if dist != float("inf"):
dist_cazadores.append(dist)
if len(dist_cazadores) > 0:
mas_cercano = min(dist_cazadores)
promedio = sum(dist_cazadores) / len(dist_cazadores)
# Limitando valores
if mas_cercano > 15.0: mas_cercano = 15.0
if promedio > 15.0: promedio = 15.0
comp_cazador = (10.0 * mas_cercano) + (2.0 * promedio)

if mas_cercano <= 1:
peligro = -700.0
elif mas_cercano <= 2:
peligro = -250.0
elif mas_cercano <= 3:
peligro = -80.0
else:
comp_cazador = 10.0 * 15.0
else:
comp_cazador = 10.0 * 15.0
# --- 3. Progreso ---
comp_pendientes = -20.0 * len(entregas)
comp_puntaje = 1.0 * puntaje
bono_urgencia = 0.0
if len(entregas) > 0 and len(cazadores) > 0:
for e in entregas:
d_dron = bfs_distance(layout, dron, e, False)
if d_dron == float("inf"):
continue
# Bucle manual para encontrar el cazador más cercano a la entrega
min_cazador = 999.0
for c in cazadores:
d_caz = bfs_distance(layout, c, e, True)
if d_caz < min_cazador:
min_cazador = d_caz
if d_dron < min_cazador - 1:
bono_urgencia += 25.0
comp_progreso = comp_pendientes + comp_puntaje + bono_urgencia
# --- Score final ---
total = comp_entrega + comp_cazador + peligro + comp_progreso
# Clamp (limitar el valor) hecho con if/else
if total > 999.0:
return 999.0
elif total < -999.0:
return -999.0
else:
return total

propmt

1. Eliminar "Números Mágicos" (Magic Numbers):
"Extrae todos los números hardcodeados en la función (como 8.0, 15.0, -700.0)
y conviértelos en constantes globales al principio del archivo (ej. W_DELIVERY,
MAX_HUNTER_DIST). Esto facilita la calibración (tunning) de la heurística."
2. Uso de List Comprehensions y Generadores:
"Reemplaza los bucles for que usan .append() para calcular distancias
por 'list comprehensions' (ej. [bfs_distance(...) for c in cazadores]).
Para el cálculo de min_cazador en el bono de urgencia, usa la función
min() pasándole una expresión generadora."
3. Importaciones inteligentes y Tipado:
"Añade anotaciones de tipo (state: GameState -> float). Para evitar
problemas de importación circular con GameState, usa `from typing import
TYPE_CHECKING. Además, mueve la importación de bfs_distance` dentro de la
función si es pesada o causa ciclos en el arranque."
4. Simplificar la lógica de "Clamping":
"Refactoriza el bloque final de if/elif/else que limita el puntaje entre
-999.0 y 999.0 usando las funciones integradas de Python:
return max(-999.0, min(999.0, total))."
"""
