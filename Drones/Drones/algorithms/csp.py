from __future__ import annotations

from typing import TYPE_CHECKING
from collections import deque
from copy import deepcopy

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Basic backtracking search without optimizations.

    Tips:
    - An assignment is a dictionary mapping variables to values (e.g. {X1: Cell(1,2), X2: Cell(3,4)}).
    - Use csp.assign(var, value, assignment) to assign a value to a variable.
    - Use csp.unassign(var, assignment) to unassign a variable.
    - Use csp.is_consistent(var, value, assignment) to check if an assignment is consistent with the constraints.
    - Use csp.is_complete(assignment) to check if the assignment is complete (all variables assigned).
    - Use csp.get_unassigned_variables(assignment) to get a list of unassigned variables.
    - Use csp.domains[var] to get the list of possible values for a variable.
    - Use csp.get_neighbors(var) to get the list of variables that share a constraint with var.
    - Add logs to measure how good your implementation is (e.g. number of assignments, backtracks).

    You can find inspiration in the textbook's pseudocode:
    Artificial Intelligence: A Modern Approach (4th Edition) by Russell and Norvig, Chapter 5: Constraint Satisfaction Problems
    """
    def backtrack(assignment):

        if csp.is_complete(assignment):
            return assignment

        var = csp.get_unassigned_variables(assignment)[0]

        for value in csp.domains[var]:

            if csp.is_consistent(var, value, assignment):

                csp.assign(var, value, assignment)

                result = backtrack(assignment)

                if result is not None:
                    return result

                csp.unassign(var, assignment)

        return None
    return backtrack({})


def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with Forward Checking.

    Tips:
    - Forward checking: After assigning a value to a variable, eliminate inconsistent values from
      the domains of unassigned neighbors. If any neighbor's domain becomes empty, backtrack immediately.
    - Save domains before forward checking so you can restore them on backtrack.
    - Use csp.get_neighbors(var) to get variables that share constraints with var.
    - Use csp.is_consistent(neighbor, val, assignment) to check if a value is still consistent.
    - Forward checking reduces the search space by detecting failures earlier than basic backtracking.
    """
    def forward_check(var, assignment, removed):

        for neighbor in csp.get_neighbors(var):

            if neighbor in assignment:
                continue

            for val in csp.domains[neighbor][:]:

                if not csp.is_consistent(neighbor, val, assignment):

                    csp.domains[neighbor].remove(val)
                    removed.append((neighbor, val))

                    if not csp.domains[neighbor]:
                        return False

        return True


    def restore(removed):

        for var, val in removed:
            csp.domains[var].append(val)


    def backtrack(assignment):

        if csp.is_complete(assignment):
            return assignment

        var = csp.get_unassigned_variables(assignment)[0]

        for value in csp.domains[var]:

            if csp.is_consistent(var, value, assignment):

                csp.assign(var, value, assignment)

                removed = []

                if forward_check(var, assignment, removed):

                    result = backtrack(assignment)

                    if result is not None:
                        return result

                restore(removed)

                csp.unassign(var, assignment)

        return None

    return backtrack({})

def revise(csp, Xi, Xj):

    revised = False

    for x in csp.domains[Xi][:]:

        supported = False

        for y in csp.domains[Xj]:

            test_assignment = {Xi: x, Xj: y}

            if csp.is_consistent(Xi, x, test_assignment):
                supported = True
                break

        if not supported:
            csp.domains[Xi].remove(x)
            revised = True

    return revised

def ac3(csp, queue=None):

    if queue is None:
        queue = deque(
            (Xi, Xj)
            for Xi in csp.domains
            for Xj in csp.get_neighbors(Xi)
        )
    else:
        queue = deque(queue)

    while queue:
        Xi, Xj = queue.popleft()

        if revise(csp, Xi, Xj):
            if not csp.domains[Xi]:
                return False

            for Xk in csp.get_neighbors(Xi):
                if Xk != Xj:
                    queue.append((Xk, Xi))

    return True

def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with AC-3 arc consistency.

    Tips:
    - AC-3 enforces arc consistency: for every pair of constrained variables (Xi, Xj), every value
      in Xi's domain must have at least one supporting value in Xj's domain.
    - Run AC-3 before starting backtracking to reduce domains globally.
    - After each assignment, run AC-3 on arcs involving the assigned variable's neighbors.
    - If AC-3 empties any domain, the current assignment is inconsistent - backtrack.
    - You can create helper functions such as:
      - a values_compatible function to check if two variable-value pairs are consistent with the constraints.
      - a revise function that removes unsupported values from one variable's domain.
      - an ac3 function that manages the queue of arcs to check and calls revise.
      - a backtrack function that integrates AC-3 into the search process.
    """
    def backtrack(assignment):

        if csp.is_complete(assignment):
            return assignment

        var = csp.get_unassigned_variables(assignment)[0]

        for value in csp.domains[var][:]:

            if csp.is_consistent(var, value, assignment):

                saved_domains = {v: csp.domains[v][:] for v in csp.domains}

                csp.assign(var, value, assignment)
                csp.domains[var] = [value]

                queue = [(neighbor, var) for neighbor in csp.get_neighbors(var)]

                if ac3(csp, queue):
                    result = backtrack(assignment)
                    if result is not None:
                        return result

                csp.domains = {v: saved_domains[v][:] for v in saved_domains}
                csp.unassign(var, assignment)

        return None

    return backtrack({})



def select_unassigned_variable(csp, assignment):
    unassigned = csp.get_unassigned_variables(assignment)

    return min(
        unassigned,
        key=lambda var: (
            len(csp.domains[var]),
            -sum(1 for n in csp.get_neighbors(var) if n not in assignment)
        )
    )


def order_domain_values(csp, var, assignment):
    return sorted(
        csp.domains[var],
        key=lambda value: csp.get_num_conflicts(var, value, assignment)
    )

def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking with Forward Checking + MRV + LCV.

    Tips:
    - Combine the techniques from backtracking_fc, mrv_heuristic, and lcv_heuristic.
    - MRV (Minimum Remaining Values): Select the unassigned variable with the fewest legal values.
      Tie-break by degree: prefer the variable with the most unassigned neighbors.
    - LCV (Least Constraining Value): When ordering values for a variable, prefer
      values that rule out the fewest choices for neighboring variables.
    - Use csp.get_num_conflicts(var, value, assignment) to count how many values would be ruled out for neighbors if var=value is assigned.
    """
    
    def forward_check(var, assignment):
        for neighbor in csp.get_neighbors(var):
            if neighbor in assignment:
                continue

            for val in csp.domains[neighbor][:]:
                if not csp.is_consistent(neighbor, val, assignment):
                    csp.domains[neighbor].remove(val)

                    if not csp.domains[neighbor]:
                        return False
        return True

    def select_unassigned_variable(assignment):
        unassigned = csp.get_unassigned_variables(assignment)
        return min(
            unassigned,
            key=lambda var: (
                len(csp.domains[var]),
                -sum(1 for n in csp.get_neighbors(var) if n not in assignment)
            )
        )

    def order_domain_values(var, assignment):
        return sorted(
            csp.domains[var],
            key=lambda value: csp.get_num_conflicts(var, value, assignment)
        )

    def backtrack(assignment):
        if csp.is_complete(assignment):
            return assignment

        var = select_unassigned_variable(assignment)

        for value in order_domain_values(var, assignment):
            if csp.is_consistent(var, value, assignment):
                saved_domains = {v: csp.domains[v][:] for v in csp.domains}

                csp.assign(var, value, assignment)

                if forward_check(var, assignment):
                    result = backtrack(assignment)
                    if result is not None:
                        return result

                csp.domains = {v: saved_domains[v][:] for v in saved_domains}
                csp.unassign(var, assignment)

        return None

    return backtrack({})
