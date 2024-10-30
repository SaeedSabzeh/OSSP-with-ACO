from local_search.local_search_best_improvement import *
import random
from typing import Tuple

def iterated_local_search(problem, num_tries: int, num_swap: int, init_sol: list, seed: int = 0) -> Tuple[list, float]:
    """
    Perform Iterated Local Search to minimize the makespan for the given problem.
    """
    np.random.seed(seed)
    x = init_sol.copy()
    fx, _ = problem.calculate_makespan(x)  # Only the solution cost is needed
    best_sol, best_makespan = x, fx
    failed_attempts = 0

    while failed_attempts < num_tries:
        y = perturbation(x, num_swap)
        z, fz = local_search(problem, y)
        if fz < best_makespan:
            best_sol, best_makespan = z, fz
            print(f"New best solution found with makespan: {best_makespan}")
            failed_attempts = 0  # Reset failed attempts counter if a better solution is found
        else:
            failed_attempts += 1

    return best_sol, best_makespan



def perturbation(x: list, num_swap: int) -> list:
    """
    Perturb the solution path by swapping tasks in the schedule.
    """
    n = len(x)
    y = x.copy()
    indices = np.random.choice(n, (num_swap, 2), replace=False)
    for i, j in indices:
        y = do_swap(y, i, j)
    return y
